"""Unmodified five-stage prompts with execution-only, current-stage self-repair."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import re
import sys
import time

from codegen.cards import render
from codegen.runtime import execute_stage, ExecutionInterrupted
from experiments.benchlm_model_screen.support import execution_slot
from experiments.kg_version_comparison.api import call, now, read, write, ProviderFailure
from experiments.source_type_graph.repair_feedback import compose


def code_text(response):
    message = ((response.get("choices") or [{}])[0].get("message") or {})
    content = message.get("content") or ""
    if not isinstance(content, str):
        raise ProviderFailure("The model returned no text code content.")
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", content, re.S)
    return ("\n\n".join(blocks) if blocks else content).strip() + "\n"


def totals(folder):
    receipts = [read(p) for p in Path(folder).glob("[1-5]_*/attempt_*/submissions/submission_*/receipt.json")]
    successes = [r for r in receipts if r.get("response")]
    usage = [r["response"].get("usage") or {} for r in successes]

    def sum_known(values):
        return sum(values) if values and all(v is not None for v in values) else None

    return {
        "api_response_seconds": sum(r.get("seconds", 0) for r in successes),
        "api_error_seconds": sum(r.get("seconds", 0) for r in receipts if not r.get("response")),
        "api_submissions": len(receipts), "successful_api_responses": len(successes),
        "input_tokens": sum_known([u.get("prompt_tokens") for u in usage]),
        "output_tokens": sum_known([u.get("completion_tokens") for u in usage]),
        "cached_tokens": sum_known([(u.get("prompt_tokens_details") or {}).get("cached_tokens") for u in usage]),
        "reasoning_tokens": sum_known([(u.get("completion_tokens_details") or {}).get("reasoning_tokens") for u in usage]),
        "cost_usd": sum_known([u.get("cost") for u in usage]),
        "reasoning_tokens_are_subset_of_output": True,
        "cost_basis": "API-reported cost; absent values remain null.",
    }


def run_workflow(task, config, protocol):
    from experiments.kg_version_comparison.campaign import HERE, INPUTS, STAGES, TEMPLATES, CARD_FILES

    folder = Path(task["directory"])
    result_file = folder / "result.json"
    if result_file.exists():
        return read(result_file)
    folder.mkdir(parents=True, exist_ok=True)
    stopped = lambda: (HERE / "STOP").exists()
    started = time.monotonic()
    prior = read(folder / "state.json") if (folder / "state.json").exists() else {}
    started_at = prior.get("started_at") or task.get("started_at") or now()
    prefix = ""
    accepted = []
    repairs = 0
    metadata = {**task, "model_config": config, "started_at": started_at,
                "evaluation_mode": "execution_only", "physics_correctness_asserted": False,
                "max_repairs_per_stage": 5}
    write(folder / "run.json", metadata)

    def state(status, stage=None, attempt=None, **extra):
        value = {**metadata, "status": status, "stage": stage, "attempt": attempt,
                 "updated_at": now(),
                 "wall_seconds": (datetime.now(timezone.utc) - datetime.fromisoformat(started_at)).total_seconds(),
                 "current_invocation_seconds": time.monotonic() - started,
                 "repair_calls": repairs, **totals(folder), **extra}
        write(folder / "state.json", value)
        return value

    def finish(status, stage=None, **extra):
        value = state(status, stage, finished_at=now(), **extra)
        write(result_file, value)
        return value

    try:
        for number, (stage, template_file, card_file) in enumerate(zip(STAGES, TEMPLATES, CARD_FILES), 1):
            if stopped():
                raise InterruptedError("Study stopped between stages.")
            stage_folder = folder / f"{number}_{stage}"
            stage_folder.mkdir(exist_ok=True)
            card = read(INPUTS / "cases" / task["case"] / "cards" / card_file)
            graph = (INPUTS / task["graph"] / f"{stage}.txt").read_text()
            template = (INPUTS / "templates" / template_file).read_text()
            system, original, full = render(template, card, graph, prefix)
            (stage_folder / "initial_prompt.txt").write_text(full)
            write(stage_folder / "card.json", card)
            current = original
            for attempt in range(protocol["max_repairs_per_stage"] + 1):
                attempt_folder = stage_folder / f"attempt_{attempt:02d}"
                attempt_folder.mkdir(exist_ok=True)
                state("generating", stage, attempt)
                (attempt_folder / "prompt.txt").write_text(current)
                response = call(config, [{"role": "system", "content": system},
                                         {"role": "user", "content": current}], attempt_folder, stopped=stopped)
                code = code_text(response)
                source = attempt_folder / "code.py"
                if source.exists() and source.read_text() != code:
                    raise RuntimeError("A generated code artifact differs from its saved response.")
                source.write_text(code)
                if not code.strip():
                    return finish("output_limit" if (response.get("choices") or [{}])[0].get("finish_reason") == "length"
                                  else "empty_response", stage,
                                  error="No visible code was returned; raw API response and usage retained.")
                state("waiting_for_execution_slot", stage, attempt)
                with execution_slot(stopped):
                    state("executing", stage, attempt)
                    validation = execute_stage(source, prefix, attempt_folder / "execution", python=sys.executable,
                        timeout=(protocol["hyperelastic_runtime_timeout_seconds"] if task["case"] == "03_rubber_bushing"
                                 else protocol["runtime_timeout_seconds"]), stopped=stopped, stage=stage)
                if validation["runtime_pass"]:
                    prefix += f"\n# Stage {number}: {stage}\n" + code
                    accepted.append(str(source.relative_to(folder)))
                    write(stage_folder / "accepted.json", {"attempt": attempt, "source": str(source),
                        "source_relative": str(source.relative_to(folder)), "execution_only": True})
                    (folder / "cumulative_code.py").write_text(prefix)
                    break
                error = validation["error"]
                diagnostic = {"kind": "runtime_error", "stage": stage, "diagnostic": error}
                write(attempt_folder / "repair_feedback.json", diagnostic)
                if attempt == protocol["max_repairs_per_stage"]:
                    return finish("execution_failed", stage, error=error)
                repairs += 1
                current = compose(original, code, diagnostic)
            else:
                raise AssertionError("Stage attempt loop did not terminate.")
        # Replay the accepted stage sources with their original compile boundaries.
        # This also preserves legal per-stage future imports without rewriting model code.
        (folder / "final.py").write_text(
            '"""Replay the accepted generated stages; cumulative_code.py contains their verbatim concatenation."""\n'
            "from pathlib import Path\n"
            "_run_directory = Path(__file__).resolve().parent\n"
            "for _relative in " + repr(accepted) + ":\n"
            "    _source_path = _run_directory / _relative\n"
            "    exec(compile(_source_path.read_text(), str(_source_path), 'exec'), globals())\n")
        return finish("execution_success", "solve", accepted_sources=accepted,
                      final_source="final.py", cumulative_source="cumulative_code.py")
    except ProviderFailure as error:
        return finish("provider_error", locals().get("stage"), error=str(error))
    except (InterruptedError, ExecutionInterrupted) as error:
        return state("interrupted", locals().get("stage"), error=str(error))
