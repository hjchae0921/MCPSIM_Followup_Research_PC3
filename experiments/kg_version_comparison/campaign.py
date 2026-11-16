"""Run the paired six-physics, twenty-repeat KG comparison."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from experiments.kg_version_comparison.api import now, preflight, read, write

PAPER = ROOT / "results_for_paper"
RESULTS = PAPER / "benchmarks/benchmark_results"
INPUTS = HERE / "inputs"
STAGES = ("mesh", "function_space", "weak_form", "boundary_conditions", "solve")
TEMPLATES = ("1_Mesh_Prompt.txt", "2_Functionspace_Prompt.txt", "3_WeakForm_Prompt.txt",
             "4_BC_Prompt.txt", "5_Solve_Prompt.txt")
CARD_FILES = ("1_mesh.json", "2_function_space.json", "3_weak_form.json",
              "4_boundary_conditions.json", "5_solve.json")
MODELS = [
    {"folder": "Qwen 3.8 27B", "id": "qwen/qwen3.8-27b", "transport": "openrouter"},
    {"folder": "Gemma 4 31B", "id": "google/gemma-4-31b-it", "transport": "openrouter"},
    {"folder": "Kimi K 2.5", "id": "moonshotai/kimi-k2.5", "transport": "openrouter"},
    {"folder": "GLM 4.6", "id": "z-ai/glm-4.6", "transport": "openrouter"},
    {"folder": "MiniMax M2.7", "id": "minimax/minimax-m2.7", "transport": "openrouter"},
    {"folder": "Claude Haiku 4.5", "id": "anthropic/claude-haiku-4.5", "transport": "openrouter"},
    {"folder": "Deepseek V3.1", "id": "deepseek/deepseek-chat-v3.1", "transport": "openrouter"},
    {"folder": "GLM 5.3 Flash", "id": "z-ai/glm-5.3-flash", "transport": "openrouter"},
    {"folder": "MiniMax M3.1 Flash", "id": "minimax/minimax-m3.1-flash", "transport": "openrouter"},
    {"folder": "GPT-5 mini", "id": "gpt-5-mini", "transport": "openai"},
]
CASES = [
    ("01_heatsink", "1_Heat Sink"),
    ("02_tensile_specimen", "2_Tensile specimen"),
    ("03_rubber_bushing", "3_Rubber bushing"),
    ("04_I_beam", "4_Thermoelastic I-beam"),
    ("05_busbar", "5_Joule heating busbar"),
    ("06_piezo_cantilever", "6_Piezoelectric cantilever"),
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tasks():
    # Alternate arm order between repeats to reduce ordering effects.
    return [dict(model=m["folder"], case=case, case_folder=folder, graph=graph, repeat=repeat,
                 directory=str(RESULTS / m["folder"] / graph / folder / f"run_{repeat:02d}"))
            for repeat in range(1, 21) for case, folder in CASES for m in MODELS
            for graph in (("KG_ver1", "KG_ver2") if repeat % 2 else ("KG_ver2", "KG_ver1"))]


def freeze():
    destination = HERE / "protocol.json"
    if destination.exists():
        verify(read(destination))
        return read(destination)
    setting = RESULTS / "setting.txt"
    assert "Temperature: 0" in setting.read_text()
    copied = {}

    def copy(source, target):
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied[str(source)] = sha(source)

    copy(setting, INPUTS / "setting.txt")
    for filename in TEMPLATES:
        copy(ROOT / "prompt_templates/no_card_descriptions" / filename, INPUTS / "templates" / filename)
    for case, _ in CASES:
        source, target = PAPER / "benchmarks" / case, INPUTS / "cases" / case
        copy(source / "model.msh", target / "model.msh")
        copy(source / "prompt.txt", target / "prompt.txt")
        for stage, filename in zip(STAGES, CARD_FILES):
            source_card = source / "cards" / filename
            card = read(source_card)
            copied[str(source_card)] = sha(source_card)
            if stage == "mesh":
                card["mesh"]["path"] = str((target / "model.msh").resolve())
            write(target / "cards" / filename, card)
    for version in (1, 2):
        graph = PAPER / "knowledge_graph" / f"graph_ver{version}" / "graph"
        config = read(graph / "retrieval_config.json")
        copy(graph / "retrieval_config.json", INPUTS / f"KG_ver{version}" / "retrieval_config.json")
        for stage, filename in zip(STAGES, TEMPLATES):
            text = (INPUTS / "templates" / filename).read_text()
            queries = re.findall(r"^\d+\. (.+)$", text, re.M)
            assert queries == config["procedure_queries"][stage], (version, stage)
            copy(graph / "retrieved_contexts" / f"{stage}.txt", INPUTS / f"KG_ver{version}" / f"{stage}.txt")
        for filename in ("coder_graph.json", "schema.json"):
            if (graph / filename).exists():
                copied[str(graph / filename)] = sha(graph / filename)
    frozen = {str(p): sha(p) for p in INPUTS.rglob("*") if p.is_file()}
    for path in (HERE / "campaign.py", HERE / "worker.py", HERE / "api.py",
                 ROOT / "codegen/cards.py", ROOT / "codegen/runtime.py",
                 ROOT / "experiments/source_type_graph/repair_feedback.py"):
        frozen[str(path)] = sha(path)
    model_configs = {}
    for model in MODELS:
        evidence_path = HERE / "preflight" / model["folder"] / "preflight.json"
        evidence = read(evidence_path) if evidence_path.exists() else {}
        if evidence.get("status") != "ready":
            raise RuntimeError("Finish the model preflight before freezing: " + model["folder"])
        model_configs[model["folder"]] = evidence["config"]
        frozen[str(evidence_path)] = sha(evidence_path)
    protocol = {
        "created_at": now(), "models": MODELS, "cases": CASES, "repeats": 20,
        "workflows_per_model": 240, "planned_workflows": 2400, "tasks": tasks(),
        "temperature": 0, "reasoning": "low, else lowest accepted or disabled",
        "model_configs": model_configs,
        "approved_temperature_exception": "GPT-5 mini only: omit temperature (effective default 1), low reasoning; approved by user.",
        "max_repairs_per_stage": 5, "initial_attempts_per_stage": 1,
        "repair_policy": "Current stage only; original prompt, last code and runtime error/traceback. Earlier accepted code unchanged.",
        "evaluation_mode": "execution_only", "physics_correctness_asserted": False,
        "retrieval": "Cached top-3 per procedure; basic 1-hop plus HAS_ARGUMENT/HAS_RETURN/HAS_TYPE for directly connected FEEDS/IS_VALID_INPUT_TO APIs.",
        "paired_control": "All templates, cards, mesh paths and repair rules identical across arms. Only KG context differs.",
        "mesh_path_adjustment": "Only obsolete card mesh paths replaced with the same frozen existing mesh path in both arms; physical values unchanged.",
        "reasoning_recording": "Save all API-returned reasoning/reasoning_details. Unreturned internal reasoning cannot be recovered.",
        "web_search": False, "tools": False, "api_concurrency": 4,
        "one_workflow_per_model": True, "execution_slots_shared_with_existing_studies": 2,
        "api_timeout_seconds": 900, "runtime_timeout_seconds": 1200,
        "hyperelastic_runtime_timeout_seconds": 3600,
        "api_errors_are_not_coding_failures": True,
        "http_retry_policy": "At most 3 HTTP submissions for native 429 rejections without completion; 60s minimum between rejections. Unknown submissions require reconciliation.",
        "source_sha256": copied, "frozen_sha256": frozen,
        "result_directory": str(RESULTS),
    }
    write(destination, protocol)
    write(RESULTS / "KG_comparison_manifest.json", protocol)
    for model in MODELS:
        for graph in ("KG_ver1", "KG_ver2"):
            arm = RESULTS / model["folder"] / graph
            for _, folder in CASES:
                for repeat in range(1, 21):
                    (arm / folder / f"run_{repeat:02d}").mkdir(parents=True, exist_ok=True)
            write(arm / "settings.json", {
                "model": model, "graph": graph, "repeats_per_case": 20,
                "workflows": 120, "input_snapshot": str(INPUTS),
                "protocol": str(destination), "preflight": str(HERE / "preflight" / model["folder"] / "preflight.json"),
                "evaluation_mode": "execution_only", "max_repairs_per_stage": 5,
                "model_config": model_configs[model["folder"]]})
    return protocol


def verify(protocol):
    changed = [str(path) for path, digest in protocol["frozen_sha256"].items()
               if not Path(path).is_file() or sha(path) != digest]
    if changed:
        raise RuntimeError("Frozen experiment inputs or implementation changed: " + str(changed))


def preflights(gpt_temperature_exception=False):
    def probe(model):
        result = preflight(model, HERE / "preflight" / model["folder"],
                           gpt_temperature_exception=gpt_temperature_exception)
        write(RESULTS / model["folder"] / "preflight.json", result)
        print(json.dumps({"model": model["folder"], "status": result["status"],
                          "reasoning": result.get("reasoning_label"),
                          "temperature": result.get("effective_temperature")}), flush=True)
        return result
    with ThreadPoolExecutor(max_workers=4) as pool:
        return list(pool.map(probe, MODELS))


def report(protocol):
    rows = []
    counts = Counter()
    for task in protocol["tasks"]:
        folder = Path(task["directory"])
        result_file = folder / "result.json"
        state_file = folder / "state.json"
        result = read(result_file) if result_file.exists() else read(state_file) if state_file.exists() else {}
        row = {k: task[k] for k in ("model", "graph", "case", "repeat")}
        row.update(status=result.get("status", "queued"),
                   stage=result.get("stage"), wall_seconds=result.get("wall_seconds"),
                   api_response_seconds=result.get("api_response_seconds"),
                   api_error_seconds=result.get("api_error_seconds"),
                   input_tokens=result.get("input_tokens"), output_tokens=result.get("output_tokens"),
                   cached_tokens=result.get("cached_tokens"), reasoning_tokens=result.get("reasoning_tokens"),
                   cost_usd=result.get("cost_usd"), repair_calls=result.get("repair_calls"))
        rows.append(row)
        counts[row["status"]] += 1
    value = {"updated_at": now(), "planned": len(rows), "counts": dict(counts), "rows": rows}
    write(RESULTS / "KG_comparison_summary.json", value)
    with (RESULTS / "KG_comparison_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return value


def run(protocol):
    from experiments.kg_version_comparison.worker import run_workflow

    lock = (HERE / "controller.lock").open("a")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    verify(protocol)
    write(HERE / "controller.json", {"pid": os.getpid(), "started_at": now(), "status": "running"})
    pending = [task for task in protocol["tasks"] if not (Path(task["directory"]) / "result.json").exists()]
    running = {}
    last_report = 0
    with ThreadPoolExecutor(max_workers=protocol["api_concurrency"]) as pool:
        while pending or running:
            if (HERE / "STOP").exists():
                pending.clear()
            active = {task["model"] for task in running.values()}
            for task in list(pending):
                if len(running) >= protocol["api_concurrency"]:
                    break
                if task["model"] in active:
                    continue
                verify(protocol)
                running[pool.submit(run_workflow, task, protocol["model_configs"][task["model"]], protocol)] = task
                active.add(task["model"])
                pending.remove(task)
            if time.monotonic() - last_report > 10:
                summary = report(protocol)
                write(HERE / "controller.json", {"pid": os.getpid(), "status": "running", "updated_at": now(),
                    "active": list(running.values()), "pending": len(pending), "counts": summary["counts"]})
                last_report = time.monotonic()
            if not running:
                if pending:
                    write(HERE / "controller.json", {"pid": os.getpid(), "status": "waiting_for_model_configuration",
                        "updated_at": now(), "pending": len(pending)})
                    time.sleep(5)
                    continue
                break
            completed, _ = wait(running, timeout=1, return_when=FIRST_COMPLETED)
            for future in completed:
                task = running.pop(future)
                try:
                    result = future.result()
                    print(json.dumps({**{k: task[k] for k in ("model", "graph", "case", "repeat")},
                                      "status": result["status"]}), flush=True)
                except Exception as error:
                    write(Path(task["directory"]) / "result.json", {
                        **task, "status": "harness_error", "error": str(error),
                        "traceback": traceback.format_exc(), "finished_at": now()})
    report(protocol)
    write(HERE / "controller.json", {"status": "stopped" if (HERE / "STOP").exists() else "finished", "updated_at": now()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "preflight", "run", "report"))
    parser.add_argument("--gpt-temperature-default", action="store_true")
    args = parser.parse_args()
    if args.action == "preflight":
        preflights(args.gpt_temperature_default)
    elif args.action == "prepare":
        print(json.dumps({"planned_workflows": freeze()["planned_workflows"]}))
    elif args.action == "run":
        run(read(HERE / "protocol.json"))
    else:
        print(json.dumps(report(read(HERE / "protocol.json"))["counts"]))
