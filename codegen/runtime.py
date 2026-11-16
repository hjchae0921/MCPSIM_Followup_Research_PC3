"""Execute generated stages without card, reference, or physics validation."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import types


class ExecutionInterrupted(RuntimeError):
    """The user stopped an executing stage."""


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def child(prefix, source):
    """Use a real main-module namespace shared by prefix and current stage."""
    source, prefix = Path(source).resolve(), Path(prefix).resolve()
    module = types.ModuleType("__main__")
    module.__dict__.update(__package__=None, __spec__=None)
    sys.modules["__main__"] = module
    sys.argv = [str(source)]
    sys.path.insert(0, str(source.parent))
    for path in (prefix, source):
        module.__file__ = str(path)
        exec(compile(path.read_text(), str(path), "exec"), module.__dict__)


def execute_stage(source, prefix, directory, *, python=None, timeout=1200,
                  stopped=None, environment=None, stage=None):
    """Return actual process outcome; a child crash is a repairable runtime error.

    There is no benchmark snapshot argument, no mathematical check and no FEM
    monkeypatching. Successful exit does not establish physical correctness.
    """
    source, directory = Path(source).resolve(), Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    stopped = stopped or (lambda: False)
    if stopped():
        raise ExecutionInterrupted("Execution was stopped before submission.")
    prefix_file = directory / "prefix.py"
    prefix_text = prefix + "\n"
    inputs = {"source_sha256": digest(source), "prefix_sha256": hashlib.sha256(prefix_text.encode()).hexdigest(),
              "executor_sha256": digest(__file__), "python": str(python or sys.executable),
              "timeout_seconds": timeout, "evaluation_mode": "execution_only"}
    input_file, result_file = directory / "execution_inputs.json", directory / "validation.json"
    if result_file.exists():
        if not input_file.exists() or json.loads(input_file.read_text()) != inputs:
            raise ValueError("Cached process outcome belongs to different source or execution settings.")
        return json.loads(result_file.read_text())
    prefix_file.write_text(prefix_text)
    write(input_file, inputs)
    env = dict(os.environ if environment is None else environment)
    for key in ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "CODER_API_KEY"):
        env.pop(key, None)
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               PYTHONDONTWRITEBYTECODE="1", PYTHONFAULTHANDLER="1",
               XDG_CACHE_HOME=str(directory / "cache"))
    command = [str(python or sys.executable), "-B", str(Path(__file__).resolve()),
               "--prefix", str(prefix_file), "--source", str(source)]
    started = time.monotonic()
    timed_out, interrupted = False, False
    with (directory / "stdout.txt").open("w") as stdout, (directory / "stderr.txt").open("w") as stderr:
        process = subprocess.Popen(command, cwd=directory, env=env, stdout=stdout,
                                   stderr=stderr, start_new_session=True)
        while process.poll() is None:
            interrupted = bool(stopped())
            timed_out = time.monotonic() - started >= timeout
            if interrupted or timed_out:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
                break
            time.sleep(0.05)
    elapsed = time.monotonic() - started
    if interrupted or stopped():
        write(directory / "interrupted.json", {"process_returncode": process.returncode,
                                               "execution_seconds": elapsed})
        raise ExecutionInterrupted("Execution stopped; output and exit status were preserved.")
    passed = process.returncode == 0 and not timed_out
    error = None
    if not passed:
        detail = (directory / "stderr.txt").read_text().strip() or (directory / "stdout.txt").read_text().strip()
        error = (f"Generated stage execution exceeded {timeout} seconds." if timed_out else
                 f"Generated stage process exited with code {process.returncode}.")
        if detail:
            error += "\n" + detail[-16000:]
    report = {"evaluation_mode": "execution_only", "stage": stage, "runtime_pass": passed,
              "semantic_pass": None, "verified_correct": None, "field_correct": None,
              "reference_comparison_performed": False, "card_contract_grading_performed": False,
              "process_returncode": process.returncode, "execution_seconds": elapsed,
              "timed_out": timed_out, "error": error,
              "repair_feedback": None if passed else {"kind": "runtime_error", "stage": stage,
                                                       "diagnostic": error}}
    write(result_file, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    child(args.prefix, args.source)
