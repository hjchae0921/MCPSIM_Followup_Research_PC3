"""User-requested parallel scheduling; preserve frozen prompts and repair logic."""
from __future__ import annotations

from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager, nullcontext
import fcntl
import os
from pathlib import Path
import shutil
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.kg_version_comparison import campaign, worker
from experiments.kg_version_comparison.api import now, read, write

ORIGINAL_HERE = Path(__file__).resolve().parent
CONTROL = ORIGINAL_HERE / "parallel_control"
RESULTS = campaign.RESULTS
GIB = 1024 ** 3
MAX_WORKFLOWS = 40
MAX_PER_MODEL = 4


def available_memory():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("Cannot determine available memory.")


def child_memory():
    """Map this process's runtime children to their generated source paths."""
    children = set()
    for thread in Path("/proc/self/task").iterdir():
        try:
            children.update((thread / "children").read_text().split())
        except FileNotFoundError:
            pass
    result = {}
    for pid in children:
        try:
            process = Path("/proc") / pid
            args = (process / "cmdline").read_bytes().decode().split("\0")
            if "--source" not in args:
                continue
            source = args[args.index("--source") + 1]
            status = (process / "status").read_text().splitlines()
            rss = next((int(line.split()[1]) * 1024 for line in status if line.startswith("VmRSS:")), 0)
            result[source] = rss
        except (FileNotFoundError, ProcessLookupError, ValueError):
            pass
    return result


class MemoryAdmission:
    """Allow CPU-wide execution, reserving memory while new children start."""
    def __init__(self, memory=available_memory, rss=child_memory):
        self.memory, self.rss = memory, rss
        self.condition = threading.Condition()
        self.reservations = {}
        self.cpu_limit = os.cpu_count() or 1

    def can_start(self, reserve):
        actual = self.rss()
        pending = sum(max(0, size - actual.get(source, 0))
                      for source, size in self.reservations.items())
        return (len(self.reservations) < self.cpu_limit
                and self.memory() - pending >= reserve + GIB)

    @contextmanager
    def slot(self, source, stage, stopped, waiting):
        source = str(Path(source).resolve())
        reserve = 2 * GIB if stage == "solve" else GIB // 4
        started = time.monotonic()
        with self.condition:
            while not self.can_start(reserve):
                if stopped():
                    raise InterruptedError("Stopped while waiting for available memory.")
                waiting()
                self.condition.wait(timeout=0.25)
            if stopped():
                raise InterruptedError("Stopped before code execution.")
            self.reservations[source] = reserve
        try:
            yield time.monotonic() - started
        finally:
            with self.condition:
                self.reservations.pop(source, None)
                self.condition.notify_all()


def handover_ready(task, handover):
    protected = {t["directory"] for t in handover["previous_controller"].get("active", [])}
    if task["directory"] not in protected:
        return True
    folder = Path(task["directory"])
    if (folder / "result.json").exists():
        return True
    state = folder / "state.json"
    return state.exists() and read(state).get("status") == "interrupted"


def old_controller_finished():
    return read(ORIGINAL_HERE / "controller.json").get("status") in ("stopped", "finished")


def main():
    CONTROL.mkdir(exist_ok=True)
    lock = (CONTROL / "controller.lock").open("a")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    protocol = read(ORIGINAL_HERE / "protocol.json")
    campaign.verify(protocol)
    handover = read(CONTROL / "handover.json")
    policy_file = CONTROL / "execution_policy.json"
    if not policy_file.exists():
        write(policy_file, {"effective_at": now(), "reason": "User requested maximum feasible parallel execution.",
            "max_workflows": MAX_WORKFLOWS, "max_workflows_per_model": MAX_PER_MODEL,
            "fixed_two_process_limit_removed": True, "max_execution_processes": os.cpu_count(),
            "memory_headroom_bytes": GIB, "startup_reservation_solve_bytes": 2 * GIB,
            "startup_reservation_other_bytes": GIB // 4,
            "previous_protocol": str(ORIGINAL_HERE / "protocol.json"),
            "scheduler_sha256": campaign.sha(__file__),
            "unchanged": ["cards", "templates", "KG contexts", "model requests", "self-repair", "timeouts"],
            "wall_time_note": "Resumed runs retain previous waiting time; do not pool scheduling epochs for timing comparisons."})
    policy = read(policy_file)
    if policy["scheduler_sha256"] != campaign.sha(__file__):
        raise RuntimeError("The parallel scheduler differs from the recorded execution policy.")
    write(RESULTS / "KG_comparison_execution_policy.json", policy)
    # Only this new process uses the new stop marker and admission policy.
    campaign.HERE = CONTROL
    campaign.RESULTS = CONTROL
    worker.execution_slot = lambda stopped: nullcontext()
    original_execute = worker.execute_stage
    admission = MemoryAdmission()
    stopped = lambda: (CONTROL / "STOP").exists()

    def execute(source, prefix, directory, **kwargs):
        state_path = Path(source).parents[2] / "state.json"
        def phase(status):
            if state_path.exists():
                state = read(state_path)
                if state.get("status") == status:
                    return
                state.update(status=status, updated_at=now())
                write(state_path, state)
        with admission.slot(source, kwargs.get("stage"), stopped,
                            lambda: phase("waiting_for_memory")) as wait_seconds:
            phase("executing")
            write(Path(directory) / "admission.json", {
                "started_at": now(), "memory_wait_seconds": wait_seconds,
                "available_memory_bytes": available_memory(), "policy": str(policy_file)})
            return original_execute(source, prefix, directory, **kwargs)

    worker.execute_stage = execute

    def run_task(task):
        folder = Path(task["directory"])
        with (folder / "parallel_claim.lock").open("a") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            write(folder / "execution_policy.json", policy)
            return worker.run_workflow(task, protocol["model_configs"][task["model"]], protocol)

    pending = [t for t in protocol["tasks"] if not (Path(t["directory"]) / "result.json").exists()]
    running = {}
    last_report = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKFLOWS) as pool:
        while pending or running:
            if stopped():
                pending.clear()
            counts = Counter(t["model"] for t in running.values())
            for task in list(pending):
                if len(running) >= MAX_WORKFLOWS:
                    break
                if (Path(task["directory"]) / "result.json").exists():
                    pending.remove(task)
                    continue
                if counts[task["model"]] >= MAX_PER_MODEL or not handover_ready(task, handover):
                    continue
                running[pool.submit(run_task, task)] = task
                counts[task["model"]] += 1
                pending.remove(task)
            if time.monotonic() - last_report >= 10:
                campaign.verify(protocol)
                summary = campaign.report(protocol)
                write(CONTROL / "controller.json", {"pid": os.getpid(), "status": "running",
                    "updated_at": now(), "active": list(running.values()), "pending": len(pending),
                    "counts": summary["counts"], "execution_processes": len(admission.reservations),
                    "available_memory_bytes": available_memory(), "policy": str(policy_file)})
                if old_controller_finished():
                    for name in ("KG_comparison_summary.json", "KG_comparison_summary.csv"):
                        target = RESULTS / name
                        temporary = target.with_suffix(target.suffix + ".parallel.tmp")
                        shutil.copyfile(CONTROL / name, temporary)
                        temporary.replace(target)
                last_report = time.monotonic()
            completed, _ = wait(running, timeout=1, return_when=FIRST_COMPLETED) if running else ([], [])
            if not running and pending:
                time.sleep(1)
            for future in completed:
                task = running.pop(future)
                try:
                    result = future.result()
                    print({"model": task["model"], "case": task["case"], "graph": task["graph"],
                           "repeat": task["repeat"], "status": result["status"]}, flush=True)
                except Exception as error:
                    write(Path(task["directory"]) / "result.json", {**task, "status": "harness_error",
                        "error": str(error), "traceback": traceback.format_exc(), "finished_at": now()})
    campaign.report(protocol)
    write(CONTROL / "controller.json", {"status": "stopped" if stopped() else "finished", "updated_at": now()})


if __name__ == "__main__":
    main()
