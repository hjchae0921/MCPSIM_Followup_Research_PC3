"""Offline regression tests: no API requests and no changes to existing outputs."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
import importlib.util
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import threading
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from task_registry import SCHEMA, TaskRegistry
from experiments.kg_version_comparison.api import read, write


def jobs(count=3):
    return [{'model': 'Fixture model', 'graph': 'KG', 'case': '01_heatsink',
             'case_folder': 'Heatsink'} for _ in range(count)]


class ActualStartTests(unittest.TestCase):
    def test_driver_passes_the_start_hook_and_live_registry_to_the_controller(self):
        import run as driver
        with TemporaryDirectory() as tmp, ExitStack() as stack:
            home = Path(tmp)
            for obj, names in [(driver.base, ('HERE', 'INPUTS', 'RESULTS', 'MAX_ACTIVE', 'MAX_PER_MODEL', 'report')),
                               (driver.extension, ('HERE', 'postprocess')),
                               (driver.continuation, ('HERE',)),
                               (driver.worker, ('execute_stage', 'run_workflow'))]:
                for name in names:
                    stack.enter_context(patch.object(obj, name, getattr(obj, name)))
            for name, value in [('HOME', home), ('INPUTS', home / 'inputs'), ('POST', home / 'postprocess'), ('REGISTRY', None)]:
                stack.enter_context(patch.object(driver, name, value))
            stack.enter_context(patch.object(driver, 'environment', return_value={}))
            stack.enter_context(patch.object(driver, 'prepare_inputs', return_value=None))
            def preflight(model, directory, **kwargs):
                self.assertTrue(kwargs['strict_low'])
                return {'status': 'ready', 'config': {'temperature': None if model.get('omit_temperature') else 0,
                                                    'reasoning': {'effort': 'low'}}}
            stack.enter_context(patch.object(driver, 'preflight', side_effect=preflight))
            invoked = []
            def controller(protocol, *, prepare_task):
                self.assertEqual(protocol['identity_schema'], SCHEMA)
                self.assertTrue(all('directory' not in task for task in protocol['tasks']))
                task = protocol['tasks'][0]
                self.assertIs(task, driver.registry().entries[0])
                with ThreadPoolExecutor(max_workers=1) as pool:
                    pool.submit(prepare_task, task).result()
                self.assertEqual(Path(task['directory']).name, task['experiment_timestamp'])
                invoked.append(task['started_at'])
            stack.enter_context(patch.object(driver.continuation, 'resume_model', side_effect=controller))
            driver.run(2, 1)
            self.assertEqual(len(invoked), 1)
            self.assertEqual(read(home / 'protocol.json')['identity_schema'], SCHEMA)

    def test_preparing_and_reporting_never_allocate_a_start(self):
        import run as driver
        with TemporaryDirectory() as tmp:
            home = Path(tmp) / 'campaign'
            with patch.object(driver, 'HOME', home), patch.object(driver, 'POST', home / 'postprocess'), patch.object(driver, 'REGISTRY', None):
                queued = driver.tasks()
                registration = driver.registry()
                registration.clock = lambda: self.fail('Prepare/report called the execution clock')
                report = driver.report()
                self.assertEqual(len(queued), driver.PLAN['planned_workflows'])
                self.assertEqual(report['completed'], 0)
                self.assertTrue(all('experiment_timestamp' not in task and 'directory' not in task and 'started_at' not in task for task in queued))
                self.assertFalse((home / 'records').exists())

    def test_time_is_captured_in_the_execution_thread_after_waiting(self):
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            actual = datetime(2026, 11, 19, 6, 45, 12, 123456, tzinfo=timezone.utc)
            main_thread = threading.get_ident()
            calls = []
            def clock():
                self.assertNotEqual(threading.get_ident(), main_thread)
                calls.append(True)
                return actual
            registration = TaskRegistry(home / 'task_plan.json', home, jobs, clock=clock)
            self.assertEqual(calls, [])
            self.assertFalse((home / 'records').exists())
            with ThreadPoolExecutor(max_workers=1) as pool:
                task = pool.submit(registration.start, registration.entries[0]).result()
            self.assertEqual(task['started_at'], actual.isoformat())
            self.assertEqual(task['experiment_timestamp'], '2026_11_19_15_45_12_123456')
            self.assertEqual(Path(task['directory']).name, task['experiment_timestamp'])
            self.assertEqual(len(calls), 1)

    def test_parallel_starts_are_atomic_and_use_real_clock_values(self):
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            observed = []
            def clock():
                value = datetime.now(timezone.utc)
                observed.append(value)
                return value
            registration = TaskRegistry(home / 'task_plan.json', home, lambda: jobs(12), clock=clock)
            with ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(registration.start, registration.entries))
            self.assertEqual(len({task['directory'] for task in registration.entries}), 12)
            self.assertTrue(all(datetime.fromisoformat(task['started_at']) in observed for task in registration.entries))
            self.assertEqual(read(home / 'task_plan.json')['tasks'], registration.snapshot())

    def test_resume_keeps_original_start_even_before_first_record_was_written(self):
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            actual = datetime(2026, 11, 19, 1, 2, 3, 456789, tzinfo=timezone.utc)
            original = TaskRegistry(home / 'task_plan.json', home, jobs, clock=lambda: actual)
            original.start(original.entries[0])
            saved = dict(original.entries[0])
            resumed = TaskRegistry(home / 'task_plan.json', home, jobs,
                                   clock=lambda: self.fail('Resume changed the original start time'))
            self.assertEqual(resumed.start(resumed.entries[0]), saved)

    def test_legacy_registration_records_are_rejected_without_edits(self):
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            path = home / 'task_plan.json'
            write(path, {'identity': 'registered in advance', 'tasks': jobs()})
            before = path.read_bytes()
            with self.assertRaises(ValueError):
                TaskRegistry(path, home, jobs)
            self.assertEqual(path.read_bytes(), before)

    def test_clock_collision_does_not_fabricate_a_later_time(self):
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            actual = datetime(2026, 11, 19, tzinfo=timezone.utc)
            registration = TaskRegistry(home / 'task_plan.json', home, jobs, clock=lambda: actual)
            registration.start(registration.entries[0])
            with self.assertRaises(RuntimeError):
                registration.start(registration.entries[1])
            self.assertNotIn('started_at', registration.entries[1])

    def test_scheduler_allocates_only_inside_the_worker_and_resume_skips_completed(self):
        from experiments.glm_four_arm_20261114 import run as controller
        from experiments.kg_version_comparison import worker
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            registration = TaskRegistry(home / 'task_plan.json', home, lambda: jobs(4))
            main_thread = threading.get_ident()
            executions = []
            def prepare(task):
                self.assertNotEqual(threading.get_ident(), main_thread)
                registration.start(task)
            def execute(task, config, protocol):
                self.assertNotEqual(threading.get_ident(), main_thread)
                result = {**task, 'status': 'execution_failed', 'finished_at': task['started_at']}
                write(Path(task['directory']) / 'result.json', result)
                executions.append(task['experiment_timestamp'])
                return result
            protocol = {'identity_schema': SCHEMA, 'tasks': registration.snapshot(),
                        'model_configs': {'Fixture model': {}}, 'frozen_sha256': {}}
            protocol = registration.bind(protocol)
            with patch.object(controller, 'HERE', home), patch.object(controller, 'MAX_ACTIVE', 2), patch.object(controller, 'MAX_PER_MODEL', 2), patch.object(controller, 'credit', return_value=100), patch.object(controller, 'report', return_value={}), patch.object(worker, 'run_workflow', side_effect=execute):
                self.assertTrue(all('directory' not in task for task in protocol['tasks']))
                controller.run(protocol, prepare_task=prepare)
                self.assertEqual(len(executions), 4)
                before = registration.snapshot()
                controller.run(protocol, prepare_task=lambda task: self.fail('Completed task restarted'))
                self.assertEqual(registration.snapshot(), before)
                self.assertEqual(len(executions), 4)


if __name__ == '__main__':
    unittest.main()
