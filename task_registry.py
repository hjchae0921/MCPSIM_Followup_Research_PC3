"""Keep pending jobs unassigned; allocate paths inside their execution thread."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
import threading

from experiments.kg_version_comparison.api import now, read, write

SCHEMA = 'actual_workflow_start_v1'
KST = timezone(timedelta(hours=9))
DESCRIPTORS = ('model', 'graph', 'case', 'case_folder')
METHODS = {'Card_Only': 'Card', 'Document_Top3': 'RAG', 'KG': 'KG'}


class TaskRegistry:
    def __init__(self, path, home, factory, clock=None):
        self.path, self.home = Path(path), Path(home)
        self.lock = threading.RLock()
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        if self.path.exists():
            value = read(self.path)
            if value.get('identity_schema') != SCHEMA:
                raise ValueError('This is a registration-time campaign. Keep its records unchanged and use the new actual-start campaign.')
            self.entries = value['tasks']
        else:
            self.entries = factory()
            self._save(self.entries)
        for task in self.entries:
            if task.get('directory'):
                self._validate_started(task)
            elif any(task.get(key) is not None for key in ('started_at', 'experiment_timestamp')):
                raise ValueError('An unstarted task must not have a start timestamp.')

    def _save(self, entries):
        write(self.path, {'identity_schema': SCHEMA,
                         'identity': 'No experiment path or timestamp until the workflow execution thread starts.',
                         'updated_at': now(), 'tasks': entries})

    def _validate_started(self, task):
        started = datetime.fromisoformat(task['started_at'])
        if started.tzinfo is None:
            raise ValueError('Workflow start time must have a timezone.')
        stamp = started.astimezone(KST).strftime('%Y_%m_%d_%H_%M_%S_%f')
        expected = self.home / 'records' / task['model'] / METHODS[task['graph']] / task['case'] / stamp
        if task['experiment_timestamp'] != stamp or Path(task['directory']) != expected:
            raise ValueError('Experiment path must exactly match its actual workflow start time.')

    def snapshot(self):
        with self.lock:
            return [dict(task) for task in self.entries]

    def bind(self, protocol):
        if protocol.get('identity_schema') != SCHEMA:
            raise ValueError('Protocol predates actual-start identities; its records will not be migrated.')
        frozen = [{key: task[key] for key in DESCRIPTORS} for task in protocol['tasks']]
        current = [{key: task[key] for key in DESCRIPTORS} for task in self.entries]
        if frozen != current:
            raise ValueError('Task registry differs from the frozen assignment.')
        return {**protocol, 'tasks': self.entries}

    def start(self, task):
        # Called by the execution thread, never by prepare/report/the submitter.
        with self.lock:
            if task.get('directory'):
                self._validate_started(task)
                return task
            if not any(entry is task for entry in self.entries):
                raise ValueError('Execution task does not belong to this registry.')
            used = {entry.get('experiment_timestamp') for entry in self.entries}
            for _ in range(1000):
                started = self.clock()
                if started.tzinfo is None:
                    raise ValueError('Clock must return a timezone-aware actual time.')
                stamp = started.astimezone(KST).strftime('%Y_%m_%d_%H_%M_%S_%f')
                if stamp not in used:
                    break
            else:
                raise RuntimeError('The system clock did not advance; refusing to fabricate a timestamp.')
            folder = self.home / 'records' / task['model'] / METHODS[task['graph']] / task['case'] / stamp
            if folder.exists():
                raise FileExistsError('Actual-start directory already exists: ' + str(folder))
            candidate = {**task, 'started_at': started.isoformat(),
                         'experiment_timestamp': stamp, 'directory': str(folder)}
            updated = [candidate if entry is task else entry for entry in self.entries]
            self._save(updated)
            task.update(candidate)
            return task
