"""Run one station's frozen three-arm, six-physics benchmark assignment."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import zipfile
import signal

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from experiments.glm_four_arm_20261114 import run as base
from experiments.postprocess_stage6_20261116 import run as extension
from experiments.postprocess_stage6_20261116 import continue_queue as continuation
from experiments.tensile_server_export_20261117.server_export import attach
from experiments.kg_version_comparison import worker
from experiments.kg_version_comparison.api import read, write, now, preflight
from task_registry import SCHEMA, TaskRegistry

PLAN = read(ROOT / 'assignment.json')
HOME = ROOT / 'outputs' / PLAN['campaign'] / PLAN['station']
INPUTS, POST = HOME / 'inputs', HOME / 'postprocess'
ARMS = tuple(PLAN['arms'])
REGISTRY = None


def select_station(station):
    global PLAN, HOME, INPUTS, POST
    if station is None:
        return
    models = read(ROOT / 'station_assignments.json')['stations'][station]
    PLAN = {**PLAN, 'station': station, 'models': models,
            'planned_workflows': len(models) * PLAN['workflows_per_model']}
    HOME = ROOT / 'outputs' / PLAN['campaign'] / station
    INPUTS, POST = HOME / 'inputs', HOME / 'postprocess'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_bundle():
    for name, digest in read(ROOT / 'bundle_manifest.json')['files'].items():
        if not (ROOT / name).is_file() or sha(ROOT / name) != digest:
            raise ValueError('Bundle file changed or missing: ' + name)


def environment():
    from mpi4py import MPI
    from petsc4py import PETSc
    # Preserve Python's recoverable BrokenPipeError in the controller.
    signal.signal(signal.SIGPIPE, signal.SIG_IGN)
    expected = read(ROOT / 'runtime_fingerprint.json')
    measured = {name: importlib.metadata.version(name) for name in
                ('fenics-dolfinx', 'fenics-basix', 'fenics-ufl', 'fenics-ffcx', 'petsc4py', 'numpy', 'scipy', 'mpi4py', 'gmsh')}
    measured['petsc'] = '.'.join(map(str, PETSc.Sys.getVersion()))
    for name, version in measured.items():
        if version != expected[name]:
            raise ValueError(f'Runtime mismatch: {name}: {version}, expected {expected[name]}')
    if MPI.COMM_WORLD.size != 1 or not PETSc.Sys.hasExternalPackage('mumps'):
        raise ValueError('Use one MPI rank and PETSc with MUMPS.')
    import platform
    measured.update(mpi_library=MPI.Get_library_version(), has_mumps=True, mpi_size=1,
                    platform=platform.platform(), machine=platform.machine(), logical_cpus=os.cpu_count())
    HOME.mkdir(parents=True, exist_ok=True)
    write(HOME / 'runtime.json', measured)
    return measured


def prepare_inputs():
    for folder, destination in [('frozen_inputs', INPUTS), ('frozen_postprocess', POST)]:
        source = ROOT / folder
        for path in source.rglob('*'):
            if not path.is_file():
                continue
            target = destination / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            content = path.read_bytes()
            if path.name == '1_mesh.json':
                card = json.loads(content)
                card['mesh']['path'] = str((target.parent.parent / 'model.msh').resolve())
                content = (json.dumps(card, ensure_ascii=False, indent=2) + '\n').encode()
            if target.exists() and target.read_bytes() != content:
                raise ValueError('Local input changed: ' + str(target))
            if not target.exists():
                target.write_bytes(content)
    write(HOME / 'input_relocation.json', {'policy': 'Only mesh card paths change. Other cards, prompts and contexts remain byte-identical.',
                                         'source_bundle': sha(ROOT / 'bundle_manifest.json')})


def planned_tasks():
    result = []
    for cycle in range(PLAN['repeats']):
        for i, (case, case_folder) in enumerate(base.CASES):
            shift = (cycle + i) % len(ARMS)
            for arm in ARMS[shift:] + ARMS[:shift]:
                for model in PLAN['models']:
                    result.append({'model': model['folder'], 'graph': arm,
                                   'case': case, 'case_folder': case_folder})
    assert len(result) == PLAN['planned_workflows']
    return result


def registry():
    global REGISTRY
    if REGISTRY is None or REGISTRY.path != HOME / 'task_plan.json':
        REGISTRY = TaskRegistry(HOME / 'task_plan.json', HOME, planned_tasks)
    return REGISTRY


def tasks():
    return registry().snapshot()


def post_path(task):
    if not task.get('experiment_timestamp'):
        return None
    return POST / 'items' / task['model'] / task['graph'] / task['case'] / task['experiment_timestamp']


def report(_=None):
    rows = []
    for task in tasks():
        folder = Path(task['directory']) if task.get('directory') else None
        path = folder / 'result.json' if folder else None
        terminal_solve = path is not None and path.exists()
        if folder is not None and not terminal_solve:
            path = folder / 'state.json'
        state = read(path) if path is not None and path.exists() else {'status': 'queued'}
        pp = post_path(task)
        post_result = pp / 'result.json' if pp else None
        post_state = pp / 'state.json' if pp else None
        post = read(post_result if post_result.exists() else post_state) if pp and (post_result.exists() or post_state.exists()) else {}
        row = {k: task.get(k) for k in ('model', 'graph', 'case', 'experiment_timestamp', 'started_at')}
        row.update({k: state.get(k) for k in ('status', 'stage', 'wall_seconds', 'api_response_seconds', 'api_error_seconds',
                                             'input_tokens', 'output_tokens', 'reasoning_tokens', 'cached_tokens', 'cost_usd', 'repair_calls')})
        row.update(postprocess_status=post.get('status'), reference_status=post.get('reference_status'),
                   completed=terminal_solve and (state['status'] != 'execution_success' or (post_result is not None and post_result.exists())),
                   end_to_end_seconds=(state.get('wall_seconds') or 0) + (post.get('wall_seconds') or 0))
        row['postprocess_usage'] = {k: post.get(k) for k in ('api_response_seconds', 'api_error_seconds', 'input_tokens',
                                                          'output_tokens', 'reasoning_tokens', 'cached_tokens', 'cost_usd', 'repair_calls')}
        rows.append(row)
    groups = []
    for model in PLAN['models']:
        for arm in ARMS:
            for case, _ in base.CASES:
                subset = [r for r in rows if (r['model'], r['graph'], r['case']) == (model['folder'], arm, case)]
                done = sum(r['completed'] for r in subset)
                matched = sum(r['reference_status'] == 'matched' for r in subset)
                groups.append({'model': model['folder'], 'arm': arm, 'case': case, 'planned': len(subset), 'completed': done,
                    'execution_success': sum(r['status'] == 'execution_success' for r in subset),
                    'postprocess': dict(Counter(r['postprocess_status'] for r in subset if r['postprocess_status'])),
                    'reference': dict(Counter(r['reference_status'] for r in subset if r['reference_status'])),
                    'reference_agreement_rate_completed': matched / done if done else None})
    value = {'station': PLAN['station'], 'updated_at': now(), 'planned': len(rows),
             'completed': sum(r['completed'] for r in rows), 'counts': dict(Counter(r['status'] for r in rows)),
             'reference_counts': dict(Counter(r['reference_status'] for r in rows if r['reference_status'])),
             'groups': groups, 'rows': rows}
    write(HOME / 'summary.json', value)
    return {k: value[k] for k in ('station', 'planned', 'completed', 'counts', 'reference_counts')}


def run(workers, per_model):
    verify_bundle()
    environment()
    prepare_inputs()
    with (HOME / 'station.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return run_station(workers, per_model)


def run_station(workers, per_model):
    path = HOME / 'protocol.json'
    if path.exists():
        protocol = read(path)
        base.legacy.verify(protocol)
        if (protocol['api_concurrency'], protocol['per_model_concurrency']) != (workers, per_model):
            raise ValueError('Resume with the frozen concurrency settings.')
    else:
        def probe(model):
            value = preflight(model, HOME / 'preflight' / model['folder'],
                              strict_low=True, omit_temperature=model.get('omit_temperature', False))
            config = value.get('config', {})
            expected_temperature = None if model.get('omit_temperature', False) else 0
            if (value.get('status') != 'ready'
                    or config.get('temperature') != expected_temperature
                    or config.get('reasoning', {}).get('enabled') is False
                    or config.get('reasoning', {}).get('effort') != 'low'):
                raise ValueError('Reasoning/temperature preflight failed: ' + model['folder'])
            return value
        with ThreadPoolExecutor(max_workers=len(PLAN['models'])) as pool:
            evidence = list(pool.map(probe, PLAN['models']))
        frozen = {str(ROOT / n): v for n, v in read(ROOT / 'bundle_manifest.json')['files'].items()}
        frozen.update({str(p): sha(p) for p in INPUTS.rglob('*') if p.is_file()})
        frozen.update({str(p): sha(p) for p in POST.glob('*') if p.is_file()})
        protocol = {**read(ROOT / 'experiment_settings.json'), 'study': PLAN['campaign'] + '_' + PLAN['station'],
                    'identity_schema': SCHEMA,
                    'temperature_exceptions': {m['id']: {'temperature': None, 'reasoning_effort': 'low',
                                                       'policy': 'Omit temperature; retain low reasoning.'}
                                               for m in PLAN['models'] if m.get('omit_temperature', False)},
                    'models': PLAN['models'], 'tasks': tasks(), 'planned_workflows': PLAN['planned_workflows'],
                    'model_configs': {m['folder']: e['config'] for m, e in zip(PLAN['models'], evidence)},
                    'frozen_sha256': frozen, 'api_concurrency': workers, 'per_model_concurrency': per_model,
                    'station': PLAN['station'], 'bundle_sha256': sha(ROOT / 'bundle_manifest.json')}
        write(path, protocol)
    registration = registry()
    protocol = registration.bind(protocol)
    base.HERE, base.INPUTS, base.RESULTS = HOME, INPUTS, HOME
    base.MAX_ACTIVE, base.MAX_PER_MODEL, base.report = workers, per_model, report
    extension.HERE = continuation.HERE = POST
    original_postprocess = extension.postprocess
    def postprocess(job, config, session):
        card = read(INPUTS / 'cases' / job['case'] / 'cards/2_function_space.json')
        attach(session, card['unknowns'])
        return original_postprocess(job, config, session)
    extension.postprocess = postprocess
    worker.execute_stage, worker.run_workflow = continuation.retain_solve, continuation.workflow
    continuation.resume_model(protocol, prepare_task=registration.start)
    print(json.dumps(report(), indent=2))


def export():
    target = ROOT / (PLAN['station'] + '_' + PLAN['campaign'] + '_results.zip')
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in HOME.rglob('*'):
            if path.is_file() and not any(p in {'cache', '__pycache__'} for p in path.relative_to(HOME).parts):
                archive.write(path, path.relative_to(ROOT))
    print(target)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['check', 'prepare', 'run', 'status', 'stop', 'organize', 'export'])
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--per-model', type=int, default=4)
    parser.add_argument('--station', choices=['pc1', 'pc2', 'pc3'])
    args = parser.parse_args()
    select_station(args.station)
    if not 1 <= args.per_model <= args.workers <= 32:
        parser.error('Require 1 <= per-model <= workers <= 32')
    if args.command == 'check':
        verify_bundle()
        print(json.dumps(environment(), indent=2))
    elif args.command == 'prepare':
        verify_bundle()
        prepare_inputs()
        print(json.dumps(report(), indent=2))
    elif args.command == 'run':
        run(args.workers, args.per_model)
    elif args.command == 'status':
        print(json.dumps(report(), indent=2))
    elif args.command == 'export':
        export()
    elif args.command == 'organize':
        from organize import organize
        print(json.dumps(organize(HOME, tasks(), HOME / 'organized_results'), indent=2))
    else:
        HOME.mkdir(parents=True, exist_ok=True)
        POST.mkdir(parents=True, exist_ok=True)
        (HOME / 'STOP').touch()
        (POST / 'STOP').touch()
