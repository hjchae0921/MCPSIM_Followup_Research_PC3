"""Queue the remaining models behind the running, frozen GLM comparison."""
import argparse
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.glm_four_arm_20261114 import run as base
from experiments.kg_version_comparison.api import now, read, write, preflight

HERE = Path(__file__).resolve().parent
GLM = ROOT / 'experiments/glm_four_arm_20261114'
RESULTS = base.RESULTS
MODELS = base.legacy.MODELS + [
    {'folder': 'Deepseek V4 Flash-0731', 'id': 'deepseek/deepseek-v4-flash-0731', 'transport': 'openrouter'}
]
HYPOTHESES = {
    'H1': 'Without API reference retrieval, generated code is more likely to mix API versions, invent unavailable APIs, or use deprecated APIs.',
    'H2': 'API document RAG may reduce these errors, while KG dataflow and type-compatibility relationships may further improve correct API composition.',
    'status': 'Hypotheses to test, not conclusions. Preserve counterexamples and negative results.',
    'control_scope': 'Card_Only supplies completed cards and five-stage procedures; it is not a natural-language-only or one-shot control.',
    'evidence': ['version-specific API existence and deprecation', 'runtime traceback and error location',
                 'exact retrieved context', 'generated code and self-repair changes', 'provider-returned reasoning'],
    'interpretation': 'Reasoning mentions alone do not demonstrate causal use of graph relations. Compare supplied information, code, and outcomes. Execution success does not establish physical correctness.',
    'availability': 'Save API-returned reasoning and raw responses; unavailable private reasoning cannot be reconstructed.'
}


def allocation(models):
    original = base.MODELS
    try:
        base.MODELS = models
        return base.tasks()
    finally:
        base.MODELS = original


def prepare():
    destination = HERE / 'queue.json'
    if destination.exists():
        value = read(destination)
        base.legacy.verify(value)
        return value
    existing = read(GLM / 'protocol.json')
    base.legacy.verify(existing)
    folders = {p.name for p in RESULTS.iterdir() if p.is_dir()}
    assert folders == {m['folder'] for m in MODELS}, 'Unmapped or missing model folder'
    active_names = {m['folder'] for m in existing['models']}
    pending = [m for m in MODELS if m['folder'] not in active_names]
    tasks = allocation(MODELS)
    assert len(tasks) == 5280 and len({t['directory'] for t in tasks}) == 5280
    assert {t['directory'] for t in existing['tasks']} <= {t['directory'] for t in tasks}
    value = {
        'created_at': now(), 'study': 'all_models_four_arm_runtime_aligned_20261114',
        'models': MODELS, 'pending_models': pending, 'tasks': tasks,
        'planned_workflows': len(tasks), 'already_scheduled': 960, 'additional_workflows': 4320,
        'base_protocol': str(GLM / 'protocol.json'), 'hypotheses': HYPOTHESES,
        'schedule': 'Wait for the GLM controller to finish, then run one remaining model at a time, up to four workflows concurrently. No duplicate GLM submissions.',
        'sampling': 'Temperature 0, low reasoning; if unsupported, record the lowest accepted setting. GPT-5 mini uses the previously approved default temperature 1 and low reasoning.',
        'preflight': 'Fresh capability checks before each queued model; unsupported or failed checks are blocked separately, never counted as coding failures.',
        'frozen_sha256': {**existing['frozen_sha256'], str(GLM / 'protocol.json'): base.legacy.sha(GLM / 'protocol.json'),
                          str(Path(__file__)): base.legacy.sha(Path(__file__))}
    }
    for task in tasks:
        Path(task['directory']).mkdir(parents=True, exist_ok=True)
    for model in pending:
        write(HERE / 'models' / model['folder'] / 'queue_state.json', {'status': 'queued', 'model': model, 'workflows': 480})
    write(destination, value)
    write(RESULTS / 'four_arm_all_models_manifest.json', value)
    (RESULTS / 'research_hypotheses.txt').write_text('\n\n'.join(f'{key}: {item}' for key, item in HYPOTHESES.items()) + '\n')
    return value


def report(value):
    rows = []
    for task in value['tasks']:
        folder = Path(task['directory'])
        path = folder / 'result.json'
        if not path.exists():
            path = folder / 'state.json'
        state = read(path) if path.exists() else {}
        row = {key: task[key] for key in ('model', 'graph', 'case', 'repeat')}
        row.update({key: state.get(key) for key in ('stage', 'wall_seconds', 'api_response_seconds', 'api_error_seconds',
                                                  'input_tokens', 'output_tokens', 'cached_tokens', 'reasoning_tokens', 'cost_usd', 'repair_calls')})
        row['status'] = state.get('status', 'queued')
        rows.append(row)
    summary = {'updated_at': now(), 'planned': len(rows), 'counts': dict(Counter(r['status'] for r in rows)), 'rows': rows}
    write(HERE / 'summary.json', summary)
    write(RESULTS / 'four_arm_all_models_summary.json', summary)
    return summary['counts']


def run_model(name, value):
    model = next(m for m in value['pending_models'] if m['folder'] == name)
    home = HERE / 'models' / name
    if (home / 'STOP').exists():
        return
    write(home / 'queue_state.json', {'status': 'preflight', 'model': model, 'updated_at': now()})
    evidence = preflight(model, home / 'preflight', gpt_temperature_exception=name == 'GPT-5 mini')
    if evidence.get('status') != 'ready':
        write(home / 'queue_state.json', {'status': 'blocked_preflight', 'evidence': evidence, 'updated_at': now()})
        return
    source = read(GLM / 'protocol.json')
    protocol = {**source, 'study': value['study'], 'models': [model],
                'tasks': [t for t in value['tasks'] if t['model'] == name], 'planned_workflows': 480,
                'model_configs': {name: evidence['config']}, 'hypotheses': HYPOTHESES,
                'reasoning': evidence['reasoning_label'], 'temperature': evidence['effective_temperature'],
                'frozen_sha256': {**value['frozen_sha256'], str(home / 'preflight/preflight.json'): base.legacy.sha(home / 'preflight/preflight.json')}}
    write(home / 'protocol.json', protocol)
    for arm in base.ARMS:
        write(RESULTS / name / arm / 'settings.json', {'study': value['study'], 'protocol': str(home / 'protocol.json'),
              'model_config': evidence['config'], 'reasoning_label': evidence['reasoning_label'],
              'effective_temperature': evidence['effective_temperature'], 'arm': arm, 'workflows': 120,
              'repeats_per_case': 20, 'evaluation_mode': 'execution_only'})
    base.HERE = home
    base.INPUTS = GLM / 'inputs'
    base.report = lambda protocol: report(value)
    write(home / 'queue_state.json', {'status': 'running', 'updated_at': now()})
    base.run(protocol)
    write(home / 'queue_state.json', {'status': read(home / 'controller.json')['status'], 'updated_at': now()})


def supervise(value):
    lock = (HERE / 'queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    while not (HERE / 'STOP').exists():
        controller = read(GLM / 'controller.json')
        if controller['status'] == 'finished':
            break
        status = 'waiting_for_glm'
        if controller['status'] in ('stopped', 'stopping'):
            status = 'blocked_glm_stopped'
        elif not Path('/proc', str(controller['pid'])).exists():
            status = 'blocked_glm_controller_missing'
        write(HERE / 'controller.json', {'pid': os.getpid(), 'status': status, 'updated_at': now(), 'additional_workflows': 4320})
        report(value)
        time.sleep(20)
    for model in value['pending_models']:
        if (HERE / 'STOP').exists():
            break
        home = HERE / 'models' / model['folder']
        if read(home / 'queue_state.json')['status'] in ('finished', 'blocked_preflight', 'controller_error'):
            continue
        write(HERE / 'controller.json', {'pid': os.getpid(), 'status': 'running', 'model': model['folder'], 'updated_at': now()})
        with (home / 'controller.log').open('ab', buffering=0) as log:
            child = subprocess.Popen([sys.executable, '-u', '-B', str(Path(__file__)), 'model', '--name', model['folder']],
                                     cwd=ROOT, stdout=log, stderr=log, stdin=subprocess.DEVNULL)
            while child.poll() is None:
                if (HERE / 'STOP').exists():
                    (home / 'STOP').touch()
                time.sleep(20)
        if child.returncode:
            write(home / 'queue_state.json', {'status': 'controller_error', 'exit_code': child.returncode, 'updated_at': now()})
    report(value)
    states = {m['folder']: read(HERE / 'models' / m['folder'] / 'queue_state.json')['status'] for m in value['pending_models']}
    status = 'stopped' if (HERE / 'STOP').exists() else 'finished' if all(s == 'finished' for s in states.values()) else 'finished_with_blocked_models'
    write(HERE / 'controller.json', {'pid': os.getpid(), 'status': status, 'models': states, 'updated_at': now()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'run', 'model', 'status'])
    parser.add_argument('--name')
    args = parser.parse_args()
    value = prepare()
    if args.command == 'prepare':
        print(json.dumps({k: value[k] for k in ('planned_workflows', 'already_scheduled', 'additional_workflows')}))
    elif args.command == 'run':
        supervise(value)
    elif args.command == 'model':
        run_model(args.name, value)
    else:
        print(json.dumps(report(value)))
