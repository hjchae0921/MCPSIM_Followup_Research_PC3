"""Launch this station using the active, pinned Python environment."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def credentials(env, transport):
    name, filename = {
        'openai': ('OPENAI_API_KEY', 'openai_api.txt'),
        'openrouter': ('OPENROUTER_API_KEY', 'openrouter_api_key.txt'),
        'qwencloud': ('QWENCLOUD_API_KEY', 'qwenstudio_apikey.txt'),
    }[transport]
    if env.get(name):
        return
    explicit = env.get(name + '_FILE')
    candidates = [Path(explicit)] if explicit else [ROOT / filename, ROOT.parent / filename]
    for path in candidates:
        if path.is_file():
            value = path.read_text().strip()
            if value and not any(c.isspace() for c in value):
                env[name] = value
                return
    raise SystemExit(f'Set {name} or {name}_FILE; API keys must stay outside Git.')


def main(args):
    lock = (ROOT / 'launcher.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('This station launcher is already running.')
    env = os.environ.copy()
    plan = json.loads((ROOT / 'assignment.json').read_text())
    if args.station:
        plan['models'] = json.loads((ROOT / 'station_assignments.json').read_text())['stations'][args.station]
    for transport in {m['transport'] for m in plan['models']}:
        credentials(env, transport)
    env['PYTHONUNBUFFERED'] = '1'
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        env[name] = '1'
    commands = [('check',), ('prepare',),
                ('run', '--workers', str(args.workers), '--per-model', str(args.per_model)),
                ('organize',), ('export',)]
    # The controller writes directly to a file; no launcher output pipe can break.
    with (ROOT / 'worker.log').open('a', buffering=1) as log:
        for command in commands:
            if args.station:
                command = (*command, '--station', args.station)
            print('Starting: ' + ' '.join(command), file=log, flush=True)
            result = subprocess.run([sys.executable, '-u', '-B', str(ROOT / 'run.py'), *command],
                                    cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                    stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                raise SystemExit(f'{command[0]} stopped with exit code {result.returncode}; see worker.log.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--background', action='store_true')
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--per-model', type=int, default=2)
    parser.add_argument('--station', choices=['pc1', 'pc2', 'pc3'])
    args = parser.parse_args()
    if not 1 <= args.per_model <= args.workers <= 32:
        parser.error('Require 1 <= per-model <= workers <= 32')
    if args.background:
        with (ROOT / 'launcher.log').open('a') as log:
            command = [sys.executable, '-u', '-B', str(Path(__file__).resolve()),
                       '--workers', str(args.workers), '--per-model', str(args.per_model)]
            if args.station:
                command += ['--station', args.station]
            process = subprocess.Popen(command,
                                       cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        print(f'Background launcher PID: {process.pid}')
        print(f'Progress: {ROOT / "worker.log"}')
        print(f'Launcher: {ROOT / "launcher.log"}')
    else:
        main(args)
