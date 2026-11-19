"""Copy terminal workflows into model/method/physics/outcome/timestamp folders."""
import csv
import json
from pathlib import Path
import shutil


def organize(home, tasks, destination):
    methods = {'KG': 'KG', 'Card_Only': 'Card', 'Document_Top3': 'RAG'}
    for model in {task['model'] for task in tasks}:
        for method in methods.values():
            for case in {task['case'] for task in tasks}:
                for outcome in ('Success', 'Fail'):
                    (destination / model / method / case / outcome).mkdir(parents=True, exist_ok=True)
    rows = []
    for task in tasks:
        if not task.get('directory'):
            continue
        source = Path(task['directory'])
        result_file = source / 'result.json'
        if not result_file.exists():
            continue
        result = json.loads(result_file.read_text())
        outcome = 'Success' if result['status'] == 'execution_success' else 'Fail'
        stamp = task['experiment_timestamp']
        relative = Path(task['model']) / methods[task['graph']] / task['case'] / outcome / stamp
        target = destination / relative
        shutil.copytree(source, target, dirs_exist_ok=True)
        post = home / 'postprocess/items' / task['model'] / task['graph'] / task['case'] / stamp
        if post.exists():
            shutil.copytree(post, target / 'postprocess', dirs_exist_ok=True)
        info = {**task, 'method': methods[task['graph']], 'outcome': outcome,
                'status': result['status'], 'folder': relative.as_posix()}
        (target / 'case_info.json').write_text(json.dumps(info, ensure_ascii=False, indent=2) + '\n')
        rows.append({k: info[k] for k in ('model', 'method', 'case', 'outcome', 'status', 'experiment_timestamp', 'folder')})
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / 'index.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['model', 'method', 'case', 'outcome', 'status', 'experiment_timestamp', 'folder'])
        writer.writeheader()
        writer.writerows(rows)
    return {'organized': len(rows), 'directory': str(destination)}
