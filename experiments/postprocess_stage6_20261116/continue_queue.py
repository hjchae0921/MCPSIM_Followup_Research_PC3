"""Resume the queued models with stage six in the successful solve process."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import fcntl
import os
from pathlib import Path
import signal
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.postprocess_stage6_20261116 import run as extension
from experiments.postprocess_stage6_20261116.session import Session
from experiments.all_models_four_arm_20261114 import run_queue as queue
from experiments.kg_version_comparison import worker
from experiments.kg_version_comparison.api import read,write,now

HERE=extension.HERE
LOCAL=threading.local()
EXECUTE=worker.execute_stage
WORKFLOW=worker.run_workflow
BASE_RUN=queue.base.run


def make_job(task,result):
    original=Path(task['directory'])
    return {**task,'original_directory':str(original),
        'directory':str(HERE/'items'/task['model']/task['graph']/task['case']/task['experiment_timestamp']),
        'sources':[str(original/p) for p in result['accepted_sources']]}


def retain_solve(source,prefix,directory,**kwargs):
    if kwargs.get('stage')!='solve':return EXECUTE(source,prefix,directory,**kwargs)
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    cached=directory/'validation.json'
    # Prior failed attempts remain failed; only a successful solve needs a live namespace.
    if cached.exists() and not read(cached)['runtime_pass']:
        return EXECUTE(source,prefix,directory,**kwargs)
    prefix_file=directory/'prefix.py';prefix_file.write_text(prefix+'\n')
    import hashlib
    from codegen.runtime import digest
    write(directory/'execution_inputs.json',{'source_sha256':digest(source),
        'prefix_sha256':hashlib.sha256((prefix+'\n').encode()).hexdigest(),
        'executor_sha256':digest(ROOT/'codegen/runtime.py'),'python':str(kwargs.get('python') or sys.executable),
        'timeout_seconds':kwargs.get('timeout',1200),'evaluation_mode':'execution_only'})
    started=time.monotonic();error=None;timed_out=False
    try:
        if getattr(LOCAL,'session',None) is not None:LOCAL.session.close();LOCAL.session=None
        LOCAL.session=Session([prefix_file,source],directory/'retained',python=kwargs.get('python') or sys.executable,
                              replay_timeout=kwargs.get('timeout',1200))
    except Exception as failure:
        error=str(failure);timed_out=isinstance(failure,TimeoutError);LOCAL.session=None
    report={'evaluation_mode':'execution_only','stage':'solve','runtime_pass':error is None,
        'semantic_pass':None,'verified_correct':None,'field_correct':None,'reference_comparison_performed':False,
        'card_contract_grading_performed':False,'process_returncode':None if error is None else 1,
        'execution_seconds':time.monotonic()-started,'timed_out':timed_out,
        'error':error,'process_retained_for_stage6':error is None}
    write(cached,report)
    return report


def workflow(task,config,protocol):
    LOCAL.session=None
    try:
        result=WORKFLOW(task,config,protocol)
        if result['status']=='execution_success':
            job=make_job(task,result)
            value=extension.run_job(job,config,held_session=LOCAL.session)
            LOCAL.session=None
            write(HERE/'continuation_latest.json',value)
        return result
    finally:
        if LOCAL.session is not None:LOCAL.session.close();LOCAL.session=None


def resume_model(protocol, *, prepare_task=None):
    # Previously completed solves require one replay; new solves remain live for stage six.
    successes=[]
    for task in protocol['tasks']:
        if not task.get('directory'):continue
        path=Path(task['directory'])/'result.json'
        if path.exists() and read(path)['status']=='execution_success':
            job=make_job(task,read(path))
            if not (Path(job['directory'])/'result.json').exists():successes.append(job)
    def backfill(job):
        guard=extension.HYPER if job['case']=='03_rubber_bushing' else nullcontext()
        with guard,extension.ADMISSION.slot(job['sources'][-1],'solve',lambda:(HERE/'STOP').exists(),lambda:None):
            return extension.run_job(job,protocol['model_configs'][job['model']])
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(backfill,successes))
    BASE_RUN(protocol, prepare_task=prepare_task)


def alive(pid):
    path=Path(f'/proc/{pid}/stat')
    return path.exists() and path.read_text().split(') ',1)[1].split()[0]!='Z'


def prioritize(models, order):
    names=[m['folder'] for m in models]
    if len(order)!=len(set(order)) or not set(order)<=set(names):
        raise ValueError('Queue priority contains duplicate or unknown models')
    positions={name:index for index,name in enumerate(order)}
    return sorted(models,key=lambda model:positions.get(model['folder'],len(order)))


def main():
    assert read(HERE/'pilot_gate.json')['passed']
    protocol=read(HERE/'protocol.json')
    assert all((Path(j['directory'])/'result.json').exists() for j in protocol['jobs']), 'GLM/Qwen backfill must finish first'
    if (HERE/'glm46/protocol.json').exists():
        if (HERE/'STOP').exists():return
        import subprocess
        write(HERE/'continuation.json',{'status':'running_glm46_backfill','updated_at':now()})
        subprocess.run([sys.executable,'-u','-B',str(HERE/'run_glm46.py'),'run'],check=True)
    drain=read(HERE/'drain.json')
    while alive(drain['child']['pid']):
        write(HERE/'continuation.json',{'status':'waiting_for_original_active_jobs','updated_at':now()})
        if (HERE/'STOP').exists():return
        time.sleep(10)
    parent=drain['parent']['pid']
    if alive(parent):
        assert 'run_queue.py' in Path(f'/proc/{parent}/cmdline').read_text()
        os.kill(parent,signal.SIGKILL)
    lock=(queue.HERE/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    value=queue.prepare()
    worker.execute_stage=retain_solve;worker.run_workflow=workflow;queue.base.run=resume_model
    priority=read(HERE/'queue_priority.json')
    ownership_path=HERE/'station_assignments.json'
    owners=read(ownership_path)['model_owners'] if ownership_path.exists() else {}
    for model in prioritize(value['pending_models'],priority['model_order']):
        if (HERE/'STOP').exists():break
        name=model['folder'];home=queue.HERE/'models'/name
        if owners.get(name,'pc1')!='pc1':continue
        if read(home/'queue_state.json')['status'] in ('finished','blocked_preflight','controller_error'):continue
        write(HERE/'continuation.json',{'status':'running','model':name,'pid':os.getpid(),'updated_at':now(),
                                       'execution':'Accepted solve and stage six share a persistent Python process'})
        # BASE_RUN installs its admission wrapper once per model; reset before the next model.
        worker.execute_stage=retain_solve
        queue.run_model(name,value)
    write(HERE/'continuation.json',{'status':'stopped' if (HERE/'STOP').exists() else 'finished','updated_at':now()})

if __name__=='__main__':main()
