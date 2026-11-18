"""Stage-six pilot and append-only postprocessing of successful saved workflows."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys
import threading
import time
import traceback

ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from experiments.kg_version_comparison.api import call,read,write,now
from experiments.kg_version_comparison.worker import code_text
from experiments.source_type_graph.repair_feedback import compose
from experiments.postprocess_stage6_20261116.session import Session
from experiments.postprocess_stage6_20261116.evaluate import validate,compare
from experiments.glm_four_arm_20261114 import run as base
from experiments.kg_version_comparison.parallel import MemoryAdmission

PHYSICS=dict(zip((c[0] for c in base.CASES),('heat_transfer','linear_elasticity','hyperelasticity','thermoelasticity','joule_heating','piezoelectricity')))
MODELS={'GLM 5.3 Flash','Qwen 3.8 27B'}
ADMISSION=MemoryAdmission();HYPER=threading.Semaphore(1);REPORT_LOCK=threading.Lock()

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def prepare():
    destination=HERE/'protocol.json'
    if destination.exists():return read(destination)
    source=read(ROOT/'experiments/all_models_four_arm_20261114/queue.json')
    jobs=[]
    for task in source['tasks']:
        if task['model'] not in MODELS:continue
        original=Path(task['directory']);result=read(original/'result.json')
        if result['status']!='execution_success':continue
        jobs.append({**task,'original_directory':str(original),
            'directory':str(HERE/'items'/task['model']/task['graph']/task['case']/f"run_{task['repeat']:02d}"),
            'sources':[str(original/p) for p in result['accepted_sources']]})
    requests=[('GLM 5.3 Flash','Card_Only','01_heatsink'),('Qwen 3.8 27B','Document_Top3','02_tensile_specimen'),
              ('GLM 5.3 Flash','KG_ver1','04_I_beam'),('Qwen 3.8 27B','KG_ver2','06_piezo_cantilever')]
    pilot=[next(j for j in jobs if (j['model'],j['graph'],j['case'])==req) for req in requests]
    configs={'GLM 5.3 Flash':read(base.HERE/'protocol.json')['model_configs']['GLM 5.3 Flash'],
             'Qwen 3.8 27B':read(ROOT/'experiments/all_models_four_arm_20261114/models/Qwen 3.8 27B/protocol.json')['model_configs']['Qwen 3.8 27B']}
    inputs=list(HERE.glob('*.py'))+list(HERE.glob('*.txt'))
    original_paths={p for j in jobs for p in j['sources']}
    value={'created_at':now(),'jobs':jobs,'pilot':pilot,'configs':configs,'max_repairs':5,'parallel':2,
        'scope':'Existing execution-success workflows only; stages 1-5 remain unchanged.',
        'gate':'Pilot must export structurally valid NPZ for all four cases using the same retained solve process. Field agreement is reported separately and is never repair feedback.',
        'criterion':'Coordinate-matched relative L2 against fixed reference fields, with the established thermal-increment normalization also reported.',
        'web_search':False,'tools':False,'source_sha256':{p:sha(p) for p in original_paths},
        'extension_sha256':{str(p):sha(p) for p in inputs}}
    write(destination,value);return value


def usage(folder):
    receipts=[read(p) for p in Path(folder).glob('attempt_*/submissions/submission_*/receipt.json')]
    good=[r for r in receipts if r.get('response')]
    values=[r['response'].get('usage',{}) for r in good]
    def total(items):return sum(items) if items and all(v is not None for v in items) else None
    return {'api_response_seconds':sum(r['seconds'] for r in good),'api_error_seconds':sum(r['seconds'] for r in receipts if not r.get('response')),
        'input_tokens':total([v.get('prompt_tokens') for v in values]),'output_tokens':total([v.get('completion_tokens') for v in values]),
        'reasoning_tokens':total([(v.get('completion_tokens_details') or {}).get('reasoning_tokens') for v in values]),
        'cached_tokens':total([(v.get('prompt_tokens_details') or {}).get('cached_tokens') for v in values]),
        'cost_usd':total([v.get('cost') for v in values])}


def postprocess(job,config,session):
    folder=Path(job['directory']);folder.mkdir(parents=True,exist_ok=True)
    source_dir=base.INPUTS/'cases'/job['case']/'cards'
    card=read(source_dir/'2_function_space.json')
    context=(HERE/f"{job['graph']}.txt").read_text()
    previous='\n\n'.join(f'# Accepted stage {i+1}\n'+Path(p).read_text() for i,p in enumerate(job['sources']))
    system=(HERE/'6_Postprocess_Prompt.txt').read_text()
    original='## Card\n'+json.dumps(card,ensure_ascii=False)+'\n'
    if context:original+='## API reference material (DOLFINx 0.10 API)\n'+context+'\n'
    original+='## Previously executed code\n```python\n'+previous+'\n```'
    (folder/'initial_prompt.txt').write_text(system+'\n\n'+original)
    write(folder/'card.json',card);current=original
    for attempt in range(6):
        path=folder/f'attempt_{attempt:02d}';path.mkdir(exist_ok=True)
        write(folder/'state.json',{'status':'generating','attempt':attempt,'pid':session.ready['pid'],'updated_at':now()})
        (path/'prompt.txt').write_text(current)
        response=call(config,[{'role':'system','content':system},{'role':'user','content':current}],path,stopped=lambda:(HERE/'STOP').exists())
        code=code_text(response);file=path/'code.py';file.write_text(code)
        if not code.strip():return {'status':'empty_response','repair_calls':attempt}
        write(folder/'state.json',{'status':'postprocessing','attempt':attempt,'pid':session.ready['pid'],'updated_at':now()})
        execution=session.execute(file,path/'output',path/'execution.json')
        assert execution['pid']==session.ready['pid']
        error=execution.get('error')
        if not error:
            try:validate(path/'output',card['unknowns'])
            except Exception:error=traceback.format_exc()
        if error:
            diagnostic={'kind':'postprocess_runtime_or_export_error','stage':'postprocess','diagnostic':error}
            write(path/'repair_feedback.json',diagnostic)
            if attempt==5:return {'status':'postprocess_failed','error':error,'repair_calls':attempt}
            current=compose(original,code,diagnostic);continue
        reference=ROOT/'experiments/glm_physical_review_20261115/references'/PHYSICS[job['case']]/'reference/solution.npz'
        assessment=compare(path/'output',card['unknowns'],reference,nonlinear=job['case']=='03_rubber_bushing')
        write(folder/'reference_assessment.json',assessment)
        return {'status':'export_success','reference_status':assessment['status'],'repair_calls':attempt,
                'accepted_source':str(file),'output_directory':str(path/'output'),'pid':execution['pid'],
                'same_process':True,'reference':str(reference),'reference_sha256':sha(reference)}


def run_job(job,config,held_session=None):
    folder=Path(job['directory']);folder.mkdir(parents=True,exist_ok=True)
    result=folder/'result.json'
    if result.exists():return read(result)
    started=time.monotonic();session=held_session
    hashes={p:sha(p) for p in job['sources']}
    write(folder/'state.json',{'status':'replaying' if session is None else 'retained_solve','updated_at':now()})
    try:
        if session is None:session=Session(job['sources'],folder/'session')
        value=postprocess(job,config,session)
    except Exception:
        value={'status':'stage6_error' if session is not None else 'replay_failed','error':traceback.format_exc()}
    finally:
        if session is not None:session.close()
    value.update({k:job[k] for k in ('model','graph','case','experiment_timestamp')},finished_at=now(),wall_seconds=time.monotonic()-started,
        replay_seconds=session.ready['replay_seconds'] if session is not None else None,
        original_sources_unchanged=all(sha(p)==h for p,h in hashes.items()),**usage(folder))
    if not value['original_sources_unchanged']:raise RuntimeError('Accepted source changed during stage-six execution')
    write(result,value);write(folder/'state.json',value)
    return value


def report(protocol):
    rows=[]
    for job in protocol['jobs']:
        folder=Path(job['directory']);p=folder/'result.json'
        if not p.exists():p=folder/'state.json'
        state=read(p) if p.exists() else {'status':'pending'}
        rows.append({**{k:job[k] for k in ('model','graph','case','repeat')},**state})
    value={'updated_at':now(),'planned':len(rows),'counts':dict(Counter(r['status'] for r in rows)),
           'reference_counts':dict(Counter(r.get('reference_status') for r in rows if r['status']=='export_success')),'rows':rows}
    write(HERE/'summary.json',value);return value


def run(mode):
    import fcntl
    lock=(HERE/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    protocol=prepare()
    assert all(sha(p)==h for p,h in protocol['source_sha256'].items())
    assert all(sha(p)==h for p,h in protocol['extension_sha256'].items())
    if mode=='backfill':assert read(HERE/'pilot_gate.json')['passed'], 'Pilot gate is not satisfied'
    jobs=protocol['pilot'] if mode=='pilot' else protocol['jobs']
    def work(job):
        if (HERE/'STOP').exists():return
        guard=HYPER if job['case']=='03_rubber_bushing' else threading.Semaphore(1)
        with guard,ADMISSION.slot(job['sources'][-1],'solve',lambda:(HERE/'STOP').exists(),lambda:None):
            value=run_job(job,protocol['configs'][job['model']])
        with REPORT_LOCK:report(protocol)
        print(json.dumps({k:value.get(k) for k in ('model','graph','case','repeat','status','reference_status','wall_seconds')}),flush=True)
    write(HERE/'controller.json',{'status':'running','mode':mode,'pid':__import__('os').getpid(),'started_at':now()})
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(work,[j for j in jobs if not (Path(j['directory'])/'result.json').exists()]))
    if mode=='pilot':
        outcomes=[read(Path(j['directory'])/'result.json') for j in jobs]
        write(HERE/'pilot_gate.json',{'passed':all(r['status']=='export_success' and r['same_process'] for r in outcomes),
            'outcomes':outcomes,'note':'Reference mismatch alone does not fail the export pipeline gate.'})
    report(protocol)
    write(HERE/'controller.json',{'status':'finished','mode':mode,'finished_at':now()})

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','pilot','backfill','status'])
    mode=parser.parse_args().mode
    if mode=='prepare':print(json.dumps({'jobs':len(prepare()['jobs']),'pilot':len(prepare()['pilot'])}))
    elif mode=='status':print(json.dumps(report(prepare())['counts']))
    else:run(mode)
