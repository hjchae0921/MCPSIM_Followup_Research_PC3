"""Frozen four-arm, six-physics GLM comparison using the established stage worker."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from contextlib import nullcontext
import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time
import traceback
import urllib.request

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from common.models import credentials
from codegen.cards import render as render_original
from experiments.kg_version_comparison import campaign as legacy, worker
from experiments.kg_version_comparison.api import now, read, write, preflight
from experiments.kg_version_comparison.parallel import MemoryAdmission

RESULTS = ROOT / 'results_for_paper/benchmarks/benchmark_results'
INPUTS = HERE / 'inputs'
MODELS = [m for m in legacy.MODELS if m['folder'] in {'GLM 4.6','GLM 5.3 Flash'}]
ARMS = ('Card_Only','Document_Top3','KG_ver1','KG_ver2')
STAGES, TEMPLATES, CARD_FILES, CASES = legacy.STAGES, legacy.TEMPLATES, legacy.CARD_FILES, legacy.CASES
MAX_ACTIVE, MAX_PER_MODEL = 8, 4


def neutral_template(text):
    text = text.replace('and API knowledge graph,', 'and any supplied API reference material,')
    text = text.replace('## Knowledge graph (dolfinx 0.10 API)', '## API reference material (DOLFINx 0.10 API)')
    text = text.replace(' The necessary APIs are in the knowledge graph.', '')
    return text


def render(template, card, context, previous):
    system, user, full = render_original(template, card, context, previous)
    if not context.strip():
        marker = '## API reference material (DOLFINx 0.10 API)\n'
        user = user.replace(marker, '', 1).lstrip()
        full = full.replace(marker, '', 1)
    return system, user, full


def tasks():
    result=[]
    for repeat in range(1,21):
        for case_index,(case,folder) in enumerate(CASES):
            shift=(repeat-1+case_index)%len(ARMS)
            arms=ARMS[shift:]+ARMS[:shift]
            for model in MODELS:
                for arm in arms:
                    result.append(dict(model=model['folder'],case=case,case_folder=folder,graph=arm,repeat=repeat,
                        directory=str(RESULTS/model['folder']/arm/folder/f'run_{repeat:02d}')))
    return result


def preflights():
    def probe(model):
        value=preflight(model,HERE/'preflight'/model['folder'])
        if value.get('status')!='ready' or value.get('reasoning_label')!='low':
            raise RuntimeError('Low-reasoning preflight did not pass: '+model['folder']+' '+str(value.get('status')))
        print(json.dumps({'model':model['folder'],'status':value['status'],'config':value['config']}),flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(probe,MODELS))


def freeze():
    path=HERE/'protocol.json'
    if path.exists():
        value=read(path);legacy.verify(value);return value
    provenance={}
    def copy(src,dst):
        dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst);provenance[str(src)]=legacy.sha(src)
    copy(RESULTS/'setting.txt',INPUTS/'setting.txt')
    for filename in TEMPLATES:
        src=ROOT/'prompt_templates/no_card_descriptions'/filename
        provenance[str(src)]=legacy.sha(src)
        target=INPUTS/'templates'/filename;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(neutral_template(src.read_text()))
    for case,_ in CASES:
        src=ROOT/'results_for_paper/benchmarks'/case;dst=INPUTS/'cases'/case
        copy(src/'model.msh',dst/'model.msh');copy(src/'prompt.txt',dst/'prompt.txt')
        for stage,filename in zip(STAGES,CARD_FILES):
            source=src/'cards'/filename;card=read(source);provenance[str(source)]=legacy.sha(source)
            if stage=='mesh':card['mesh']['path']=str((dst/'model.msh').resolve())
            write(dst/'cards'/filename,card)
    canonical=None
    for arm in ARMS:
        folder=INPUTS/arm;folder.mkdir(parents=True,exist_ok=True)
        if arm=='Card_Only':
            for stage in STAGES:(folder/(stage+'.txt')).write_text('')
            continue
        source=(ROOT/'results_for_paper/document_rag') if arm=='Document_Top3' else (ROOT/'results_for_paper/knowledge_graph'/arm.replace('KG_', 'graph_')/'graph')
        config=read(source/'retrieval_config.json')
        ranks=read(source/'retrieved_contexts/rankings.json')
        names={v:k for k,v in read(source/'review_id_map.json').items()} if arm!='Document_Top3' else None
        named={s:[[names[h['id']] if names else h['id'] for h in q['top_3']] for q in qs] for s,qs in ranks.items()}
        if canonical is None:canonical=named
        else:assert named==canonical,'Seed ranks differ between arms'
        for stage,filename in zip(STAGES,TEMPLATES):
            text=(INPUTS/'templates'/filename).read_text()
            assert re.findall(r'^\d+\. (.+)$',text,re.M)==config['procedure_queries'][stage]
            copy(source/'retrieved_contexts'/(stage+'.txt'),folder/(stage+'.txt'))
        for filename in ['retrieval_config.json','retrieved_contexts/rankings.json','openai_embeddings/api_embeddings.npz']:
            copy(source/filename,folder/filename)
        if arm!='Document_Top3':copy(source/'coder_graph.json',folder/'coder_graph.json')
    model_configs={}
    for model in MODELS:
        evidence=read(HERE/'preflight'/model['folder']/'preflight.json')
        assert evidence['status']=='ready' and evidence['reasoning_label']=='low'
        assert evidence['config']['temperature']==0
        model_configs[model['folder']]=evidence['config']
    tracked=[p for p in INPUTS.rglob('*') if p.is_file()]
    tracked += [Path(legacy.__file__),Path(worker.__file__),Path(__file__),
                ROOT/'experiments/kg_version_comparison/api.py',ROOT/'experiments/kg_version_comparison/parallel.py',
                ROOT/'codegen/cards.py',ROOT/'codegen/runtime.py',ROOT/'experiments/source_type_graph/repair_feedback.py',
                ROOT/'common/models.py']
    tracked += list((HERE/'preflight').rglob('preflight.json'))
    value={'created_at':now(),'study':'glm_four_arm_runtime_aligned_20261114','models':MODELS,'arms':ARMS,
        'cases':CASES,'repeats':20,'planned_workflows':960,'workflows_per_model':480,'workflows_per_model_arm':120,
        'tasks':tasks(),'model_configs':model_configs,'temperature':0,'reasoning':'low',
        'versions':{'dolfinx':'0.10.0','basix':'0.10.0','ufl':'2025.2.1'},
        'max_repairs_per_stage':5,'initial_attempts_per_stage':1,
        'repair_policy':'Same unchanged stage worker: retry current stage with original prompt, last code, runtime error and traceback. Earlier accepted stages are not rewritten.',
        'prompt_policy':'Same card, Procedure, Rules and cumulative prior code in all arms. Neutral API-reference labels replace KG-only claims. Empty reference section omitted for Card_Only. Existing template files remain unchanged.',
        'mesh_path_adjustment':'Only mesh path replaced by the same immutable local copy across arms; physical card values unchanged.',
        'retrieval_policy':'Identical 366 candidates, description embeddings and cosine top-3 per Procedure. RAG sends selected API chunks; KG adds existing one-hop plus conditional definition expansion.',
        'evaluation_mode':'execution_only','physics_correctness_asserted':False,
        'api_concurrency':MAX_ACTIVE,'per_model_concurrency':MAX_PER_MODEL,
        'execution_policy':'Existing CPU/memory admission; OpenMP and BLAS one thread per process.',
        'api_timeout_seconds':900,'runtime_timeout_seconds':1200,'hyperelastic_runtime_timeout_seconds':3600,
        'http_retry_policy':'Existing at most three HTTP submissions for 429 without completion; 60-second minimum retry wait. Other provider failures reported separately; unknown submissions not repeated automatically.',
        'timing_policy':'Record workflow wall time, successful API response time, failed-request time and memory-admission wait separately. Provider-internal waiting cannot be separated from response latency.',
        'usage_policy':'API-reported input/output/cached/reasoning tokens and OpenRouter cost. Reasoning tokens are a subset of output tokens. Missing usage remains null.',
        'reasoning_recording':'Save only reasoning/reasoning_details actually returned by the provider, along with raw responses.',
        'web_search':False,'tools':False,'source_sha256':provenance,'frozen_sha256':{str(p):legacy.sha(p) for p in tracked}}
    assert len(value['tasks'])==960
    write(path,value);write(RESULTS/'GLM_four_arm_manifest.json',value)
    for model in MODELS:
        for arm in ARMS:
            target=RESULTS/model['folder']/arm
            target.mkdir(parents=True,exist_ok=True)
            write(target/'settings.json',{'study':value['study'],'protocol':str(path),'model_config':model_configs[model['folder']],
                  'arm':arm,'repeats_per_case':20,'workflows':120,'evaluation_mode':'execution_only'})
    for task in value['tasks']:Path(task['directory']).mkdir(parents=True,exist_ok=True)
    return value


def report(protocol):
    rows=[]
    for task in protocol['tasks']:
        folder=Path(task['directory']);file=folder/'result.json'
        if not file.exists():file=folder/'state.json'
        value=read(file) if file.exists() else {}
        row={k:task[k] for k in ('model','graph','case','repeat')}
        row.update({k:value.get(k) for k in ('stage','wall_seconds','api_response_seconds','api_error_seconds','input_tokens','output_tokens','cached_tokens','reasoning_tokens','cost_usd','repair_calls')})
        row['status']=value.get('status','queued');rows.append(row)
    counts=dict(Counter(r['status'] for r in rows))
    value={'updated_at':now(),'planned':len(rows),'counts':counts,'rows':rows}
    write(HERE/'summary.json',value);write(RESULTS/'GLM_four_arm_summary.json',value)
    with (RESULTS/'GLM_four_arm_summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return counts


def credit():
    req=urllib.request.Request('https://openrouter.ai/api/v1/credits',headers={'Authorization':'Bearer '+credentials('openrouter')})
    with urllib.request.urlopen(req,timeout=30) as response:data=json.load(response)['data']
    return float(data['total_credits'])-float(data['total_usage'])


def run(protocol, *, prepare_task=None):
    with (HERE/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return run_locked(protocol, prepare_task)


def run_locked(protocol, prepare_task):
    legacy.verify(protocol)
    legacy.HERE,legacy.INPUTS=HERE,INPUTS
    worker.render=render
    worker.execution_slot=lambda stopped:nullcontext()
    execute=worker.execute_stage;admission=MemoryAdmission();stopped=lambda:(HERE/'STOP').exists()
    def execute_with_admission(source,prefix,directory,**kwargs):
        state_file=Path(source).parents[2]/'state.json'
        def waiting():
            state=read(state_file)
            if state.get('status')!='waiting_for_memory':write(state_file,{**state,'status':'waiting_for_memory','updated_at':now()})
        with admission.slot(source,kwargs.get('stage'),stopped,waiting) as delay:
            state=read(state_file);write(state_file,{**state,'status':'executing','updated_at':now()})
            write(Path(directory)/'admission.json',{'started_at':now(),'memory_wait_seconds':delay})
            return execute(source,prefix,directory,**kwargs)
    worker.execute_stage=execute_with_admission
    def work(task):
        try:
            if prepare_task is not None:prepare_task(task)
            return worker.run_workflow(task,protocol['model_configs'][task['model']],protocol)
        except Exception as error:
            if not task.get('directory'):raise
            result={**task,'status':'harness_error','error':str(error),'traceback':traceback.format_exc(),'finished_at':now()}
            write(Path(task['directory'])/'result.json',result);return result
    pending=[t for t in protocol['tasks'] if not t.get('directory') or not (Path(t['directory'])/'result.json').exists()]
    running={};last_report=last_credit=0;balance=None;enabled=False;blocked_balance=None
    started_at=now()
    with ThreadPoolExecutor(max_workers=MAX_ACTIVE) as pool:
        while pending or running:
            if stopped():pending.clear()
            if time.monotonic()-last_credit>60:
                try:
                    balance=credit();enabled=balance>0 and (blocked_balance is None or balance>blocked_balance+0.01)
                    write(HERE/'credits.json',{'checked_at':now(),'available_usd':balance,'requests_enabled':enabled})
                except Exception as error:
                    enabled=False;write(HERE/'credits.json',{'checked_at':now(),'error':str(error)[:500]})
                last_credit=time.monotonic()
            counts=Counter(t['model'] for t in running.values())
            if enabled and not stopped():
                for task in list(pending):
                    if len(running)>=MAX_ACTIVE:break
                    if counts[task['model']]>=MAX_PER_MODEL:continue
                    running[pool.submit(work,task)]=task;counts[task['model']]+=1
                    pending=[entry for entry in pending if entry is not task]
            if time.monotonic()-last_report>10:
                report(protocol)
                write(HERE/'controller.json',{'pid':os.getpid(),'started_at':started_at,'updated_at':now(),
                    'status':'stopping' if stopped() else 'running' if enabled else 'waiting_for_credits',
                    'active':list(running.values()),'pending':len(pending),'available_credit_usd':balance})
                last_report=time.monotonic()
            done,_=wait(running,timeout=1,return_when=FIRST_COMPLETED) if running else ([],[])
            if not running and pending:time.sleep(1)
            for future in done:
                task=running.pop(future);result=future.result()
                print(json.dumps({k:result.get(k) for k in ('model','case','graph','experiment_timestamp','status','wall_seconds')}),flush=True)
                if result.get('status')=='provider_error':
                    receipts=[read(p) for p in Path(task['directory']).glob('[1-5]_*/attempt_*/api_call.json')]
                    if any(r.get('status_code')==402 for r in receipts):enabled=False;blocked_balance=balance
    counts=report(protocol)
    write(HERE/'controller.json',{'pid':os.getpid(),'status':'stopped' if stopped() else 'finished','updated_at':now(),'counts':counts})

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['preflight','prepare','run','status']);args=parser.parse_args()
    if args.command=='preflight':preflights()
    elif args.command=='prepare':print(json.dumps({'planned':freeze()['planned_workflows']}))
    elif args.command=='run':run(freeze())
    else:print(json.dumps(report(read(HERE/'protocol.json')),indent=2))
