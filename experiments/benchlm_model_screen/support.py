"""Provider capability checks, process-safe execution slots and posthoc checks."""
from pathlib import Path
from contextlib import contextmanager
import fcntl, json, os, time

ROOT=Path(__file__).resolve().parents[2]
BASE=Path(__file__).resolve().parent


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n');tmp.replace(path)


@contextmanager
def execution_slot(stopped=lambda:False):
    handles=[];selected=None
    try:
        for index in range(2):handles.append(open(f'/tmp/mcpsim_benchlm_execution_{index}.lock','a'))
        while selected is None:
            if stopped():raise InterruptedError('Study stopped while waiting for an execution slot.')
            for handle in handles:
                try:fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB);selected=handle;break
                except BlockingIOError:pass
            if selected is None:time.sleep(.2)
        yield
    finally:
        if selected is not None:fcntl.flock(selected.fileno(),fcntl.LOCK_UN)
        for handle in handles:handle.close()


def install_execution_limit():
    import codegen.runtime as runtime
    original=runtime.execute_stage
    def limited(*args,**kwargs):
        with execution_slot(kwargs.get('stopped') or (lambda:False)):return original(*args,**kwargs)
    runtime.execute_stage=limited


def preflight(row):
    import urllib.request
    from common.models import credentials
    from openai import OpenAI
    key=row['key'];folder=BASE/'output'/key
    if (folder/'preflight.json').exists():return json.loads((folder/'preflight.json').read_text())
    endpoint_file=folder/'endpoints.json'
    try:
        with urllib.request.urlopen('https://openrouter.ai/api/v1/models/'+row['model']+'/endpoints',timeout=45) as r:endpoints=json.load(r)
        write(endpoint_file,endpoints)
        available=[x for x in endpoints['data']['endpoints'] if x.get('status')==0 and 'temperature' in x.get('supported_parameters',[])]
        if not available:raise ValueError('No active temperature-capable endpoint.')
        caps=[x.get('max_completion_tokens') for x in available if x.get('max_completion_tokens')]
        max_output=min(65536,max(caps)) if caps else 8192
        active=[x for x in available if not x.get('max_completion_tokens') or x['max_completion_tokens']>=max_output]
        advertised=row['reasoning_advertised'] and any('reasoning' in x.get('supported_parameters',[]) for x in active)
        levels=[('low',{'effort':'low'}),('minimal',{'effort':'minimal'}),('minimum_budget_1024',{'max_tokens':1024}),('none',None)] if advertised else [('not_supported',None)]
        provider={'allow_fallbacks':True,'require_parameters':True}
        # Maintain at least FP8 where available; explicit NVFP4 CSV entries retain their stated model precision.
        if 'NVFP4' not in row['csv_name'] and any(x.get('quantization') in ('fp8','bf16','fp16','fp32') for x in active):provider['quantizations']=['fp8','bf16','fp16','fp32']
        probes=[];chosen=None
        with OpenAI(api_key=credentials('openrouter'),base_url='https://openrouter.ai/api/v1',timeout=100,max_retries=0) as client:
            for label,reasoning in levels:
                request={'model':row['model'],'messages':[{'role':'user','content':'Reply OK.'}],'temperature':0,
                         'max_tokens':min(max_output,4096),'extra_body':{'provider':provider}}
                if reasoning:request['extra_body']['reasoning']=reasoning
                try:
                    response=client.chat.completions.create(**request).model_dump(mode='json')
                    if response.get('error'):raise ValueError(str(response['error']))
                    probes.append({'request':request,'response':response,'excluded_from_coding_metrics':True})
                    chosen=(label,reasoning);break
                except Exception as error:
                    probes.append({'request':request,'status_code':getattr(error,'status_code',None),'error':str(error)[:4000]})
                    if getattr(error,'status_code',None) not in (400,404,422):raise
        if chosen is None:raise ValueError('No temperature=0 configuration accepted.')
        label,reasoning=chosen
        evidence={'created_at':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
                  'status':'ready','model_record':row,'max_output_tokens':max_output,'effective_reasoning_label':label,'effective_reasoning':reasoning,
                  'models':[{'id':row['model'],'provider':provider,'reasoning':reasoning}],
                  'sampling_probe':{'temperature_zero_supported_with_low':True},'probes':probes,'new_coder_calls':0}
        write(folder/'preflight.json',evidence)
        return evidence
    except Exception as error:
        evidence={'status':'preflight_error','model_record':row,'error':str(error)[:4000],'new_coder_calls':0}
        write(folder/'preflight_error.json',evidence)
        return evidence


def review_model(key,holdout=False):
    import subprocess,sys
    folder=BASE/'output'/(key+'_holdout' if holdout else key)/key
    for arm in (('kg',) if holdout else ('no_kg','kg')):
        out=folder/arm
        protocol=json.loads((out/'protocol.json').read_text())
        for task in protocol['tasks']:
            trial=out/'runs'/task['id'];result=trial/'result.json'
            if not result.exists() or json.loads(result.read_text()).get('status')!='execution_success':continue
            review=trial/'posthoc';review.mkdir(exist_ok=True)
            output=review/'assessment.json'
            if output.exists():continue
            env=os.environ.copy()
            for name in ('OPENAI_API_KEY','OPENROUTER_API_KEY','CODER_API_KEY'):env.pop(name,None)
            env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',XDG_CACHE_HOME=str(review/'cache'))
            with (review/'stdout.txt').open('w') as stdout,(review/'stderr.txt').open('w') as stderr:
                try:
                    subprocess.run([sys.executable,'-B',str(BASE/'posthoc.py'),'--snapshot',str(out/'inputs/cases'/task['case']/'snapshot.json'),
                                    '--source',str(trial/'final.py'),'--output',str(output)],cwd=review,env=env,stdout=stdout,stderr=stderr,timeout=1200,check=False)
                except subprocess.TimeoutExpired:write(output,{'status':'review_unavailable','error':'Posthoc replay timed out.'})
            if not output.exists():write(output,{'status':'review_unavailable','error':'Posthoc replay failed before an assessment was written.'})
