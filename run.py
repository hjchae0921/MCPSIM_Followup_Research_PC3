"""Run a model-owned shard of the frozen four-arm experiment on another Linux PC."""
import argparse
from collections import Counter
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import zipfile

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from experiments.glm_four_arm_20261114 import run as base
from experiments.postprocess_stage6_20261116 import run as extension
from experiments.postprocess_stage6_20261116 import continue_queue as continuation
from experiments.kg_version_comparison import worker
from experiments.kg_version_comparison.api import read,write,now,preflight

PLAN=read(ROOT/'assignment.json')
HOME=ROOT/'outputs'/PLAN['station']
INPUTS=HOME/'inputs'
POST=HOME/'postprocess'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_bundle():
    manifest=read(ROOT/'bundle_manifest.json')
    for name,digest in manifest['files'].items():
        if not (ROOT/name).is_file() or sha(ROOT/name)!=digest:raise ValueError('Bundle file changed or missing: '+name)


def environment():
    from mpi4py import MPI
    from petsc4py import PETSc
    expected=read(ROOT/'runtime_fingerprint.json')
    measured={name:importlib.metadata.version(name) for name in ('fenics-dolfinx','fenics-basix','fenics-ufl','fenics-ffcx','petsc4py','numpy','scipy')}
    measured['petsc']='.'.join(map(str,PETSc.Sys.getVersion()))
    for name,version in measured.items():
        if version!=expected[name]:raise ValueError(f'Runtime mismatch: {name}: {version}, expected {expected[name]}')
    if MPI.COMM_WORLD.size!=1:raise ValueError('Use one MPI rank; do not launch this controller through mpirun')
    if not PETSc.Sys.hasExternalPackage('mumps'):raise ValueError('The cards require PETSc with MUMPS')
    measured.update(mpi_library=MPI.Get_library_version(),has_mumps=True,mpi_size=1)
    import platform
    measured.update(platform=platform.platform(),machine=platform.machine(),logical_cpus=os.cpu_count())
    HOME.mkdir(parents=True,exist_ok=True);write(HOME/'runtime.json',measured)
    return measured


def prepare_inputs():
    source=ROOT/'frozen_inputs'
    for path in source.rglob('*'):
        if not path.is_file():continue
        target=INPUTS/path.relative_to(source);target.parent.mkdir(parents=True,exist_ok=True)
        content=path.read_bytes()
        if path.name=='1_mesh.json':
            card=json.loads(content);mesh=INPUTS/path.relative_to(source).parent.parent/'model.msh'
            card['mesh']['path']=str(mesh.resolve());content=(json.dumps(card,ensure_ascii=False,indent=2)+'\n').encode()
        if target.exists() and target.read_bytes()!=content:raise ValueError('Local input changed: '+str(target))
        if not target.exists():target.write_bytes(content)
    POST.mkdir(parents=True,exist_ok=True)
    for name in ['6_Postprocess_Prompt.txt','Card_Only.txt','Document_Top3.txt','KG_ver1.txt','KG_ver2.txt']:
        source=ROOT/'experiments/postprocess_stage6_20261116'/name
        target=POST/name
        if target.exists() and target.read_bytes()!=source.read_bytes():raise ValueError('Postprocess context changed')
        if not target.exists():shutil.copy2(source,target)
    write(HOME/'input_relocation.json',{'policy':'Only the mesh file path is relocated; all other card values, prompt text and retrieval contexts are unchanged.',
        'inputs':str(INPUTS),'source_bundle':sha(ROOT/'bundle_manifest.json')})


def tasks(model):
    result=[]
    for repeat in range(1,21):
        for i,(case,case_folder) in enumerate(base.CASES):
            shift=(repeat-1+i)%len(base.ARMS)
            for arm in base.ARMS[shift:]+base.ARMS[:shift]:
                result.append({'model':model['folder'],'graph':arm,'case':case,'case_folder':case_folder,'repeat':repeat,
                    'directory':str(HOME/'runs'/model['folder']/arm/case/f'run_{repeat:02d}')})
    return result


def report(_=None):
    rows=[]
    for model in PLAN['models']:
        for task in tasks(model):
            folder=Path(task['directory']);path=folder/'result.json'
            if not path.exists():path=folder/'state.json'
            state=read(path) if path.exists() else {'status':'pending'}
            post=POST/'items'/task['model']/task['graph']/task['case']/f"run_{task['repeat']:02d}"/'result.json'
            row={k:task[k] for k in ('model','graph','case','repeat')}
            row.update({k:state.get(k) for k in ('status','stage','wall_seconds','input_tokens','output_tokens','reasoning_tokens','cached_tokens','cost_usd','repair_calls')})
            if post.exists():
                data=read(post);row.update(postprocess_status=data['status'],reference_status=data.get('reference_status'))
            rows.append(row)
    counts=dict(Counter(r['status'] for r in rows))
    write(HOME/'summary.json',{'station':PLAN['station'],'updated_at':now(),'planned':len(rows),'counts':counts,'rows':rows})
    return counts


def run(workers):
    verify_bundle();environment();prepare_inputs()
    write(HOME/'execution_settings.json',{'workers':workers,'station':PLAN['station'],'started_at':now(),
        'timing_note':'Host hardware and concurrency affect wall time; provider-internal waiting remains part of API response latency.'})
    lock=(HOME/'station.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    base.INPUTS=INPUTS;base.RESULTS=HOME;base.MAX_ACTIVE=workers;base.MAX_PER_MODEL=workers;base.report=report
    extension.HERE=POST;continuation.HERE=POST
    worker.run_workflow=continuation.workflow
    for model in PLAN['models']:
        if (HOME/'STOP').exists():break
        control=HOME/'controls'/model['folder'];control.mkdir(parents=True,exist_ok=True)
        if all((Path(t['directory'])/'result.json').exists() and
               (read(Path(t['directory'])/'result.json')['status']!='execution_success' or
                (POST/'items'/t['model']/t['graph']/t['case']/f"run_{t['repeat']:02d}"/'result.json').exists()) for t in tasks(model)):
            continue
        path=control/'protocol.json'
        if path.exists():protocol=read(path)
        else:
            evidence=preflight(model,control/'preflight',gpt_temperature_exception=model['folder']=='GPT-5 mini')
            if evidence.get('status')!='ready':
                write(control/'blocked.json',evidence);continue
            blueprint=read(ROOT/'experiment_settings.json')
            frozen={str(ROOT/n):v for n,v in read(ROOT/'bundle_manifest.json')['files'].items()}
            frozen.update({str(f):sha(f) for f in INPUTS.rglob('*') if f.is_file()})
            protocol={**blueprint,'study':'four_arm_distributed_'+PLAN['station'],'models':[model],'tasks':tasks(model),
                'planned_workflows':480,'model_configs':{model['folder']:evidence['config']},'frozen_sha256':frozen,
                'station':PLAN['station'],'bundle_sha256':sha(ROOT/'bundle_manifest.json')}
            write(path,protocol)
        worker.execute_stage=continuation.retain_solve
        base.HERE=control
        continuation.resume_model(protocol)
        report()
    print(json.dumps(report(),indent=2))


def export():
    target=ROOT/(PLAN['station']+'_results.zip')
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for path in HOME.rglob('*'):
            if path.is_file() and not any(p in {'cache','__pycache__'} for p in path.relative_to(HOME).parts):
                archive.write(path,path.relative_to(ROOT))
    print(target)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['check','prepare','run','status','stop','export']);parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    if not 1<=args.workers<=4:parser.error('--workers must be between 1 and 4')
    if args.command=='check':verify_bundle();print(json.dumps(environment(),indent=2))
    elif args.command=='prepare':verify_bundle();prepare_inputs();print(json.dumps(report(),indent=2))
    elif args.command=='run':run(args.workers)
    elif args.command=='status':print(json.dumps(report(),indent=2))
    elif args.command=='export':export()
    elif args.command=='stop':
        HOME.mkdir(parents=True,exist_ok=True);(HOME/'STOP').touch();POST.mkdir(parents=True,exist_ok=True);(POST/'STOP').touch()
        for path in (HOME/'controls').glob('*'):
            if path.is_dir():(path/'STOP').touch()
