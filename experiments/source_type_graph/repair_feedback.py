"""Deterministic repair diagnostics; no model call or numerical solution data."""

from __future__ import annotations

import json
from pathlib import PurePath
import re


FRAME=re.compile(r'^\s*File "(.+)", line ([0-9]+)(?:, in (.+))?$')
EXCEPTION=re.compile(r'^[A-Za-z_][\w.]*(?:Error|Exception|Interrupt|Exit):|^(?:Exception|Error):')


def runtime_feedback(error,code,prefix_lines=1):
    lines=error.splitlines();frames=[]
    for i,line in enumerate(lines):
        match=FRAME.match(line)
        if not match:continue
        frames.append({'file':match[1],'line':int(match[2]),'function':match[3],
                       'source':lines[i+1].strip() if i+1<len(lines) else '','index':i})
    generated=[f for f in frames if f['file'].endswith('executable.py')]
    if not generated:
        # Native-process and unrecognized diagnostics keep their original text.
        return {'kind':'runtime_error','diagnostic':error},False
    caller=generated[-1];stage_line=caller['line']-prefix_lines
    result={'kind':'runtime_error','stage_line':stage_line,'executable_line':caller['line'],
            'source':caller['source']}
    source=code.splitlines()
    if 1<=stage_line<=len(source):
        result['context']=[{'line':i+1,'source':source[i]} for i in range(max(0,stage_line-2),min(len(source),stage_line+1))]
    # Keep the complete final native overload/signature diagnostic, not merely
    # its last line (which can be "Invoked with types: ...").
    final=frames[-1]
    tail=lines[final['index']+2:]
    if tail and re.match(r'^\s*[~^]+\s*$',tail[0]):tail=tail[1:]
    result['exception']='\n'.join(tail).strip()
    if final['file']!=caller['file']:
        file=final['file'].split('site-packages/')[-1]
        result['library']={'file':file,'line':final['line'],'source':final['source']}
    causes=[line for line in lines[:final['index']] if EXCEPTION.match(line)]
    if causes:result['causes']=causes
    return result,True


def contract_feedback(validation,card):
    if validation.get('validation_error') or validation.get('harness_error'):
        raise ValueError('Validation infrastructure failed; do not ask the model to repair the validator')
    failed=validation.get('failed_checks') or []
    obligations=[]
    for name in failed:
        if name=='mesh_geometry_and_dimension':
            description='Load the supplied mesh at the card path and preserve its geometry, dimensions and prescribed coordinate units.'
        elif name in {'cell_tags','facet_tags'}:
            description='Retain the imported '+name+' marker table.'
        elif name.startswith('region:'):
            description='Retain the card region and its actual dimension/tag association: '+name.split(':',1)[1]+'.'
        elif name=='tagged_measure:cell':
            description='Provide a volume integration measure on the imported mesh bound to its cell marker table.'
        elif name=='tagged_measure:exterior_facet':
            description='Provide an exterior-boundary integration measure on the imported mesh bound to its facet marker table.'
        else:
            role,_,field=name.partition(':')
            unknown=next((x for x in card.get('unknowns',[]) if x['name']==field),None)
            details=({k:unknown[k] for k in ['name','shape','function_space']} if unknown else {'name':field})
            description={'coefficient':'Prepare a persistent finite element coefficient field matching the card.',
                         'space':'Prepare a finite element space matching the card family, degree and value shape.',
                         'trial':'Prepare a trial argument with the required field shape on the coefficient field space.',
                         'test':'Prepare a test argument with the required field shape on the coefficient field space.'}.get(role,'Satisfy the supplied card condition.')
            obligations.append({'check':name,'requirement':description,'card':details});continue
        obligations.append({'check':name,'requirement':description})
    return {'kind':'card_contract_error','failed_requirements':obligations},False


def feedback(validation,code,prefix_lines,card):
    if validation.get('runtime_pass'):
        return contract_feedback(validation,card)
    text=validation.get('error') or validation.get('harness_error') or ''
    if not text:raise ValueError('No runtime diagnostic available')
    return runtime_feedback(text,code,prefix_lines)


def compose(original_prompt,code,diagnostic):
    return (original_prompt+'\n\n## Last attempt\n```python\n'+code+
            '\n```\n\n## Execution or card-contract error\n```json\n'+
            json.dumps(diagnostic,ensure_ascii=False,separators=(',',':'))+
            '\n```\nReturn complete corrected Python source for this stage only. Preserve the supplied card and earlier-stage objects.')
