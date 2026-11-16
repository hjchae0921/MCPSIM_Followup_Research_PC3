"""Validate exports and compare fields; never send reference values to the coder."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree


def validate(folder,unknowns):
    folder=Path(folder)
    manifest=json.loads((folder/'manifest.json').read_text())
    fields=manifest['fields']
    expected=[{'name':u['name'],'shape':u['shape'],'index':i} for i,u in enumerate(unknowns)]
    if fields!=expected: raise ValueError(f'manifest.fields must equal {expected!r}')
    with np.load(folder/'solution.npz',allow_pickle=False) as data:
        arrays={k:data[k].copy() for k in data.files}
    required={f'f{i}_{suffix}' for i in range(len(fields)) for suffix in ('points','values')}
    if set(arrays)!=required: raise ValueError(f'NPZ keys must equal {sorted(required)}')
    for f in fields:
        points,values=(arrays[f"f{f['index']}_{s}"] for s in ('points','values'))
        count=int(np.prod(f['shape'])) if f['shape'] else 1
        if points.ndim!=2 or points.shape[1]!=3 or values.shape!=(len(points),count) or not len(points):
            raise ValueError(f"Invalid points/values shapes for {f['name']}: {points.shape}, {values.shape}")
        if points.dtype.kind!='f' or values.dtype.kind not in 'fc': raise ValueError('Arrays must contain floating-point coordinates and real or complex field values')
        if not np.isfinite(points).all(): raise ValueError('Coordinates contain nonfinite entries')
    return fields,arrays


def compare(folder,unknowns,reference,nonlinear=False):
    fields,arrays=validate(folder,unknowns)
    reference=Path(reference)
    refmeta=json.loads(reference.with_name('summary.json').read_text())['fields']
    reports={}
    with np.load(reference,allow_pickle=False) as ref:
        for field in fields:
            match=[r for r in refmeta if r['name']==field['name'] and r['shape']==field['shape']]
            if len(match)!=1: raise ValueError('Reference field metadata does not match the card')
            r=match[0];pts=arrays[f"f{field['index']}_points"];values=arrays[f"f{field['index']}_values"]
            target=ref[f"f{r['index']}_points"];expected=ref[f"f{r['index']}_values"]
            if values.shape!=expected.shape:
                reports[field['name']]={'status':'not_comparable','reason':'Different point count or component shape'};continue
            distance,indices=cKDTree(pts).query(target)
            if distance.max()>1e-7 or len(np.unique(indices))!=len(indices):
                reports[field['name']]={'status':'not_comparable','reason':'No bijective coordinate match within 1e-7 mm'};continue
            actual=values[indices]
            if not np.isfinite(actual).all():
                reports[field['name']]={'status':'mismatch','reason':'Nonfinite field values'};continue
            delta=actual-expected
            norm=float(np.linalg.norm(expected));absolute=float(np.linalg.norm(delta))
            relative=absolute/max(norm,1e-15)
            # Keep the established temperature-increment normalization as well as raw relative L2.
            scale=max(float(np.linalg.norm(expected-293.15)),1.) if field['name']=='temperature' else max(norm,1e-15)
            error=absolute/scale;tolerance=2e-3 if nonlinear else 1e-5
            reports[field['name']]={'status':'matched' if error<tolerance else 'mismatch','relative_l2':relative,
                'normalized_error':error,'tolerance':tolerance,'maximum_absolute_error':float(np.abs(delta).max()),
                'normalization':'temperature minus 293.15 K' if field['name']=='temperature' else 'reference L2 norm'}
    status='matched' if all(r['status']=='matched' for r in reports.values()) else 'not_comparable' if any(r['status']=='not_comparable' for r in reports.values()) else 'mismatch'
    return {'status':status,'fields':reports,'criterion':'Reference field agreement only; no independent equilibrium or operator check'}
