import json,os
from pathlib import Path
import numpy as np
from vit_cdi.reference_numeric import Word


def _flatten(tree,arrays,dead_state_fields=()):
    if isinstance(tree,Word):return ['word',_flatten(tree.value,arrays),_flatten(tree.state,arrays)]
    if isinstance(tree,dict):
        entries=[]
        for k in sorted(tree):
            if k in dead_state_fields:
                if not isinstance(tree[k],Word):raise ValueError('dead-state declaration requires a Word')
                entry=['value_only_word',_flatten(tree[k].value,arrays)]
            else:entry=_flatten(tree[k],arrays)
            entries.append([k,entry])
        return ['dict',entries]
    if tree is None:return ['none']
    i=len(arrays);a=np.asarray(tree)
    if a.dtype.hasobject:raise ValueError('object cache forbidden')
    arrays.append(a);return ['array',i,a.dtype.str,list(a.shape)]


def _restore(schema,arrays):
    if schema[0]=='word':return Word(_restore(schema[1],arrays),_restore(schema[2],arrays))
    if schema[0]=='value_only_word':
        value=_restore(schema[1],arrays);return Word(value,np.zeros_like(value,dtype=np.uint64))
    if schema[0]=='dict':return {k:_restore(s,arrays) for k,s in schema[1]}
    if schema[0]=='none':return None
    a=arrays[schema[1]]
    if a.dtype.str!=schema[2] or list(a.shape)!=schema[3]:raise ValueError('cache dtype/shape mismatch')
    return a


def save(path,trees,binding,request_ids,*,dead_state_fields=()):


    if not set(dead_state_fields)<={'exps','value','centered'} or (dead_state_fields and binding.get('phase')!='prepared'):
        raise ValueError('unproved cache state omission')
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    rows=[];schema=None
    if len(trees)!=len(request_ids) or len(request_ids)!=len(set(request_ids)):raise ValueError('cache request identity mismatch')
    for tree in trees:
        arrays=[];s=_flatten(tree,arrays,dead_state_fields)
        if schema is not None and schema!=s:raise ValueError('heterogeneous cache shard')
        schema=s;rows.append(arrays)
    data={f'a{i}':np.stack([r[i] for r in rows]) for i in range(len(rows[0]) if rows else 0)}
    packed=[]
    for key,a in data.items():
        if a.dtype==np.int64 and a.size and a.min()>=-(1<<31) and a.max()<(1<<31):
            data[key]=a.astype(np.int32);packed.append(key)
    data['meta']=np.asarray(json.dumps(dict(binding=binding,request_ids=request_ids,schema=schema,lossless_int32_arrays=packed),sort_keys=True))
    tmp=path.with_suffix('.partial')
    with tmp.open('wb') as f:np.savez(f,**data)
    os.replace(tmp,path)


def load(path,binding):
    with np.load(path,allow_pickle=False) as f:
        meta=json.loads(str(f['meta']))
        if meta['binding']!=binding:raise ValueError('cache prefix/graph binding mismatch')
        packed=set(meta.get('lossless_int32_arrays',[]))
        arrays=[]
        for i in range(len(f.files)-1):
            key=f'a{i}';a=f[key]
            if key in packed:
                if a.dtype!=np.int32:raise ValueError('lossless cache storage tag mismatch')
                a=a.astype(np.int64)
            arrays.append(a)
    ids=meta['request_ids']
    if any(len(a)!=len(ids) for a in arrays):raise ValueError('cache length mismatch')
    return [_restore(meta['schema'],[a[j] for a in arrays]) for j in range(len(ids))],ids
