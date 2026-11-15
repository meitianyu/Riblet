import hashlib
from functools import lru_cache
import numpy as np
import model as m
import causal_graph as g
import plaintext_numeric as p
from layout import LAYOUT,physical_length


def kind(point):return 0 if point==0 else 1+(point-1)%3


def parameters(params,point):
    if point==0:return params['bert']['embeddings']
    layer=params['bert']['encoder']['layer'][str((point-1)//3)]
    if kind(point)==1:return layer['attention']
    if kind(point)==2:return layer['attention']['output']['LayerNorm']
    return {'intermediate':layer['intermediate'],'output':layer['output']}


def prepare(k,h,weights,ids,segments,mask,seed):
    if k==0:return g.prepare_embedding({'bert':{'embeddings':weights}},ids,segments,seed)
    if k==1:return g.prepare_softmax(h,weights,seed,mask)
    ctx=p.Arithmetic(seed)
    if k==3:h=m.feedforward(ctx,h,weights)
    return g.prepare_ln(ctx,h)


def finish(k,prep,weights,selection,counts,diagnostics=False):
    if k==1:
        h,record=g.finish_softmax(prep,weights,selection,counts=counts)
        if not diagnostics:record.pop('normalization_output')
        return h,record
    w=weights['LayerNorm'] if k==0 else weights['output']['LayerNorm'] if k==3 else weights
    return g.finish_ln(prep,w,selection,counts=counts)


def full(k,h,weights,ids,segments,mask,seed):
    prepared=prepare(k,h,weights,ids,segments,mask,seed)
    return finish(k,prepared,weights,None,(prepared['counter'],prepared['truncations']))


def padded_inputs(row):

    n=row['effective_length']
    ids=np.asarray(row['input_ids'],np.int32)[None,:]
    segments=np.asarray(row['token_type_ids'],np.int32)[None,:]
    if ids.shape!=(1,n):raise ValueError('effective length differs from actual token count')
    m.validate_b1(ids,segments)
    mask=np.ones_like(ids)
    if LAYOUT=='pad128':
        width=physical_length(n)-n
        ids=np.pad(ids,((0,0),(0,width)));segments=np.pad(segments,((0,0),(0,width)))
        mask=np.pad(mask,((0,0),(0,width)));m.validate_physical_b1(ids,segments,mask)
    return ids,segments,mask

MASK=(1<<64)-1

def mix_point_seed(seed,point):
    if not 0<=point<37:raise ValueError('point outside model')
    return np.uint64(int(seed) if point==0 else int(seed)^(((point+1)*0xD1B54A32D192ED03)&MASK))

@lru_cache(maxsize=16384)
def _request_seed(digest):
    return int.from_bytes(hashlib.sha256(('CDI_G9_V2_DESIGN_0\0'+digest).encode()).digest()[:8],'little')

def request_point_seed(row,point):return mix_point_seed(_request_seed(row['request_digest']),point)

def request_queues(indices,rows,size=128):
    groups={}
    for i in indices:groups.setdefault(physical_length(rows[i]['effective_length']),[]).append(i)
    for group in groups.values():
        for first in range(0,len(group),size):yield group[first:first+size]

def restore_order(indices,values):
    if len(indices)!=len(values) or len(set(indices))!=len(indices):raise ValueError('duplicate/misaligned request outputs')
    order=sorted(range(len(indices)),key=indices.__getitem__)
    return [indices[i] for i in order],[values[i] for i in order]

class BatchedKernels:
    def __init__(self,batch_size=8,diagnostics=False):
        import jax
        self.jax=jax;self.batch_size=batch_size;self.diagnostics=diagnostics;self.inputs={}
        self.prepare=jax.jit(jax.vmap(prepare,in_axes=(None,0,None,0,0,0,0)),static_argnums=(0,))
        self.finish=jax.jit(jax.vmap(finish,in_axes=(None,0,None,0,None,None)),static_argnums=(0,4,5))
        self.classify=jax.jit(jax.vmap(m.classify,in_axes=(0,None,0)))
    def _stack(self,items):
        if all(x is None for x in items):return None
        return self.jax.tree_util.tree_map(lambda *xs:np.stack(xs),*items)
    def _unstack(self,value,count):
        value=self.jax.device_get(value)
        return [self.jax.tree_util.tree_map(lambda x:x[i],value) for i in range(count)]
    def prepare_many(self,point,hidden,params,rows):
        if len(hidden)!=len(rows):raise ValueError('unaligned prepare queue')
        if len({physical_length(r['effective_length']) for r in rows})>1:raise ValueError('batch mixes physical shapes')
        result=[];weights=parameters(params,point)
        for first in range(0,len(rows),self.batch_size):
            rr=rows[first:first+self.batch_size];hh=hidden[first:first+self.batch_size];count=len(rr)
            rr=rr+[rr[-1]]*(self.batch_size-count);hh=hh+[hh[-1]]*(self.batch_size-count)
            inputs=[]
            for row in rr:
                key=(row['request_digest'],physical_length(row['effective_length']))
                if key not in self.inputs:self.inputs[key]=padded_inputs(row)
                inputs.append(self.inputs[key])
            ids,segments,mask=(np.stack([v[i] for v in inputs]) for i in range(3))
            seeds=np.asarray([request_point_seed(r,point) for r in rr],np.uint64)
            value=self.prepare(kind(point),self._stack(hh),weights,ids,segments,mask,seeds)
            result.extend(self._unstack(value,count))
        return result
    def finish_many(self,point,prepared,params,choices):
        if len(prepared)!=len(choices):raise ValueError('unaligned finish queue')
        result=[];weights=parameters(params,point)
        for first in range(0,len(prepared),self.batch_size):
            pp=prepared[first:first+self.batch_size];ss=choices[first:first+self.batch_size];count=len(pp)
            counts={(int(v['counter']),int(v['truncations'])) for v in pp}
            if len(counts)!=1 or len({v is None for v in ss})!=1:raise ValueError('mixed numerical counter/selection templates')
            pp=pp+[pp[-1]]*(self.batch_size-count);ss=ss+[ss[-1]]*(self.batch_size-count)
            value=self.finish(kind(point),self._stack(pp),weights,self._stack(ss),next(iter(counts)),self.diagnostics)
            result.extend(self._unstack(value,count))
        return result
    def classify_many(self,hidden,weights,rows):
        result=[]
        for first in range(0,len(rows),self.batch_size):
            rr=rows[first:first+self.batch_size];hh=hidden[first:first+self.batch_size];count=len(rr)
            rr=rr+[rr[-1]]*(self.batch_size-count);hh=hh+[hh[-1]]*(self.batch_size-count)
            seeds=np.asarray([_request_seed(r['request_digest']) for r in rr],np.uint64)
            value=self.classify(self._stack(hh),weights,seeds);result.extend(self._unstack(value,count))
        return result
