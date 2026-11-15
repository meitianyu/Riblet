import numpy as np
from vit_cdi import layout

def compact_index(op,site,condition):
    layout.public_index(op,site,condition) 
    if op=='reciprocal':return None if condition=='CLS' else site*36+int(condition[4:])
    return 5184+site*10+layout.conditions(op).index(condition)

def pack_coefficients(records):
    if len(records)!=layout.SLOTS or [r['index'] for r in records]!=list(range(layout.SLOTS)):raise ValueError('logical table must close')
    out=np.zeros((5434,2),np.uint64)
    for r in records:
        if r['index']!=layout.public_index(r['operator'],r['site'],r['condition']):raise ValueError('logical identity')
        index=compact_index(r['operator'],r['site'],r['condition'])
        if index is None:
            if r['route']!='SPU_FALLBACK':raise ValueError('CLS reciprocal cannot deploy CDI')
            continue
        if r['route']=='CDI_NORMAL':
            for j,key in enumerate(('a_encoded','b_encoded')):
                v=r[key]
                if type(v) is not int or not -(1<<63)<=v<(1<<63):raise ValueError('coefficient word')
                out[index,j]=v%2**64
        elif r['route']!='SPU_FALLBACK':raise ValueError('route')
    return out
