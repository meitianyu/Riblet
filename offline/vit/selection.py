import hashlib
import numpy as np
from vit_cdi import layout,numeric
import abi


class SelectionPlan:
    def __init__(self,doc,point):
        self.point=point;heads=12 if point%3==1 else 1;op=doc['operator']
        if doc.get('graph')!=abi.GRAPH_ID:raise ValueError('wrong graph/parameter binding')
        sites={int(s['site']):s for s in doc['sites']};aa=[];bb=[];kk=[];on=[];seed_scopes=[]
        for head in range(heads):
            site=point//3*12+head if heads==12 else (24 if point==36 else 2*(point//3)+int(point%3==2))
            for pos in range(197):
                row=sites[site]['routes'][layout.condition(op,pos)];enabled=row['route']=='CDI_NORMAL'
                k=row['public_Ks']
                abi.check_row(op,row)
                aa.append(row['a_encoded'] if enabled else 0);bb.append(row['b_encoded'] if enabled else numeric.S)
                kk.append(k);on.append(enabled)
                if enabled:
                    prefix=f'{numeric.NAMESPACE}\0{abi.scope(op,site,5)}\0'.encode()
                    seed_scopes.append((len(aa)-1,head,pos,hashlib.sha256(prefix)))
        self.constants={k:np.asarray(v,t) for k,v,t in [('a',aa,np.int64),('b',bb,np.int64),('k',kk,np.int64),('enabled',on,bool)]}
        for v in self.constants.values():v.setflags(write=False)
        self.seed_scopes=seed_scopes

    def __call__(self,row):
        seeds=np.zeros(len(self.constants['a']),np.uint64);rid=int(row['request_ordinal'])
        for index,head,pos,base in self.seed_scopes:
            h=base.copy();h.update(str(layout.event_id(rid,self.point,head,pos)).encode())
            seeds[index]=int.from_bytes(h.digest()[:8],'little')
        return dict(self.constants,seeds=seeds)
