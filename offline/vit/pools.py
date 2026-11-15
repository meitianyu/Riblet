import numpy as np

def ordered(x,ids):
    lead=list(dict.fromkeys([int(x.argmin()),int(x.argmax())]+np.linspace(0,len(x)-1,min(256,len(x)),dtype=np.int64).tolist()))
    mask=np.ones(len(x),bool);mask[lead]=False;order=np.concatenate([np.asarray(lead,np.int64),np.flatnonzero(mask)])
    return np.ascontiguousarray(x[order]),np.ascontiguousarray(ids[order])

class Pool:
    def __init__(self,x,ids,folds):
        self.raws=[];self.folds=[]
        for f in range(6):
            train=np.ones(len(x),bool) if f==5 else folds!=f
            target=np.ones(len(x),bool) if f==5 else folds==f
            if not train.any() or not target.any():raise ValueError('empty OOF source/target')
            self.raws.append((int(x[train].min()),int(x[train].max())))
            xx,ee=ordered(x[target],ids[target]);xx.flags.writeable=False;ee.flags.writeable=False
            self.folds.append((xx,ee,(int(xx.min()),int(xx.max()))))

def source_raws(x,folds):
    return [(int(x[folds!=f].min()),int(x[folds!=f].max())) if f<5 else (int(x.min()),int(x.max())) for f in range(6)]
