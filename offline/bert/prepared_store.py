import json,os,re
from pathlib import Path
import numpy as np
from plaintext_numeric import Word

def _encode(tree,arrays,dead=()):
    if isinstance(tree,Word):return ['word',_encode(tree.value,arrays),_encode(tree.state,arrays)]
    if isinstance(tree,dict):
        entries=[]
        for key in sorted(tree):
            value=tree[key]
            if key in dead:
                if not isinstance(value,Word):raise ValueError('declared dead state must be a Word')
                node=['deadword',_encode(value.value,arrays)]
            else:node=_encode(value,arrays)
            entries.append([key,node])
        return ['dict',entries]
    a=np.asarray(tree)
    if a.dtype.hasobject:raise ValueError('object cache forbidden')
    index=len(arrays);arrays.append(a);return ['array',index,a.dtype.str,list(a.shape)]

def _decode(schema,arrays):
    if schema[0]=='word':return Word(_decode(schema[1],arrays),_decode(schema[2],arrays))
    if schema[0]=='deadword':
        value=_decode(schema[1],arrays);return Word(value,np.zeros_like(value,dtype=np.uint64))
    if schema[0]=='dict':return {key:_decode(s,arrays) for key,s in schema[1]}
    a=arrays[schema[1]]
    if a.dtype.str!=schema[2] or list(a.shape)!=schema[3]:raise ValueError('cache dtype/shape mismatch')
    return a

class PreparedStore:
    def __init__(self,root,binding,*,kind):
        self.root=Path(root).resolve();self.root.mkdir(parents=True,exist_ok=True)
        if kind not in (0,1,2,3):raise ValueError('BERT operator kind')
        self.binding=json.loads(json.dumps(binding,sort_keys=True));self.kind=kind
        self.dead=('exps','value') if kind==1 else ('centered',)
        self.path=self.root/'manifest.json'
        self.meta=dict(schema='bert-prepared-live-state-v1',binding=self.binding,kind=kind,shards=[],consumed=[])
        if self.path.exists():
            self.meta=json.loads(self.path.read_text())
            if self.meta['schema']!='bert-prepared-live-state-v1' or self.meta['binding']!=self.binding or self.meta['kind']!=kind:raise ValueError('cache binding changed')
        else:self._commit()
        self.seen={i for s in self.meta['shards'] for i in s['ids']}
        self.stats=dict(written_bytes=0,read_bytes=0,peak_loaded_rows=0,stored_rows=len(self.seen))
    def _commit(self):
        tmp=self.path.with_suffix('.tmp');tmp.write_text(json.dumps(self.meta,sort_keys=True));os.replace(tmp,self.path)
    def put(self,ids,rows):
        ids=list(map(int,ids));rows=list(rows)
        if not ids:return
        if len(ids)!=len(rows) or len(set(ids))!=len(ids) or self.seen.intersection(ids):raise ValueError('duplicate/unaligned prepared IDs')
        flat=[];schema=None
        for row in rows:
            arrays=[];s=_encode(row,arrays,self.dead)
            if schema is not None and schema!=s:raise ValueError('mixed physical shape in shard')
            schema=s;flat.append(arrays)
        data={f'a{i}':np.stack([r[i] for r in flat]) for i in range(len(flat[0]))}
        packed=[]
        for key,a in data.items():
            if a.dtype==np.int64 and a.size and a.min()>=-(1<<31) and a.max()<(1<<31):data[key]=a.astype(np.int32);packed.append(key)
        data['meta']=np.asarray(json.dumps(dict(binding=self.binding,kind=self.kind,ids=ids,schema=schema,packed=packed),sort_keys=True))
        name=f'shard-{len(self.meta["shards"]):06d}.npz';path=self.root/name;tmp=path.with_suffix('.partial')
        with tmp.open('xb') as f:np.savez(f,**data)
        os.replace(tmp,path)
        self.meta['shards'].append(dict(name=name,ids=ids));self._commit();self.seen.update(ids)
        self.stats['written_bytes']+=path.stat().st_size;self.stats['stored_rows']+=len(ids)
    def iter_shards(self):
        for row in self.meta['shards']:
            if row['name'] in self.meta['consumed']:continue
            path=self.root/row['name']
            with np.load(path,allow_pickle=False) as f:
                meta=json.loads(str(f['meta']))
                if meta['binding']!=self.binding or meta['kind']!=self.kind or meta['ids']!=row['ids']:raise ValueError('wrong prepared shard')
                arrays=[];packed=set(meta['packed'])
                for i in range(len(f.files)-1):
                    key=f'a{i}';a=f[key]
                    if key in packed:
                        if a.dtype!=np.int32:raise ValueError('lossless packed dtype mismatch')
                        a=a.astype(np.int64)
                    if len(a)!=len(row['ids']):raise ValueError('cache row count')
                    arrays.append(a)
            self.stats['read_bytes']+=path.stat().st_size
            self.stats['peak_loaded_rows']=max(self.stats['peak_loaded_rows'],len(row['ids']))
            yield row['ids'],[_decode(meta['schema'],[a[j] for a in arrays]) for j in range(len(row['ids']))],path
    def consume(self,path):
        path=Path(path).resolve()
        names={s['name'] for s in self.meta['shards']}
        if path.parent!=self.root or not re.fullmatch(r'shard-\d{6}\.npz',path.name) or path.name not in names:raise ValueError('unsafe scratch retirement')
        if path.name in self.meta['consumed']:raise ValueError('already consumed shard')

        path.unlink();self.meta['consumed'].append(path.name);self._commit()
