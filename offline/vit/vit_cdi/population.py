from collections import defaultdict
from pathlib import PurePosixPath
import re

SELECTION_POLICY='sorted_synset_then_lexicographic_relative_id_first10_rank_mod5_v1'


def freeze_identities(rows):
    groups=defaultdict(list);seen=set();synsets={}
    for raw in rows:
        r=dict(raw);c=r.get('class_id');s=r.get('synset');name=r.get('relative_id','')
        p=PurePosixPath(name)
        if r.get('split')!='train' or isinstance(c,bool) or not isinstance(c,int) or not 0<=c<1000:
            raise ValueError('train split and official class ID0..999 required')
        if not isinstance(s,str) or not re.fullmatch(r'n[0-9]{8}',s):raise ValueError('bad synset')
        if p.is_absolute() or len(p.parts)!=2 or p.parts[0]!=s or '..' in p.parts or '\\' in name:
            raise ValueError('canonical synset/image identity required')
        if 'ILSVRC2012_val_' in name or 'ILSVRC2012_test_' in name:
            raise ValueError('evaluation image cannot enter Development')
        if name in seen:raise ValueError('duplicate original identity')
        seen.add(name)
        if synsets.setdefault(c,s)!=s:raise ValueError('class mapping drift')
        groups[c].append(dict(relative_id=name,class_id=c,synset=s,split='train'))
    if set(groups)!=set(range(1000)) or len(set(synsets.values()))!=1000:
        raise ValueError('all1000 classes must be present')
    if [synsets[c] for c in range(1000)]!=sorted(synsets.values()):
        raise ValueError('class IDs must follow the canonical sorted synset mapping')
    result=[]
    for c in range(1000):
        group=sorted(groups[c],key=lambda r:r['relative_id'])
        if len(group)<10:raise ValueError('fewer than10 images in class')
        for rank,r in enumerate(group[:10]):
            result.append(dict(r,request_ordinal=len(result),fold_id=rank%5,class_rank=rank))
    return result
