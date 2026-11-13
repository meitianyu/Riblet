import hashlib,json
from pathlib import Path


def load_schedule():
    data = Path(__file__).with_name('vit_schedule.json').read_bytes()
    if hashlib.sha256(data).hexdigest() != '73290dd57de1875b7c25451e29af8ef6a258b04ab28bb0240402d31809ef26ed':
        raise ValueError('ViT geometry differs from the supported model')
    return json.loads(data)

def ours_fingerprints():


    result=[[2,[196,768],[768,768],[196,768]]]
    for _ in range(12):
        result.extend([
          [1,[197,768],[768,2304],[197,2304]],
          [3,[12,197,64],[12,64,197],[12,197,197]],
          [3,[12,64,197],[12,197,197],[12,64,197]],
          [1,[197,768],[768,768],[197,768]],
          [1,[197,768],[768,3072],[197,3072]],
          [1,[197,3072],[3072,768],[197,768]],
        ])
    result.append([1,[1,768],[768,1000],[1,1000]])
    return result

def fingerprint_text():
    return ';'.join(','.join(map(str,[family,len(lhs),*lhs,*rhs,*out])) for family,lhs,rhs,out in ours_fingerprints())

def plan_hash():
    doc={'native_logical_sites':98,'MJV_eligible_logical_consumers':97,
         'physical_calls':ours_fingerprints(),'model_rank':0,'client_rank':1,
         'model':'google/vit-base-patch16-224@3f49326eb077187dfe1c2a2bb15fbd74e6ab91e3'}
    return hashlib.sha256(json.dumps(doc,sort_keys=True,separators=(',',':')).encode()).hexdigest()
