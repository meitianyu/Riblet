import ctypes,hashlib,subprocess
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent
LIB=None
def library():
    global LIB
    if LIB is None:
        here=Path(__file__).parent
        key=hashlib.sha256((here/'kernel.cc').read_bytes()+(BASE/'fast_score.cc').read_bytes()+(BASE/'vit_cdi/exact_predicate.cc').read_bytes()).hexdigest()[:16]
        out=here/f'numeric-{key}.so'
        if not out.exists():subprocess.run(['g++','-std=c++17','-O3','-fPIC','-shared','-Wno-deprecated-declarations',str(here/'kernel.cc'),'-I',str(BASE),'-I',str(BASE/'vit_cdi'),'-lcrypto','-o',str(out)],check=True)
        LIB=ctypes.CDLL(str(out));fn=LIB.numeric_eval
        fn.argtypes=[ctypes.c_void_p]*2+[ctypes.c_int64]*3+[ctypes.c_int]*2+[ctypes.c_char_p]+[ctypes.c_int]*6+[ctypes.c_void_p]*9
        fn.restype=ctypes.c_int
    return LIB

def evaluate(x,ids,a,b,op,k,scope,*,aggregate=False,certified=False,reassoc=False,bits=18,scale=0,output='native',init_nearest=False,iter_nearest=False,oracle=False):
    x=np.ascontiguousarray(x,np.int64);ids=np.ascontiguousarray(ids,np.uint64)
    if x.ndim!=1 or ids.shape!=x.shape or not len(x):raise ValueError('event alignment')
    flags=int(reassoc)+2*int(init_nearest)+4*int(iter_nearest)+8*int(oracle)+16*int(output=='nearest')
    mask=np.zeros(1,np.uint32);initial=np.zeros(1,np.uint8);visited=np.zeros(1,np.uint64)
    failures=np.empty(k+1,np.uint64);why=np.empty(k+1,np.uint8)
    shape=(1,1) if aggregate else (k+1,len(x))
    y=np.empty(shape,np.int64);valid=np.empty(shape,np.uint8);passed=np.empty(shape,np.uint8);residual=np.empty(shape,np.float64)
    prefix=('VIT_CDI_G9_DESIGN_V1\0'+scope+'\0').encode()
    rc=library().numeric_eval(x.ctypes.data,ids.ctypes.data,len(x),a,b,int(op=='rsqrt'),k,prefix,len(prefix),flags,bits,scale,int(aggregate),int(certified),
                            *(v.ctypes.data for v in (mask,initial,visited,failures,why,y,valid,passed,residual)))
    if rc:raise RuntimeError(f'numeric evaluator {rc}')
    return dict(mask=int(mask[0]),initial_valid=bool(initial[0]),visited=int(visited[0]),failure_ids=failures.tolist(),failure_kind=why.tolist(),y=y,valid=valid,passed=passed,residual=residual)

def trace(*args,**kwargs):return evaluate(*args,**kwargs)
def score(*args,**kwargs):
    r=evaluate(*args,aggregate=True,**kwargs)
    return {k:v for k,v in r.items() if k not in ('y','valid','passed','residual')}
