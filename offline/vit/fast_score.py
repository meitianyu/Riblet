import ctypes,hashlib,os,subprocess,tempfile
from pathlib import Path
import numpy as np
from vit_cdi import numeric as n
_LIB=None


def library():
    global _LIB
    if _LIB is None:
        source=Path(__file__).with_suffix('.cc');inc=Path(n.__file__).parent
        sha=hashlib.sha256(source.read_bytes()+(inc/'exact_predicate.cc').read_bytes()).hexdigest()[:16]
        root=Path(tempfile.gettempdir())/'vit-cdi-fast'/sha;root.mkdir(parents=True,exist_ok=True);target=root/'fast.so'
        if not target.exists():
            tmp=root/f'fast-{os.getpid()}.so'
            subprocess.run(['g++','-O3','-std=c++17','-fPIC','-shared','-fopenmp','-Wno-deprecated-declarations',str(source),'-I',str(inc),'-lcrypto','-o',str(tmp)],check=True)
            os.replace(tmp,target)
        _LIB=ctypes.CDLL(str(target));fn=_LIB.vit_clear_trajectory
        fn.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int64,ctypes.c_int64,ctypes.c_int64,ctypes.c_int,ctypes.c_int,ctypes.c_char_p,ctypes.c_int]+[ctypes.c_void_p]*7
        fn.restype=ctypes.c_int
    return _LIB


def replay(operator,x,a,b,*,event_ids,scope,maximum_k=None):
    if operator not in n.MAX_K:raise ValueError('invalid operator')
    k=n.MAX_K[operator] if maximum_k is None else maximum_k
    if type(k) is not int or not 0<=k<=n.MAX_K[operator]:raise ValueError('invalid K')
    x=np.asarray(x);ids=np.asarray(event_ids)
    if x.dtype!=np.int64 or x.ndim!=1 or ids.dtype!=np.uint64 or ids.shape!=x.shape:raise ValueError('unaligned input')
    if not isinstance(scope,str) or not scope:raise ValueError('scope required')
    if any(type(v) is not int or not -2**63<=v<2**63 for v in (a,b)):raise ValueError('bad coefficient')
    x=np.ascontiguousarray(x);ids=np.ascontiguousarray(ids);shape=(k+1,len(x))
    y=np.empty(shape,np.int64);v=np.empty(shape,np.uint8);d=np.empty(shape,np.uint8);p13=np.empty(shape,np.uint8);e=np.empty(shape,np.uint8)
    first=np.empty(len(x),np.uint8);fk=np.empty(len(x),np.int8)
    prefix=f'{n.NAMESPACE}\0{scope}\0'.encode()
    rc=library().vit_clear_trajectory(x.ctypes.data,ids.ctypes.data,len(x),a,b,int(operator=='rsqrt'),k,prefix,len(prefix),
                *(arr.ctypes.data for arr in [y,v,d,p13,e,first,fk]))
    if rc:raise RuntimeError('compiled trajectory failed')
    return n.Trajectory(y,v.astype(bool),d.astype(bool),p13.astype(bool),e,first,fk)
