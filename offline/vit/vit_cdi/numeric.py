from dataclasses import dataclass
import ctypes
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
import numpy as np
from . import reference_numeric as p

S=p.S
MAX_K={'reciprocal':10,'rsqrt':9}
NAMESPACE='VIT_CDI_G9_DESIGN_V1'
ERRORS={0:'none',1:'nonpositive_operand',2:'nonpositive_initial_output',
        3:'initial_convergence_state',4:'raw_product_wrap',5:'addition_wrap',
        6:'signed_truncation_range',7:'nonpositive_newton_output',8:'other_protocol_exception'}
_LIB=None


def _precision_library():
    global _LIB
    if _LIB is None:
        source=Path(__file__).with_name('exact_predicate.cc')
        digest=hashlib.sha256(source.read_bytes()).hexdigest()[:16]
        root=Path(tempfile.gettempdir())/'vit-cdi-exact'/digest;root.mkdir(parents=True,exist_ok=True)
        target=root/'predicate.so'
        if not target.exists():
            tmp=root/f'predicate-{os.getpid()}.so'
            subprocess.run(['g++','-std=c++17','-O3','-DNDEBUG','-fPIC','-shared',str(source),'-o',str(tmp)],check=True)
            os.replace(tmp,target)
        _LIB=ctypes.CDLL(str(target))
        fn=_LIB.vit_precision_batch
        fn.argtypes=[ctypes.POINTER(ctypes.c_int64),ctypes.POINTER(ctypes.c_int64),ctypes.c_int64,
                     ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.POINTER(ctypes.c_uint8)]
        fn.restype=ctypes.c_int
    return _LIB


def precision(operator,x,y,bits=12,inclusive=False):
    x=np.asarray(x);y=np.asarray(y)
    if operator not in MAX_K or bits not in (12,13) or type(inclusive) is not bool:
        raise ValueError('invalid precision predicate')
    if x.dtype!=np.int64 or y.dtype!=np.int64 or x.ndim!=1 or y.shape!=x.shape:
        raise ValueError('aligned int64 vectors required')
    if os.name=='nt':
        m=1<<bits;out=[]
        for a,b in zip(x,y):
            a=int(a);b=int(b)
            if a<=0 or b<=0:out.append(False);continue
            if operator=='reciprocal':
                v=abs(S*S-a*b);limit=S*S//m;out.append(v<=limit if inclusive else v<limit)
            else:
                v=m*m*a*b*b;lo=(m-1)**2*S**3;hi=(m+1)**2*S**3
                out.append(lo<=v<=hi if inclusive else lo<v<hi)
        return np.asarray(out,bool)
    x=np.ascontiguousarray(x);y=np.ascontiguousarray(y);out=np.empty(len(x),np.uint8)
    result=_precision_library().vit_precision_batch(
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),y.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        len(x),int(operator=='rsqrt'),bits,int(inclusive),out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)))
    if result:raise RuntimeError('exact residual predicate rejected input')
    return out.astype(bool)


def event_seeds(scope,event_ids):
    if not isinstance(scope,str) or not scope:raise ValueError('stable candidate/fold scope required')
    prefix=f'{NAMESPACE}\0{scope}\0'.encode()
    return np.asarray([int.from_bytes(hashlib.sha256(prefix+str(int(i)).encode()).digest()[:8],'little')
                       for i in event_ids],np.uint64)


class ObservedArithmetic(p.Arithmetic):
    def __init__(self,seeds):
        super().__init__(seeds,per_event=True)
        self.first_error=np.zeros(np.shape(seeds),np.uint8)
        self.first_k=np.full(np.shape(seeds),-1,np.int8);self.k=0;self.operation=8

    def record(self,bad,code):
        first=np.asarray(bad)&(self.first_error==0)
        self.first_error[first]=code;self.first_k[first]=self.k

    def mark(self,bad,code):
        self.record(bad,6 if code==2 else 8 if code==8 else 7 if code==16 else self.operation)
        super().mark(bad,code)

    def add(self,a,b):
        self.operation=5;return super().add(a,b)

    def mul(self,a,b,bits=18,sign='unknown'):
        self.operation=4;return super().mul(a,b,bits,sign)


@dataclass
class Trajectory:
    y:np.ndarray
    valid:np.ndarray
    design:np.ndarray
    p13:np.ndarray
    errors:np.ndarray
    first_error:np.ndarray
    first_k:np.ndarray


def replay(operator,x,a,b,*,event_ids,scope,maximum_k=None):
    if operator not in MAX_K:raise ValueError('invalid operator')
    kmax=MAX_K[operator] if maximum_k is None else maximum_k
    if type(kmax) is not int or not 0<=kmax<=MAX_K[operator]:raise ValueError('invalid K')
    x=np.asarray(x);ids=np.asarray(event_ids)
    if x.dtype!=np.int64 or x.ndim!=1 or ids.dtype!=np.uint64 or ids.shape!=x.shape:
        raise ValueError('aligned encoded inputs and event IDs required')
    if len(np.unique(ids))!=len(ids):raise ValueError('duplicate event IDs')
    if any(type(v) is not int or not -2**63<=v<2**63 for v in (a,b)):
        raise ValueError('signed quantized coefficient words required')
    with np.errstate(over='ignore'):
        ctx=ObservedArithmetic(event_seeds(scope,ids));ctx.record(x<=0,1)
        operand=ctx.input(x);av=ctx.input(np.int64(a));bv=ctx.input(np.int64(b))
        y=ctx.add(ctx.mul(av,operand),bv);ctx.record(y.value<=0,2)
        initial=x.astype(object)*y.value.astype(object)
        if operator=='rsqrt':initial*=y.value.astype(object)
        upper=2*S*S if operator=='reciprocal' else 3*S**3
        inside=np.asarray((initial>0)&(initial<upper),bool);ctx.record(~inside,3)
        active=(x>0)&(y.value>0)&inside&(ctx.flags==0)
        ys=[];valid=[];design=[];p13=[];errors=[]
        for k in range(kmax+1):
            ctx.k=k;ctx.record(y.value<=0,2 if k==0 else 7)
            active&=(y.value>0)&(ctx.flags==0)
            ys.append(y.value.copy());valid.append(active.copy())
            design.append(active&precision(operator,x,y.value,12,False))
            p13.append(active&precision(operator,x,y.value,13,False))
            errors.append(np.where(active,0,ctx.first_error).astype(np.uint8))
            if k==kmax:break
            ctx.k=k+1
            if operator=='reciprocal':y=ctx.mul(y,ctx.add(2*S,-ctx.mul(operand,y)))
            else:
                yy=ctx.mul(y,y);xyy=ctx.mul(operand,yy)
                y=ctx.trunc(ctx.mul(y,ctx.add(3*S,-xyy)),1)
    return Trajectory(np.asarray(ys),np.asarray(valid),np.asarray(design),np.asarray(p13),
                      np.asarray(errors),ctx.first_error,ctx.first_k)
