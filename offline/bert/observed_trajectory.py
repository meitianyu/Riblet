from dataclasses import dataclass
import numpy as np
import plaintext_numeric as p
from experiment_contract import design_pass
from cdi_bert.batch import precision_pass_batch

ERRORS={0:'none',1:'nonpositive_operand',2:'nonpositive_initial_output',
        3:'initial_convergence_state',4:'raw_product_wrap',5:'addition_wrap',
        6:'signed_truncation_range',7:'nonpositive_newton_output',8:'other_protocol_exception'}


class ObservedArithmetic(p.Arithmetic):
    def __init__(self,seeds):
        super().__init__(seeds,per_event=True)
        self.first_error=np.zeros(np.shape(seeds),np.uint8)
        self.first_k=np.full(np.shape(seeds),-1,np.int8)
        self.k=0;self.operation=8

    def record(self,bad,code):
        first=np.asarray(bad)&(self.first_error==0)
        self.first_error[first]=code;self.first_k[first]=self.k

    def mark(self,bad,code):
        self.record(bad,6 if code==2 else 8 if code==8 else 7 if code==16 else self.operation)
        super().mark(bad,code)

    def add(self,a,b):
        self.operation=5
        return super().add(a,b)

    def mul(self,a,b,bits=18,sign='unknown'):
        self.operation=4
        return super().mul(a,b,bits,sign)


@dataclass
class ObservedTrajectory:
    y_encoded: np.ndarray
    valid: np.ndarray
    design_pass: np.ndarray
    p13_pass: np.ndarray
    error_code: np.ndarray
    first_error_code: np.ndarray
    first_failure_K: np.ndarray

    @property
    def first_error_kind(self):
        return np.asarray([ERRORS[int(k)] for k in self.first_error_code],dtype=object)


def replay(operator,x,a,b,seeds,maximum_k=8):
    if operator not in ('reciprocal','rsqrt') or not 0<=maximum_k<=8:
        raise ValueError('invalid operator or K')
    ctx=ObservedArithmetic(seeds);ctx.record(x<=0,1)
    operand=ctx.input(x);av=ctx.input(np.int64(a));bv=ctx.input(np.int64(b))
    y=ctx.add(ctx.mul(av,operand),bv);ctx.record(y.value<=0,2)
    initial=x.astype(object)*y.value.astype(object)
    if operator=='rsqrt':initial*=y.value.astype(object)
    upper=2*p.S*p.S if operator=='reciprocal' else 3*p.S**3
    inside=np.asarray((initial>0)&(initial<upper),bool);ctx.record(~inside,3)
    active=(x>0)&(y.value>0)&inside&(ctx.flags==0)
    ys=[];valid=[];design=[];p13=[];errors=[]
    for k in range(maximum_k+1):
        ctx.k=k;ctx.record(y.value<=0,2 if k==0 else 7)
        active &= (y.value>0)&(ctx.flags==0)
        ys.append(y.value.copy());valid.append(active.copy())
        design.append(active&design_pass(operator,x,y.value))
        p13.append(active&precision_pass_batch(x,y.value,operator=operator,bits=13,inclusive=False))
        errors.append(np.where(active,0,ctx.first_error).astype(np.uint8))
        if k==maximum_k:break
        ctx.k=k+1
        if operator=='reciprocal':y=ctx.mul(y,ctx.add(2*p.S,-ctx.mul(operand,y)))
        else:
            yy=ctx.mul(y,y);xyy=ctx.mul(operand,yy)
            y=ctx.trunc(ctx.mul(y,ctx.add(3*p.S,-xyy)),1)
    return ObservedTrajectory(np.asarray(ys),np.asarray(valid),np.asarray(design),np.asarray(p13),
        np.asarray(errors),ctx.first_error,ctx.first_k)
