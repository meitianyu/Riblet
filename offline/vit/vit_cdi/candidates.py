from decimal import Decimal, localcontext
from functools import lru_cache

S=262144


@lru_cache(maxsize=8192)
def _robust_center(low,high,j):
    with localcontext() as context:
        context.prec=96
        L=Decimal(low)/S;H=Decimal(high)/S;u=Decimal(j)/32
        return L.sqrt() if j==0 else H.sqrt() if j==32 else (((1-u)*L.ln()+u*H.ln())/2).exp()


@lru_cache(maxsize=33)
def _robust_scale(m):
    with localcontext() as context:
        context.prec=96
        return (Decimal(m)/32).exp()


@lru_cache(maxsize=2)
def families(operator):
    if operator=='reciprocal':return (('analytic',),)
    if operator=='rsqrt':
        return (('analytic',),)+tuple(('robust',j,m) for j in range(33) for m in range(-16,17))
    raise ValueError('unknown operator')


def interval(operator,raw_low,raw_high):
    if not 0<raw_low<=raw_high<2**63:raise ValueError('invalid positive encoded interval')
    families(operator)
    return (raw_low,(17*raw_high+9)//10) if operator=='reciprocal' else (raw_low,raw_high)


def coefficients(operator,low,high,family):
    if not 0<low<=high<2**63:raise ValueError('invalid positive encoded interval')
    if family not in families(operator):raise ValueError('undeclared candidate family')
    if operator=='reciprocal':
        q=low*low+6*low*high+high*high
        a=-(8*S**3//q);b=8*(low+high)*S**2//q
    else:
        with localcontext() as context:
            context.prec=96
            L=Decimal(low)/S;H=Decimal(high)/S
            if family==('analytic',):
                u=L.sqrt();v=H.sqrt();s=u*u+u*v+v*v
                c=2/(u*v*(u+v)+Decimal(2)/3*s*(s/3).sqrt())
                a=int(-c*S);b=int(c*s*S)
            else:
                _,j,m=family;u=Decimal(j)/32


                t=_robust_center(low,high,j)
                z=_robust_scale(m)
                a=int(-z*S/(2*t**3));b=int(3*z*S/(2*t))
    if not -2**63<=a<2**63 or not -2**63<=b<2**63:raise OverflowError('coefficient outside FM64')
    return a,b
