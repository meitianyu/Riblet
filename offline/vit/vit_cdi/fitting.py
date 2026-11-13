from dataclasses import dataclass
from functools import cached_property, cmp_to_key
import numpy as np
from . import candidates as c, numeric as n, layout


@dataclass(frozen=True)
class Events:
    operands:np.ndarray
    event_ids:np.ndarray
    request_ids:np.ndarray
    folds:np.ndarray

    def __post_init__(self):
        size=len(self.operands)
        if self.operands.dtype!=np.int64 or self.event_ids.dtype!=np.uint64:
            raise ValueError('int64 operands and uint64 event IDs required')
        if any(v.ndim!=1 or len(v)!=size for v in (self.operands,self.event_ids,self.request_ids,self.folds)):
            raise ValueError('unaligned event arrays')
        if len(np.unique(self.event_ids))!=size or np.any((self.folds<0)|(self.folds>4)):
            raise ValueError('event identity/fold conflict')
        if len(np.unique(np.stack([self.request_ids,self.folds],axis=1),axis=0))!=len(np.unique(self.request_ids)):
            raise ValueError('one request belongs to multiple folds')

    @cached_property
    def prepared(self):

        support=len(np.unique(self.request_ids));folds=[]
        reason='nonpositive_observed_operand' if np.any(self.operands<=0) else None
        if support<919:reason=reason or 'insufficient_complete_request_support'
        for f in range(5):
            held=np.flatnonzero(self.folds==f);train=self.folds!=f
            if not len(held):reason=reason or f'held_out_fold_{f}_empty'
            if np.count_nonzero(train)<40 or len(np.unique(self.request_ids[train]))<100:
                reason=reason or f'fold_{f}_training_support'
            if np.any(train):folds.append((f,held,int(self.operands[train].min()),int(self.operands[train].max())))
        if len(self.operands):
            folds.append(('full',np.arange(len(self.operands)),int(self.operands.min()),int(self.operands.max())))
        return support,reason,tuple(folds)


def common_feasible(masks):
    arrays=[np.asarray(m) for m in masks]
    if len(arrays)!=6 or len({a.shape for a in arrays})!=1 or any(a.ndim!=1 or a.dtype!=np.bool_ for a in arrays):
        raise ValueError('five held-out plus full aligned boolean masks required')
    return [int(k) for k in np.flatnonzero(np.logical_and.reduce(arrays))]


def _scope(operator,site,source,condition,family):
    tag='analytic' if family==('analytic',) else f'robust_j{family[1]:02d}_m{family[2]+16:02d}'
    return f'VIT_JOINT_G9_V1:{operator}:site{site}:{source}:{condition}:{tag}'


def fit_candidate(events,*,operator,site,source,condition,family,chunk_size=65536):
    if type(chunk_size) is not int or chunk_size<1:raise ValueError('positive chunk size required')
    if operator not in n.MAX_K or family not in c.families(operator):raise ValueError('unknown candidate')
    if source not in ('CHILD','PATCH_PARENT'):raise ValueError('unknown source')
    if source=='PATCH_PARENT' and (operator!='rsqrt' or condition is not None):raise ValueError('patch-only RSqrt parent')
    if source=='CHILD' and (condition not in layout.conditions(operator) or (operator=='reciprocal' and condition=='CLS')):
        raise ValueError('invalid child target')
    scope=_scope(operator,site,source,condition,family)
    support,reason,folds=events.prepared
    row=dict(candidate_id=scope,source=source,condition=condition,family=list(family),
             operator=operator,site=site,support_request_count=support,support_event_count=len(events.operands),
             qualified=False,rejection_reason=reason,scores=[],feasible_Ks=[],L_encoded=None,H_encoded=None,
             a_encoded=None,b_encoded=None,design_bits=12,design_inclusive=False)
    if reason:return row
    masks=[];kmax=n.MAX_K[operator]
    for fold,indices,raw_l,raw_h in folds:
        L,H=c.interval(operator,raw_l,raw_h)
        try:a,b=c.coefficients(operator,L,H,family)
        except (OverflowError,ValueError):
            row['rejection_reason']='coefficient_encoding_or_domain_failure';return row
        valid=np.zeros(kmax+1,np.int64);passing=valid.copy();p13=valid.copy()
        max_res=np.zeros(kmax+1,np.float64);errors={};outside=0
        for start in range(0,len(indices),chunk_size):
            ii=indices[start:start+chunk_size];x=events.operands[ii]
            t=n.replay(operator,x,a,b,event_ids=events.event_ids[ii],scope=f'{scope}:fold={fold}',maximum_k=kmax)
            valid+=t.valid.sum(axis=1);passing+=t.design.sum(axis=1);p13+=t.p13.sum(axis=1)
            outside+=int(np.count_nonzero((x<L)|(x>H)))
            factor=x.astype(np.float64)/n.S
            if operator=='rsqrt':factor=np.sqrt(factor)
            residual=np.abs(1-factor[None,:]*(t.y.astype(np.float64)/n.S))
            max_res=np.maximum(max_res,residual.max(axis=1,initial=0))
            kinds,counts=np.unique(t.first_error[t.first_error!=0],return_counts=True)
            for kind,count in zip(kinds,counts):errors[n.ERRORS[int(kind)]]=errors.get(n.ERRORS[int(kind)],0)+int(count)
        feasible=(valid==len(indices))&(passing==len(indices))
        masks.append(feasible)
        row['scores'].append(dict(fold=fold,scored_events=len(indices),raw_L=raw_l,raw_H=raw_h,
            L_encoded=L,H_encoded=H,a_encoded=a,b_encoded=b,valid_counts=valid.tolist(),
            design_counts=passing.tolist(),p13_diagnostic_counts=p13.tolist(),feasible=feasible.tolist(),
            max_residual=[float(r) if v==len(indices) else None for r,v in zip(max_res,valid)],
            out_of_interval_events=outside,first_error_counts=errors))
        if fold=='full':row.update(L_encoded=L,H_encoded=H,a_encoded=a,b_encoded=b)
    row['feasible_Ks']=common_feasible(masks);row['qualified']=bool(row['feasible_Ks'])
    row['K_min']=min(row['feasible_Ks']) if row['qualified'] else None
    row['rejection_reason']=None if row['qualified'] else 'no_common_feasible_K'
    return row


def fit_all_families(events,**kwargs):

    for family in c.families(kwargs['operator']):yield fit_candidate(events,family=family,**kwargs)


def _compare(a,b):
    cross=int(a['H_encoded'])*int(b['L_encoded'])-int(b['H_encoded'])*int(a['L_encoded'])
    if cross:return -1 if cross<0 else 1
    ka=(a['source']!='CHILD',-a['support_request_count'],min(a['feasible_Ks']),a['candidate_id'])
    kb=(b['source']!='CHILD',-b['support_request_count'],min(b['feasible_Ks']),b['candidate_id'])
    return -1 if ka<kb else 1 if ka>kb else 0


def select_site(operator,counts,candidate_rows):

    conditions=layout.conditions(operator)
    if set(counts)!=set(conditions) or any(type(v) is not int or v<0 for v in counts.values()):
        raise ValueError('complete condition event counts required')
    candidates=list(candidate_rows)
    if len({r['candidate_id'] for r in candidates})!=len(candidates):raise ValueError('duplicate candidates')
    for r in candidates:
        if r['qualified']:
            ks=r['feasible_Ks']
            if not ks or ks!=sorted(set(ks)) or any(type(k) is not int or k<0 or k>n.MAX_K[operator] for k in ks):
                raise ValueError('actual feasible K set required')
    routes={};normal=fallback=child=cost=iteration_events=0;frontier={}
    for condition in conditions:
        choices=[];count=counts[condition]
        for k in range(n.MAX_K[operator]+1):
            feasible=[r for r in candidates if r['qualified'] and k in r['feasible_Ks'] and
                      ((r['source']=='CHILD' and r['condition']==condition) or
                       (r['source']=='PATCH_PARENT' and operator=='rsqrt' and condition!='CLS'))]
            if operator=='reciprocal' and condition=='CLS':feasible=[]

            if k>8:feasible=[]
            if feasible and count:
                best=min(feasible,key=cmp_to_key(_compare));nn=count;ff=0
                cc=count if best['source']=='CHILD' else 0
                route=dict(route='CDI_NORMAL',candidate_id=best['candidate_id'],source=best['source'],public_Ks=k,
                    a_encoded=best['a_encoded'],b_encoded=best['b_encoded'],L_encoded=best['L_encoded'],H_encoded=best['H_encoded'])
            else:
                nn=cc=0;ff=count
                route=dict(route='SPU_FALLBACK',candidate_id=None,source=None,public_Ks=0,
                                       a_encoded=None,b_encoded=None,L_encoded=None,H_encoded=None)
            objective=(nn*k+ff*8,-nn,k,-cc)
            choices.append(dict(objective=objective,public_Ks=k,route=route,normal_events=nn,fallback_events=ff,child_events=cc))
        chosen=min(choices,key=lambda r:r['objective']);routes[condition]=chosen['route']
        normal+=chosen['normal_events'];fallback+=chosen['fallback_events'];child+=chosen['child_events']
        cost+=chosen['objective'][0];iteration_events+=chosen['normal_events']*chosen['route']['public_Ks']
        frontier[condition]=[{key:r[key] for key in ('public_Ks','normal_events','fallback_events','objective')} for r in choices]
    return dict(K_scope='public_site_condition',public_Ks=None,routes=routes,normal_events=normal,
        fallback_events=fallback,cost_proxy=cost,iteration_events=iteration_events,child_events=child,frontier=frontier)
