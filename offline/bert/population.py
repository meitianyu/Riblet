from collections import Counter

TASK_COUNTS={'cola':5873,'mnli':9398,'mrpc':1779,'qnli':9398,'qqp':9398,'rte':1164,'sst2':9398,'stsb':3592}


def validate_rows(rows,*,task_counts=TASK_COUNTS,token_count=None):
    counts=Counter();folds={t:[0]*5 for t in task_counts};seen=set();tokens=0
    expected_tasks=[t for t,n in task_counts.items() for _ in range(n)]
    if len(rows)!=len(expected_tasks):raise ValueError('wrong population size')
    for i,(r,task) in enumerate(zip(rows,expected_tasks)):
        ordinal = r['request_ordinal']
        if (type(ordinal) is not int or not 0 <= ordinal < 2**47
                or (i and ordinal <= rows[i-1]['request_ordinal']) or r['task'] != task):
            raise ValueError('original ordinal/task order changed')
        if r['request_digest'] in seen:raise ValueError('duplicate request identity')
        seen.add(r['request_digest'])
        n=r['effective_length'];f=r['fold_id']
        if type(n) is not int or not 1<=n<=128 or type(f) is not int or f not in range(5):
            raise ValueError('invalid length or original fold')
        if r['source_role']!='Development':raise ValueError('non-Development request')
        if any(len(r[k])!=n for k in ('input_ids','token_type_ids')):
            raise ValueError('padding or token arrays disagree')
        if any(type(v) is not int or v<0 for v in r['input_ids']):raise ValueError('invalid token ID')
        if any(type(v) is not int or v not in (0,1) for v in r['token_type_ids']):raise ValueError('invalid segment ID')
        counts[task]+=1;folds[task][f]+=1;tokens+=n
    if dict(counts)!=task_counts or (token_count is not None and tokens!=token_count):raise ValueError('population totals changed')
    return dict(requests=len(rows),effective_tokens=tokens,reciprocal_events=144*tokens,
                rsqrt_events=25*tokens,fold_counts=folds)
