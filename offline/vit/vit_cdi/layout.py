SLOTS=5578


def _int(value,lo,hi):
    if isinstance(value,bool) or not isinstance(value,int) or not lo<=value<=hi:
        raise ValueError('public layout coordinate out of range')
    return value


def conditions(operator):
    if operator=='reciprocal':return ('CLS',)+tuple(f'GRID{i}' for i in range(36))
    if operator=='rsqrt':return ('CLS',)+tuple(f'REGION{i}' for i in range(9))
    raise ValueError('unknown operator')


def condition(operator,position):
    _int(position,0,196)
    conditions(operator)
    if position==0:return 'CLS'
    i,j=divmod(position-1,14)
    if operator=='reciprocal':return f'GRID{6*(6*i//14)+6*j//14}'
    return f'REGION{3*(3*i//14)+3*j//14}'


def points():
    out=[]
    for block in range(12):
        out.extend([dict(point=3*block,operator='rsqrt',site=2*block,kind='pre_attention_ln'),
                    dict(point=3*block+1,operator='reciprocal',site=block,kind='softmax'),
                    dict(point=3*block+2,operator='rsqrt',site=2*block+1,kind='pre_mlp_ln')])
    out.append(dict(point=36,operator='rsqrt',site=24,kind='final_encoder_ln'))
    return tuple(out)


def public_index(operator,site,condition_id):
    choices=conditions(operator)
    if condition_id not in choices:raise ValueError('not a deployable public condition')
    if operator=='reciprocal':return 37*_int(site,0,143)+choices.index(condition_id)
    return 5328+10*_int(site,0,24)+choices.index(condition_id)


def event_id(request_ordinal,point,head,position):

    _int(request_ordinal,0,9999);_int(point,0,36);_int(head,0,11);_int(position,0,196)
    return ((request_ordinal*37+point)*12+head)*197+position
