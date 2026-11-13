MARKER='SPU_INPUT_REFERENCE_V1'
def reference(uuid,origin):
    return (MARKER,str(uuid),str(origin))
def is_reference(value):
    return isinstance(value,tuple) and len(value)==3 and value[0]==MARKER
def resolve(arguments,local_node,local_get,remote_get_many):
    groups={};resolved={}
    for value in arguments:
        if not is_reference(value):continue
        _,uuid,origin=value;key=(uuid,origin)
        if key in resolved:continue
        if origin==local_node:resolved[key]=local_get(uuid)
        else:
            items=groups.setdefault(origin,[])
            if uuid not in items:items.append(uuid)
    for origin,ids in groups.items():
        values=remote_get_many(origin,ids)
        if len(values)!=len(ids):raise ValueError('remote infeed object count mismatch')
        for uuid,value in zip(ids,values):resolved[(uuid,origin)]=value
    result=[resolved[(v[1],v[2])] if is_reference(v) else v for v in arguments]
    return result,dict(remote_batches=len(groups),remote_unique_refs=sum(map(len,groups.values())),
                       remote_reference_occurrences=sum(is_reference(v) and v[2]!=local_node for v in arguments))
