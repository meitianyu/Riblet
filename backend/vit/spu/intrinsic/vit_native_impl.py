from .mjv_context_impl import _mjv_context_call_prim

def vit_linear_manifest(anchor,*,plan_sha256,records):
    if len(plan_sha256)!=64 or any(c not in '0123456789abcdef' for c in plan_sha256):
        raise ValueError('invalid public plan digest')
    if any(c not in '0123456789,;' for c in records):raise ValueError('invalid fingerprint encoding')
    return _mjv_context_call_prim.bind(anchor,target=f'mjv_vit_linear_manifest:{plan_sha256}:{records}')[0]
