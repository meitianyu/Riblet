import numpy as np
from plaintext_numeric import REFERENCE_MODEL
from experiment_contract import event_seeds
from observed_trajectory import replay


class PlaintextBackend:
    abi_id = REFERENCE_MODEL['id']
    evidence_class = 'plaintext_numeric_reference_not_native_joint_distribution'
    deterministic_by_operand = False

    def __init__(self, namespace):
        if not isinstance(namespace,str) or not namespace: raise ValueError('namespace required')
        self.namespace = namespace

    def require_purpose(self, purpose):
        if purpose not in ('plaintext_fit', 'synthetic_unit'):
            raise ValueError('explicit plaintext_fit purpose required; this is not native MPC evidence')

    def score(self, operator, operands, *, a_encoded, b_encoded, event_ids, scope, maximum_k=8):
        x = np.asarray(operands); ids = np.asarray(event_ids)
        if x.dtype != np.int64 or x.ndim != 1 or ids.dtype != np.uint64 or ids.shape != x.shape:
            raise ValueError('aligned int64 operands and uint64 event IDs required')
        if len(np.unique(ids)) != len(ids): raise ValueError('duplicate event IDs')
        seeds = event_seeds(self.namespace,scope,ids)
        with np.errstate(over='ignore'):
            return replay(operator,x,a_encoded,b_encoded,seeds,maximum_k)
