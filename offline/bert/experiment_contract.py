import hashlib
import json
from pathlib import Path
import numpy as np
from cdi_bert.batch import precision_pass_batch

ALGORITHM_VERSION='public-site-f18-p12'
DESIGN_BITS=12
DESIGN_INCLUSIVE=False
from layout import GRAPH_ID,JOINT_ID,PHYSICAL_METADATA,LAYOUT,NUMERIC_REFERENCE
LINEAGES={'joint':JOINT_ID}
SEED_LINEAGES={JOINT_ID:'JOINT_ROLE6_F18_COMP_G9_V2'}


def numeric_scope(scope):
    prefix,separator,suffix=scope.partition(':')
    return SEED_LINEAGES.get(prefix,prefix)+separator+suffix


def event_seeds(namespace,scope,event_ids):
    prefix=f'{namespace}\0{numeric_scope(scope)}\0'.encode()
    return np.asarray([int.from_bytes(hashlib.sha256(prefix+str(int(i)).encode()).digest()[:8],
                                     'little') for i in event_ids],np.uint64)


def design_pass(operator,x,y):
    return precision_pass_batch(x,y,operator=operator,bits=DESIGN_BITS,inclusive=DESIGN_INCLUSIVE)


def final_pass(operator,x,y):
    return precision_pass_batch(x,y,operator=operator,bits=12,inclusive=True)


def load_config(path=None):
    config=json.loads(Path(path or Path(__file__).with_name('config.json')).read_text())
    if config.get('family_id')!=ALGORITHM_VERSION:
        raise ValueError('wrong experiment family')
    if config.get('design_precision')!={op:{'bits':DESIGN_BITS,'inclusive':DESIGN_INCLUSIVE} for op in ('reciprocal','rsqrt')}:
        raise ValueError('both operators require the same strict design p12')
    if config.get('seed_lineage_aliases')!=SEED_LINEAGES:
        raise ValueError('numeric seed identity changed')
    if config.get('precision')!={'ring_bits':64,'fractional_bits':18,'scale':262144,'N_max':128}:
        raise ValueError('numerical format changed')
    if config.get('support')!={'folds':5,'complete_distinct_requests':919,
        'training_complement_distinct_requests':100,'training_complement_events':40,'held_out_fold_nonempty':True}:
        raise ValueError('support gate changed')
    fit=config['candidate_fit'];numeric=config['numeric_abi']
    if fit['K_candidates']!=list(range(9)) or fit['reciprocal_upper_cap_encoded'] is not None or fit['reciprocal_upper']!='ceil(17*H_raw_encoded/10)_once' or fit['rsqrt_endpoints']!='L_raw_and_H_raw_without_expansion':
        raise ValueError('candidate K/domain rules changed')
    if numeric['rsqrt_compensation']['sqrt2_word']!=370727 or numeric['rsqrt_compensation']['final_product_shift']!=27:
        raise ValueError('G9 compensation changed')
    if config['execution']['reference_model']!=NUMERIC_REFERENCE or config['execution']['cryptographic_execution']:
        raise ValueError('clear numerical execution required')
    if config.get('graph_id')!=GRAPH_ID or config['execution']['model_input_shape']!=('[1,128]' if LAYOUT=='pad128' else '[1,n]') or config.get('requested_modes')!=['joint']:
        raise ValueError('native actual-length Joint graph required')
    return config


def validate_input_binding(meta,lineage,population=None):
    if lineage not in SEED_LINEAGES:
        raise ValueError('unknown production lineage')
    if not population:raise ValueError('missing design identity')
    expected=dict(algorithm_version=ALGORITHM_VERSION,design_bits=DESIGN_BITS,
        design_inclusive=DESIGN_INCLUSIVE,numeric_reference=NUMERIC_REFERENCE,
        numeric_seed_lineage_id=SEED_LINEAGES[lineage],population_sha256=population)
    expected.update(graph_id=GRAPH_ID,physical_length=PHYSICAL_METADATA)
    if any(meta.get(k)!=v for k,v in expected.items()):
        raise ValueError('point family/gate/population/numerical identity mismatch')
