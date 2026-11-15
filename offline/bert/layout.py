LAYOUT = 'pad128'
PHYSICAL_METADATA = 128
GRAPH_ID = 'BERT_PUBLIC_SITE_pad128_NATIVE_INVERSE18'
JOINT_ID = 'public-site-pad128'
NUMERIC_REFERENCE = 'G9_PLAINTEXT_NATIVE_MIXED_SEG3_EXP6_V2'

def physical_length(n):
    if not 1 <= int(n) <= 128:
        raise ValueError('effective length outside contract')
    return PHYSICAL_METADATA
