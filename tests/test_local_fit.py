import importlib
import json
from pathlib import Path
import tempfile
import unittest


class LocalFitTests(unittest.TestCase):
    def setUp(self):
        self.module = importlib.import_module('riblet.offline')

    def test_local_fit_is_reused_only_for_the_same_encoded_operands(self):
        import hashlib
        import numpy as np
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            x = root / 'operands.npz'
            np.savez(x, operands=np.full((1,2),262144,np.int64),
                     request_ids=np.array([0,1],np.int64),folds=np.array([0,1],np.uint8),
                     lengths=np.array([1,1],np.int32))
            meta = {
                'algorithm_version':'public-site-f18-p12',
                'design_bits':12,'design_inclusive':False,
                'numeric_reference':'G9_PLAINTEXT_NATIVE_MIXED_SEG3_EXP6_V2',
                'numeric_seed_lineage_id':'JOINT_ROLE6_F18_COMP_G9_V2',
                'population_sha256':'9ecbbd63f91b6a74136d2afc02825dbad6b4668ff0e4a23a41eb4b9cde667374',
                'graph_id':'BERT_PUBLIC_SITE_pad128_NATIVE_INVERSE18',
                'physical_length':128,
                'lineage':'public-site-pad128',
                'point':0,'task':'cola','model_hash':'fixture-not-deployable',
                'upstream_profiles':{},'sha256':hashlib.sha256(x.read_bytes()).hexdigest(),
            }
            (root/'input.json').write_text(json.dumps(meta))
            result = self.module.fit_bert_point(x, root/'fit', point=0, task='cola')
            site=json.loads((root/'fit/tables.json').read_bytes())['sites'][0]
            self.assertIsNone(site['K']);self.assertFalse(site['enabled'])
            self.assertEqual(site['events'],2)
            self.assertEqual(result['status'],'point_fit_complete')
            before=(root/'fit/result.json').stat().st_mtime_ns
            resumed=self.module.fit_bert_point(x,root/'fit',point=0,task='cola')
            self.assertEqual(result,resumed)
            self.assertEqual(before,(root/'fit/result.json').stat().st_mtime_ns)
            x.write_bytes(x.read_bytes()+b'changed')
            with self.assertRaises(ValueError):
                self.module.fit_bert_point(x,root/'fit',point=0,task='cola')


if __name__ == '__main__':
    unittest.main()
