import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class CPUReferenceTests(unittest.TestCase):
    def test_numpy_and_cpu_jax_have_identical_encoded_intermediates(self):
        import numpy as np
        root = Path(__file__).resolve().parents[1]
        for model in ('bert', 'vit'):
            source = root / 'offline' / model
            self.assertTrue(source.is_dir(), 'CPU construction source is missing')
            with tempfile.TemporaryDirectory() as temp:
                outputs = []
                for backend in ('numpy', 'jax'):
                    target = Path(temp) / (backend + '.npz')
                    env = {**os.environ, 'PYTHONPATH':str(source),
                           'CDI_ARRAY_BACKEND':backend, 'JAX_PLATFORMS':'cpu',
                           'JAX_ENABLE_X64':'true', 'BERT_CDI_LAYOUT':'pad128',
                           'OMP_NUM_THREADS':'1', 'OPENBLAS_NUM_THREADS':'1'}
                    subprocess.run([sys.executable, '-B', str(root / 'tests/numeric_worker.py'),
                                    model, str(target)], env=env, check=True)
                    with np.load(target, allow_pickle=False) as data:
                        outputs.append({key:data[key].copy() for key in data.files})
                self.assertEqual(set(outputs[0]), set(outputs[1]))
                for key in outputs[0]:
                    np.testing.assert_array_equal(outputs[0][key], outputs[1][key],
                                                  err_msg=model + ':' + key)


if __name__ == '__main__':
    unittest.main()
