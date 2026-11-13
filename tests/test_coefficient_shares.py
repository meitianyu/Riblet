import importlib
import unittest


class CoefficientShareTests(unittest.TestCase):
    def setUp(self):
        self.module = importlib.import_module('riblet.coefficient_shares')

    def test_fresh_model0_shares_reconstruct_encoded_words(self):
        import numpy as np
        table=np.array([[0,2**64-1],[2**63,131072]],dtype=np.uint64)
        a,b=self.module.split(table,owner_rank=0)
        np.testing.assert_array_equal(a+b,table)
        c,d=self.module.split(table,owner_rank=0)
        self.assertFalse(np.array_equal(a,c))
        self.assertFalse(np.array_equal(b,d))
        self.assertEqual(a.dtype,np.uint64)
        with self.assertRaises(ValueError):self.module.split(table,owner_rank=1)

    def test_float_or_non_record_table_is_not_silently_reencoded(self):
        import numpy as np
        with self.assertRaises(ValueError):self.module.split(np.ones((2,2)),owner_rank=0)
        with self.assertRaises(ValueError):self.module.split(np.ones((2,3),np.uint64),owner_rank=0)


if __name__=='__main__':unittest.main()
