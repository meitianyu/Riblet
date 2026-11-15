import importlib,unittest
import numpy as np

class BertRuntimeDtypeTests(unittest.TestCase):
    def test_runtime_boundary_restores_original_i64_without_changing_values(self):
        m=importlib.import_module('riblet.inputs')
        self.assertTrue(hasattr(m,'bert_runtime_inputs'),'original BERT runtime input ABI missing')
        ids=np.zeros((1,128),np.int32);ids[0,:3]=[101,2009,102]
        mask=np.zeros_like(ids);mask[0,:3]=1
        values=dict(input_ids=ids,attention_mask=mask,token_type_ids=np.zeros_like(ids))
        result=m.bert_runtime_inputs(values)
        for key in ('input_ids','attention_mask','token_type_ids'):
            self.assertEqual(result[key].dtype,np.dtype('int64'))
            np.testing.assert_array_equal(result[key],values[key])
        self.assertEqual(ids.dtype,np.dtype('int32'))
        self.assertEqual(set(result), set(values))
    def test_old_undeclared_or_i32_bert_graph_is_rejected(self):
        from riblet.party import validate_graph_identity
        g=dict(model='bert-cola',family='bert',layout='pad128',method='riblet-full',model_rank=0,data_rank=1)
        with self.assertRaises(ValueError):validate_graph_identity(g)
        with self.assertRaises(ValueError):validate_graph_identity({**g,'input_dtype':'int32'})
        validate_graph_identity({**g,'input_dtype':'int64'})

if __name__=='__main__':unittest.main()
