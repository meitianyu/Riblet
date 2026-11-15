import importlib
import json
from pathlib import Path
import tempfile
import unittest


class ModelLoadingTests(unittest.TestCase):
    def test_declared_head_and_architecture_are_checked_before_loading(self):
        module=importlib.import_module('riblet.offline')
        self.assertTrue(hasattr(module,'validate_bert_checkpoint'),'checkpoint admission is missing')
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            cfg=dict(model_type='bert',hidden_act='gelu',hidden_size=768,
                     num_attention_heads=12,num_hidden_layers=12,intermediate_size=3072,
                     vocab_size=30522,type_vocab_size=2,layer_norm_eps=1e-12,
                     max_position_embeddings=512,id2label={'0':'LABEL_0','1':'LABEL_1'})
            (root/'config.json').write_text(json.dumps(cfg))
            self.assertEqual(module.validate_bert_checkpoint(root,'cola')['num_labels'],2)
            with self.assertRaises(ValueError):module.validate_bert_checkpoint(root,'stsb')
            del cfg['id2label']
            (root/'config.json').write_text(json.dumps(cfg))
            checked=module.validate_bert_checkpoint(root,'cola')
            self.assertEqual(checked['num_labels'],2)
            self.assertFalse(checked['head_metadata_present'])
            cfg['hidden_size']=1024
            (root/'config.json').write_text(json.dumps(cfg))
            with self.assertRaises(ValueError):module.validate_bert_checkpoint(root,'cola')


if __name__=='__main__':unittest.main()
