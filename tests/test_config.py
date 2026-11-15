import importlib,unittest

class ConfigurationTests(unittest.TestCase):
    def test_default_network_label_describes_an_unshaped_network(self):
        import json,tempfile
        from pathlib import Path
        from riblet.config import load_run_config
        source=Path(__file__).resolve().parents[1]/'riblet.example.json'
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'riblet.json'
            path.write_text(source.read_text())
            self.assertEqual(load_run_config(path,'model','run')['network'],'unshaped')

    def setUp(self):
        self.c = importlib.import_module('riblet.config')

    def test_same_roles_for_both_models(self):
        for model in ['bert-cola','vit-base-patch16-224']:
            self.assertEqual(self.c.model_spec(model)['model_rank'],0)
            self.assertEqual(self.c.model_spec(model)['data_rank'],1)
            self.assertEqual(self.c.rank_for(model,'model'),0)
            self.assertEqual(self.c.rank_for(model,'data'),1)
        with self.assertRaises(ValueError):self.c.rank_for('vit-base-patch16-224','legacy-client0')

    def test_model_head_and_integration_are_not_guessed(self):
        self.assertEqual(self.c.model_spec('bert-stsb')['output_shape'],[1,1])
        self.assertEqual(self.c.model_spec('bert-mnli')['output_shape'],[1,3])
        self.assertEqual(self.c.model_spec('bert-cola')['family'],'bert')
        self.assertEqual(self.c.model_spec('vit-base-patch16-224')['family'],'vit')
        with self.assertRaises(ValueError):self.c.model_spec('arbitrary-model')

    def test_peer_config_rejects_old_rank_and_same_endpoint(self):
        good={'model':{'rank':0,'endpoint':'model.example:65001'},'data':{'rank':1,'endpoint':'data.example:65001'}}
        self.assertEqual(self.c.validate_parties(good),['model.example:65001','data.example:65001'])
        bad={'model':{'rank':1,'endpoint':'model.example:65001'},'data':{'rank':0,'endpoint':'data.example:65001'}}
        with self.assertRaises(ValueError):self.c.validate_parties(bad)
        bad={'model':good['model'],'data':{'rank':1,'endpoint':'model.example:65001'}}
        with self.assertRaises(ValueError):self.c.validate_parties(bad)

if __name__=='__main__':unittest.main()
