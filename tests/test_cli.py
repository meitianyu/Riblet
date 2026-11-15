import importlib,json,tempfile,unittest
from pathlib import Path


class CliTests(unittest.TestCase):
    def test_role_is_a_required_command_argument(self):
        import io
        from contextlib import redirect_stderr
        from riblet.cli import main
        error=None
        with redirect_stderr(io.StringIO()):
            try:main(['run','--config','not-read.json'])
            except (SystemExit,PermissionError) as caught:error=caught
        self.assertIsInstance(error,SystemExit)
        self.assertEqual(error.code,2)
    def test_prepare_model_accepts_construction_without_peer_configuration(self):
        from riblet.config import load_run_config
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'model.json'
            doc=dict(host_role='model',profile='paper',model=dict(name='bert-cola',checkpoint_dir='model',preprocessor_dir='processor'),
                     parameters=dict(source='construct',package_dir='parameters'),
                     construction=dict(selection={'mode':'paper'},workers=16,output_dir='construction-output',split='train'))
            path.write_text(json.dumps(doc))
            try:config=load_run_config(path,'model','prepare-model')
            except ValueError as e:self.fail(str(e))
            self.assertEqual(config['construction_root'],Path(folder)/'construction-output')
            self.assertEqual(config['construction']['workers'],16)
            self.assertNotIn('parties',config)
            self.assertNotIn('runtime',config)
            doc['host_role']='data';path.write_text(json.dumps(doc))
            with self.assertRaises(PermissionError):load_run_config(path,'data','prepare-model')

    def test_data_role_cannot_compile_or_load_model_assets(self):
        m=importlib.import_module('riblet.cli')
        with tempfile.TemporaryDirectory() as temp:
            doc=json.loads((Path(__file__).resolve().parents[1]/'riblet.example.json').read_text())
            doc['host_role']='data'
            cfg=Path(temp)/'data.json';cfg.write_text(json.dumps(doc))
            for command in ('compile','prepare-model'):
                with self.assertRaises(PermissionError):
                    m.main([command,'--role','data','--config',str(cfg)])


if __name__=='__main__':unittest.main()
