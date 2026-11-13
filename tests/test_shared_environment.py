import hashlib,importlib,json,sys,tempfile,unittest
from pathlib import Path

class SharedEnvironmentTests(unittest.TestCase):
    def test_resolve_manifest_relative_backend_and_reject_tampered_file(self):

        module = importlib.import_module('riblet.environment')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);backend=root/'lib/bert';backend.mkdir(parents=True)
            (backend/'probe').write_bytes(b'bound')
            (backend/'RUNTIME_FILES.json').write_text(json.dumps({'probe':hashlib.sha256(b'bound').hexdigest()}))
            manifest=root/'installation.json'
            manifest.write_text(json.dumps({'python_prefix':sys.prefix,'backends':{'bert':'lib/bert'}}))
            self.assertEqual(module.resolve({'installation':str(manifest)},'bert'),backend.resolve())
            (backend/'probe').write_bytes(b'changed')
            with self.assertRaises(ValueError):module.resolve({'installation':str(manifest)},'bert')
    def test_other_python_environment_is_rejected(self):
        module = importlib.import_module('riblet.environment')
        with tempfile.TemporaryDirectory() as directory:
            manifest=Path(directory)/'installation.json'
            manifest.write_text(json.dumps({'python_prefix':str(Path(directory)/'other'),'backends':{}}))
            with self.assertRaises(ValueError):module.resolve({'installation':str(manifest)},'vit')

if __name__=='__main__':unittest.main()
