import importlib.util,sys
from pathlib import Path
import tempfile,unittest


class ConstructionAuditTests(unittest.TestCase):
    def test_partial_vit_construction_cannot_generate_deployment_files(self):
        folder=Path(__file__).parents[1]/'offline/vit';source=folder/'audit.py'
        self.assertTrue(source.is_file(),'local complete construction audit is missing')
        sys.path.insert(0,str(folder))
        spec=importlib.util.spec_from_file_location('vit_construction_audit',source)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):module.audit(temp)
            self.assertFalse((Path(temp)/'constructed-joint').exists())


if __name__=='__main__':unittest.main()
