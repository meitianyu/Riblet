import importlib.util
from pathlib import Path
import tempfile
import unittest


class RuntimeInstallTests(unittest.TestCase):
    def test_incomplete_build_is_rejected_before_destination_creation(self):
        script=Path(__file__).parents[1]/'tools/install_runtime.py'
        self.assertTrue(script.is_file(),'source-built runtime installer is missing')
        spec=importlib.util.spec_from_file_location('runtime_installer',script)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'source';source.mkdir();build=root/'build';build.mkdir()
            with self.assertRaises(ValueError):module.install(source,build,root/'runtime')
            self.assertFalse((root/'runtime').exists())


if __name__=='__main__':unittest.main()
