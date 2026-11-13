import importlib.util,tempfile,unittest
from pathlib import Path

class InstallTests(unittest.TestCase):
    def test_installation_receipt_describes_installation_only(self):
        import hashlib,json
        from unittest.mock import patch
        script=Path(__file__).parents[1]/'tools/install_environment.py'
        spec=importlib.util.spec_from_file_location('installer',script)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); runtime=root/'runtime';(runtime/'spu').mkdir(parents=True)
            (runtime/'spu/libspu.so').write_bytes(b'runtime fixture')
            (runtime/'RUNTIME_FILES.json').write_text(json.dumps({'spu/libspu.so':hashlib.sha256(b'runtime fixture').hexdigest()}))
            site=root/'site';site.mkdir()
            with patch.object(module.sys,'version_info',(3,10)),patch.object(module.venv.EnvBuilder,'create',lambda self,p:Path(p).mkdir()),patch.object(module.subprocess,'run'),patch.object(module.subprocess,'check_output',return_value=str(site)):
                manifest=module.install(root/'env',runtime,runtime)
            self.assertEqual(manifest['status'],'installed')
            self.assertEqual(json.loads((root/'env/installation.json').read_text()),manifest)
            self.assertEqual((root/'env/riblet-backends/bert/spu/libspu.so').read_bytes(),b'runtime fixture')

    def test_source_install_requires_both_complete_source_roots(self):
        script=Path(__file__).parents[1]/'tools/install_environment.py'
        spec=importlib.util.spec_from_file_location('installer',script)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        self.assertTrue(hasattr(module,'source_roots'),'shared source-build workflow missing')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name in ('bert','vit'):
                (root/name).mkdir();(root/name/'WORKSPACE').write_text('workspace()')
            self.assertEqual(module.source_roots(root),{'bert':root/'bert','vit':root/'vit'})
            (root/'vit/WORKSPACE').unlink()
            with self.assertRaises(ValueError):module.source_roots(root)
    def test_refuses_existing_prefix_before_installing_anything(self):
        script=Path(__file__).parents[1]/'tools/install_environment.py'
        self.assertTrue(script.is_file(),'shared installer missing')
        spec=importlib.util.spec_from_file_location('installer',script)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'keep').write_text('original')
            with self.assertRaises(FileExistsError):module.install(root,root,root)
            self.assertEqual((root/'keep').read_text(),'original')

if __name__=='__main__':unittest.main()
