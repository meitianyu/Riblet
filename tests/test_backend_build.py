import importlib.util,sys,tempfile,unittest
from pathlib import Path

class BackendBuildTests(unittest.TestCase):
    def test_plan_uses_current_interpreter_and_all_required_targets(self):
        script=Path(__file__).parents[1]/'tools/build_backend.py'
        self.assertTrue(script.is_file(),'reproducible backend build entry missing')
        sys.path.insert(0,str(script.parent))
        s=importlib.util.spec_from_file_location('builder',script);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            command=m.command('bazel',root/'output',root/'cache',8)
            self.assertIn('--repo_env=PYTHON_BIN_PATH='+sys.executable,command)
            for target in ('//spu:libspu.so','//libspu:spu_py_proto','//spu/utils:distributed_py_proto_grpc'):
                self.assertIn(target,command)
            self.assertIn('--jobs=8',command)
            with self.assertRaises(ValueError):m.command('bazel',root/'out',root/'cache',0)

if __name__=='__main__':unittest.main()
