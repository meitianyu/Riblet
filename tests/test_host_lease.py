import importlib,subprocess,sys,tempfile,unittest
from pathlib import Path

class HostLeaseTests(unittest.TestCase):
    def test_other_process_cannot_enter_until_owner_releases(self):
        m=importlib.import_module('riblet.lifecycle')
        self.assertTrue(hasattr(m,'host_lease'),'host-wide model exclusion missing')
        with tempfile.TemporaryDirectory() as d:
            lock=Path(d)/'host.lock'
            child='from riblet.lifecycle import host_lease; import sys\nwith host_lease(sys.argv[1]): print("entered")'
            with m.host_lease(lock):
                r=subprocess.run([sys.executable,'-c',child,str(lock)],capture_output=True,text=True)
                self.assertNotEqual(r.returncode,0);self.assertNotIn('entered',r.stdout)
            r=subprocess.run([sys.executable,'-c',child,str(lock)],capture_output=True,text=True)
            self.assertEqual(r.returncode,0,r.stderr);self.assertEqual(r.stdout.strip(),'entered')
    def test_exception_releases_host_lock(self):
        m=importlib.import_module('riblet.lifecycle')
        self.assertTrue(hasattr(m,'host_lease'),'host-wide model exclusion missing')
        with tempfile.TemporaryDirectory() as d:
            lock=Path(d)/'host.lock'
            with self.assertRaises(RuntimeError):
                with m.host_lease(lock):raise RuntimeError('owned failure')
            with m.host_lease(lock):pass

if __name__=='__main__':unittest.main()
