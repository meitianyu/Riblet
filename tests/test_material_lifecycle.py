import importlib
from pathlib import Path
import tempfile
import unittest


class MaterialLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.module = importlib.import_module('riblet.lifecycle')

    def test_reserved_or_consumed_ids_cannot_be_reused(self):
        with tempfile.TemporaryDirectory() as temp:
            store=self.module.RequestLedger(Path(temp))
            binding={'graph':'g','parameters':'p','budget':'b','model':'m','roles':[0,1]}
            ticket=store.reserve('session1-query1',binding)
            with self.assertRaises(FileExistsError):store.reserve('session1-query1',binding)
            with self.assertRaises(ValueError):ticket.begin_online()
            ticket.prepared({'rank':0,'complete':True,'binding':binding},
                            {'rank':1,'complete':True,'binding':binding})
            ticket.begin_online()
            ticket.retire()
            self.assertEqual(store.read('session1-query1')['state'],'retired')
            with self.assertRaises(FileExistsError):store.reserve('session1-query1',binding)
            with self.assertRaises(ValueError):ticket.begin_online()

    def test_peer_mismatch_and_failure_never_recycle_materials(self):
        with tempfile.TemporaryDirectory() as temp:
            store=self.module.RequestLedger(Path(temp))
            binding={'graph':'g','parameters':'p','budget':'b','model':'m','roles':[0,1]}
            ticket=store.reserve('session1-query2',binding)
            with self.assertRaises(ValueError):
                ticket.prepared({'rank':0,'complete':True,'binding':binding},
                                {'rank':1,'complete':True,'binding':{**binding,'budget':'old'}})
            ticket.abandon('preparation_failed')
            self.assertEqual(store.read('session1-query2')['state'],'abandoned')
            with self.assertRaises(FileExistsError):store.reserve('session1-query2',binding)
            with self.assertRaises(ValueError):store.reserve('../escape',binding)


if __name__=='__main__':unittest.main()
