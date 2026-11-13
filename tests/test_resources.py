import importlib,unittest


class ResourceTests(unittest.TestCase):
    def test_vit_schedule_preserves_all_runtime_families(self):
        import sys
        from pathlib import Path
        from unittest.mock import patch
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'frontends/vit'))
        import vit_native_policy as policy
        from vit_full_secure_entry import select_runtime_families
        self.assertTrue(hasattr(policy,'load_schedule'),'single geometry loader missing')
        schedule=policy.load_schedule()
        self.assertEqual(set(schedule),{'logical_sites','physical_families','object_contexts'})
        self.assertEqual([len(schedule[k]) for k in ('logical_sites','physical_families','object_contexts')],[98,74,74])
        self.assertEqual(schedule['logical_sites'][0]['native_lhs_shape'],[1,224,224,3])
        static,dynamic=select_runtime_families(schedule)
        self.assertEqual((len(static),len(dynamic)),(49,24))
        with patch.object(Path,'read_bytes',return_value=b'{}'):
            with self.assertRaises(ValueError):policy.load_schedule()

    def test_full_stock_is_bound_to_graph_roles_and_64_workers(self):
        m = importlib.import_module('riblet.resources')
        for model,family in [('bert-cola','bert'),('vit-base-patch16-224','vit')]:
            graph={'model':model,'family':family,'model_rank':0,'data_rank':1,'method':'riblet-full',
                'runtime':{'max_concurrency':64,'field':'FM64','fxp_fraction_bits':18,'intra_op_parallel':False},
                'package_id':'a'*64,'graphs':{'request':'b'*64},'request_epoch':123,'model_epoch':456}
            budget=m.bind_budget(graph,'c'*64)
            self.assertFalse(budget['online_refill'])
            self.assertEqual(budget['lanes'],48)
            self.assertGreater(budget['reserved_bytes'],0)
            with self.assertRaises(MemoryError):m.require_headroom([1,2],budget)
            m.require_headroom([128*2**30,128*2**30],budget)
            bad={**graph,'runtime':{**graph['runtime'],'max_concurrency':2}}
            with self.assertRaises(ValueError):m.bind_budget(bad,'c'*64)
            with self.assertRaises(ValueError):m.validate_budget({**budget,'online_refill':True},graph,'c'*64)


if __name__=='__main__':unittest.main()
