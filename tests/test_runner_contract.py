import importlib,unittest


class RunnerContractTests(unittest.TestCase):
    def test_lifecycle_write_error_does_not_fail_completed_work(self):
        import io, tempfile
        from contextlib import redirect_stderr
        from unittest.mock import patch
        from riblet.measurement import LifecycleSession
        identity={k:'test' for k in ('model','method','implementation','configuration','request_id','session_id','layout','network','parameter')}
        with tempfile.TemporaryDirectory() as d:
            life=LifecycleSession(identity,directory=d,rank=1,kind='validation')
            stderr=io.StringIO()
            with patch('riblet.measurement.os.link',side_effect=OSError('disk quota')),redirect_stderr(stderr):
                try:
                    with life:
                        result={'status':'complete','online_latency_seconds':1.25}
                except OSError as error:
                    self.fail('auxiliary export replaced completed work: '+str(error))
            self.assertEqual(result['status'],'complete')
            self.assertEqual(life.status,'complete')
            self.assertFalse(life.output_path.exists())
            self.assertIn('lifecycle',stderr.getvalue().lower())

    def test_lifecycle_write_error_keeps_original_protocol_error(self):
        import io, tempfile
        from contextlib import redirect_stderr
        from unittest.mock import patch
        from riblet.measurement import LifecycleSession
        identity={k:'test' for k in ('model','method','implementation','configuration','request_id','session_id','layout','network','parameter')}
        original=RuntimeError('protocol failed')
        with tempfile.TemporaryDirectory() as d:
            life=LifecycleSession(identity,directory=d,rank=0,kind='validation')
            with patch('riblet.measurement.os.link',side_effect=OSError('disk quota')),redirect_stderr(io.StringIO()):
                with self.assertRaises(RuntimeError) as caught:
                    with life:raise original
            self.assertIs(caught.exception,original)
            self.assertEqual(life.status,'failed')

    def test_phase_total_uses_sender_delta_not_sum_of_nested_scopes(self):
        import time
        m=importlib.import_module('riblet.party')
        self.assertTrue(hasattr(m,'phase_total'),'outer preparation accounting missing')
        class Counter:
            def sent(self):return 17
        row=m.phase_total(time.monotonic_ns(),5,Counter())
        self.assertEqual(row['sent_bytes'],12)
        self.assertGreaterEqual(row['seconds'],0)
        with self.assertRaises(ValueError):m.phase_total(time.monotonic_ns(),18,Counter())
    def test_failed_connection_cost_is_unknown_not_zero(self):
        import tempfile
        from riblet.measurement import LifecycleSession
        m=importlib.import_module('riblet.party')
        identity={k:'test' for k in ('model','method','implementation','configuration','request_id','session_id','layout','network','parameter')}
        def failed():raise RuntimeError('connection failed after unknown traffic')
        with tempfile.TemporaryDirectory() as d:
            life=LifecycleSession(identity,directory=d,rank=0,kind='validation')
            with self.assertRaises(RuntimeError):m.initialize_session(life,failed)
            self.assertIsNone(life.events[0]['local_sent_bytes'])
    def test_session_initialization_records_real_sender_counter(self):
        import tempfile
        from riblet.measurement import LifecycleSession
        m=importlib.import_module('riblet.party')
        self.assertTrue(hasattr(m,'initialize_session'),'session initialization accounting missing')


        class Connection:
            def sent(self):return 17
        identity={k:'test' for k in ('model','method','implementation','configuration','request_id','session_id','layout','network','parameter')}
        with tempfile.TemporaryDirectory() as d:
            with LifecycleSession(identity,directory=d,rank=0,kind='validation') as life:
                value,row=m.initialize_session(life,Connection)
                self.assertIsInstance(value,Connection)
                self.assertEqual(row['sent_bytes'],17)
                self.assertEqual(life.events[0]['local_sent_bytes'],17)
                self.assertEqual(life.initializations[0]['event'],'initialized')
    def test_graph_rejects_legacy_roles_foreign_method_and_wrong_ops(self):
        m=importlib.import_module('riblet.party')
        self.assertTrue(hasattr(m,'validate_graph_identity'),'graph role/method gate missing')
        graph=dict(model='vit-base-patch16-224',family='vit',layout='vit-standard',method='riblet-full',model_rank=0,data_rank=1,OPS=False)
        m.validate_graph_identity(graph)
        for change in ({'model_rank':1,'data_rank':0},{'OPS':True},{'method':'nimbus'},{'layout':'eff'},{'family':'bert'}):
            with self.assertRaises(ValueError):m.validate_graph_identity({**graph,**change})
    def test_owner_arguments_cannot_silently_fall_back_to_other_party(self):
        m = importlib.import_module('riblet.party')
        m.validate_owner_arguments(0,model='model-dir',private_parameters='private-package',client_input=None,preprocessor=None)
        m.validate_owner_arguments(1,model=None,private_parameters=None,client_input='user-input',preprocessor='public-tokenizer')
        with self.assertRaises(PermissionError):m.validate_owner_arguments(0,model='m',private_parameters='p',client_input='x',preprocessor=None)
        with self.assertRaises(PermissionError):m.validate_owner_arguments(1,model='m',private_parameters=None,client_input='x',preprocessor='t')


if __name__=='__main__':unittest.main()
