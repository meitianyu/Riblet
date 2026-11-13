import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tarfile
import tempfile
import types
import unittest
from contextlib import redirect_stderr
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class PaperInputsTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('riblet.paper_inputs'),
                             'paper input preparation is missing')
        from riblet import paper_inputs
        self.p = paper_inputs

    def test_index_matches_final_references_not_early_three_sample_set(self):
        refs_root = Path(os.environ.get('PAPER_RESULTS_ROOT', ROOT / 'results/paper'))
        refs = json.loads((refs_root / 'references.json').read_text())
        rows = self.p.load_paper_index()['requests']
        self.assertEqual(len(rows), 18)
        self.assertEqual({r['content_id'] for r in rows}, {r['request_id'] for r in refs})
        for r in refs:
            x = next(x for x in rows if x['content_id'] == r['request_id'])
            self.assertEqual(x['tensor_sha256'], r.get('tensor_sha256', r.get('input_sha256')))

    def test_input_action_is_owner_local_without_runtime_or_weights(self):
        from riblet.config import load_run_config
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'config.json'
            doc = dict(host_role='data', profile='paper', model=dict(name='bert-cola', preprocessor_dir='tok'),
                       parameters=dict(source='frozen'), datasets=dict(directory='datasets'))
            path.write_text(json.dumps(doc))
            c = load_run_config(path, 'data', 'inputs')
            self.assertEqual(c['dataset_root'], Path(d) / 'datasets')
            self.assertNotIn('checkpoint', c)
            doc['host_role'] = 'model'; path.write_text(json.dumps(doc))
            with self.assertRaises(PermissionError): load_run_config(path, 'model', 'inputs')

    def test_wrong_model_selection_rejected_and_explicit_input_is_not_paper(self):
        with tempfile.TemporaryDirectory() as d:
            c = dict(role='data', model='bert-mrpc', profile='paper', directory=Path(d),
                     paper_sample='samples-eval-v2-cola-01', inputs=None)
            with self.assertRaises(ValueError): self.p.resolve_paper_input(c)
            c['inputs'] = Path(d) / 'other.json'
            path, binding = self.p.resolve_paper_input(c)
            self.assertEqual(path, c['inputs']); self.assertEqual(binding['kind'], 'user-input')
            self.assertNotIn('content_id', binding)

    def test_hash_uses_elements_not_npy_header_and_detects_corruption(self):
        with tempfile.TemporaryDirectory() as d:
            x = np.zeros((3, 1, 128), dtype=np.int32); x[1,0,:2] = 1
            row = dict(content_id='fixture', model='bert-cola', shape=[3,1,128], effective_length=2,
                       tensor_sha256=hashlib.sha256(x.tobytes()).hexdigest())
            path = self.p.save_input(Path(d), row, x)
            self.p.check_tensor(path, row)
            self.assertNotEqual(hashlib.sha256(path.read_bytes()).hexdigest(), row['tensor_sha256'])
            x[0,0,0] = 12; np.save(path, x)
            with self.assertRaises(ValueError): self.p.check_tensor(path, row)

    def test_bert_source_id_is_not_row_offset(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        from riblet.preprocessing import load_tokenizer, tokenize_bert
        processor = ROOT / 'assets/reproduction/preprocess-bert-v1'

        processor = next((ROOT / 'assets/reproduction').rglob('tokenizer.json')).parent
        tok = load_tokenizer(processor / 'tokenizer.json')
        raw = dict(idx=2472, sentence1='A cat.', sentence2='A dog.')
        values = tokenize_bert('mrpc', raw, tok)
        x = np.stack([values[k] for k in ('input_ids','attention_mask','token_type_ids')])
        row = dict(content_id='fixture-mrpc', model='bert-mrpc', task='mrpc', split='validation',
                   source_id='2472', original_row_index_zero_based=1, effective_length=values['effective_length'],
                   shape=[3,1,128], tensor_sha256=hashlib.sha256(x.tobytes()).hexdigest())
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); (root/'mrpc').mkdir()
            pq.write_table(pa.Table.from_pylist([dict(raw, idx=1923), raw]), root/'mrpc/validation-00000-of-00001.parquet')
            self.p.prepare_bert_inputs(root, processor, [row], root/'out')
            self.assertTrue((root/'out/fixture-mrpc.npy').exists())
            row['original_row_index_zero_based'] = 0
            with self.assertRaises(ValueError): self.p.prepare_bert_inputs(root, processor, [row], root/'other')

    def test_complete_archive_gate_rejects_missing_and_changed_file(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); (root/'train.tar').write_bytes(b'train'); (root/'val.tar').write_bytes(b'val')
            files=[dict(path=n, md5=hashlib.md5(b).hexdigest()) for n,b in [('train.tar',b'train'),('val.tar',b'val'),('test.tar',b'test')]]
            with self.assertRaises(FileNotFoundError): self.p.verify_files(root, files)
            (root/'test.tar').write_bytes(b'test')
            self.p.verify_files(root, files)
            (root/'val.tar').write_bytes(b'bad')
            with self.assertRaises(ValueError): self.p.verify_files(root, files)

    def test_vit_pixels_recovered_from_named_archive_member(self):
        from PIL import Image
        from transformers import ViTImageProcessor
        processor = next((ROOT/'assets/reproduction').rglob('preprocessor_config.json')).parent
        im=Image.new('RGB',(256,260),(33,99,127)); raw=io.BytesIO(); im.save(raw,format='JPEG')
        with Image.open(io.BytesIO(raw.getvalue())) as image:
            x=ViTImageProcessor.from_pretrained(str(processor),local_files_only=True)(images=image.convert('RGB'),return_tensors='np')['pixel_values']
        row=dict(content_id='fixture-vit',source_id='ILSVRC2012_val_00030431.JPEG',model='vit-base-patch16-224',
                 shape=[1,3,224,224],tensor_sha256=hashlib.sha256(x.tobytes()).hexdigest())
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            with tarfile.open(root/'ILSVRC2012_img_val.tar','w') as tar:
                info=tarfile.TarInfo(row['source_id']); info.size=len(raw.getvalue()); tar.addfile(info,io.BytesIO(raw.getvalue()))
            self.p.prepare_vit_inputs(root,processor,[row],root/'out')
            np.testing.assert_array_equal(np.load(root/'out/fixture-vit.npy'),x)

    def test_model_cannot_resolve_private_input(self):
        with self.assertRaises(PermissionError): self.p.resolve_paper_input(dict(role='model'))

    def test_failed_protocol_retains_local_input_binding_without_peer_arguments(self):
        from riblet.cli import _run
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            c = dict(role='data',model='bert-cola',inputs=root/'custom.json',parties={},graph=root/'graph',
                     parameters=root/'params',state_dir=root/'state',result=root/'result',processor=root/'tok',
                     network='lan',request_id='fresh-01')
            def failed_protocol(**kwargs):
                self.assertNotIn('paper_sample', kwargs)
                self.assertNotIn('input_sha256', kwargs)
                kwargs['result_dir'].mkdir()
                (kwargs['result_dir']/'FAILED.json').write_text('{}')
                raise RuntimeError('protocol failure fixture')
            with patch('riblet.cli._activate_deployment'), patch.dict('sys.modules',{
                'riblet.party':types.SimpleNamespace(run_full=failed_protocol)}):
                with self.assertRaises(RuntimeError): _run(c)
            self.assertTrue((root/'result/paper-input.json').is_file(), 'failed execution lost input binding')
            binding=json.loads((root/'result/paper-input.json').read_text())
            self.assertEqual(binding['execution_status'],'failed')
            self.assertNotIn('content_id',binding)

    def _export_failure(self, protocol_error=None):
        from riblet.cli import _run
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            c=dict(role='data',model='bert-cola',inputs=root/'input.npy',parties={},graph=root/'graph',
                   parameters=root/'params',state_dir=root/'state',result=root/'result',processor=root/'tok',
                   network='lan',request_id='fresh-export')
            def protocol(**kwargs):
                c['result'].mkdir()
                if protocol_error:
                    (c['result']/'FAILED.json').write_text('{"status":"failed"}')
                    raise protocol_error
                value=dict(status='complete',online_latency_seconds=1.25)
                (c['result']/'result.json').write_text(json.dumps(value))
                return value
            write=Path.write_text
            def write_without_space(path,*args,**kwargs):
                if path.name=='paper-input.json':raise OSError('fixture disk quota exhausted')
                return write(path,*args,**kwargs)
            stderr=io.StringIO();result=None;caught=None
            with patch('riblet.cli._activate_deployment'),patch.dict('sys.modules',{
                'riblet.party':types.SimpleNamespace(run_full=protocol)}),patch.object(Path,'write_text',write_without_space),redirect_stderr(stderr):
                try:result=_run(c)
                except Exception as error:caught=error
            stored=json.loads((c['result']/('FAILED.json' if protocol_error else 'result.json')).read_text())
            return result,caught,stored,stderr.getvalue()

    def test_export_failure_does_not_replace_completed_protocol(self):
        result,error,stored,warning=self._export_failure()
        self.assertIsNone(error, 'metadata export must not turn completed MPC into a failed command')
        self.assertEqual(result['status'],'complete')
        self.assertEqual(result['online_latency_seconds'],1.25)
        self.assertEqual(result['provenance_export_status'],'failed')
        self.assertEqual(stored,dict(status='complete',online_latency_seconds=1.25))
        self.assertIn('provenance',warning.lower())

    def test_export_failure_preserves_original_protocol_exception(self):
        original=RuntimeError('fixture MPC failure')
        result,error,stored,warning=self._export_failure(original)
        self.assertIs(error,original,'metadata export obscured the original protocol error')
        self.assertIsNone(result)
        self.assertEqual(stored['status'],'failed')
        self.assertIn('provenance',warning.lower())

    def test_lifecycle_export_failure_keeps_input_execution_complete(self):
        from riblet.cli import _run
        from riblet.measurement import LifecycleSession
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            c=dict(role='data',model='bert-cola',inputs=root/'input.npy',parties={},graph=root/'graph',
                   parameters=root/'params',state_dir=root/'state',result=root/'result',processor=root/'tok',
                   network='lan',request_id='fresh-lifecycle-export')
            def protocol(**kwargs):
                c['result'].mkdir()
                identity={k:'test' for k in ('model','method','implementation','configuration','request_id','session_id','layout','network','parameter')}
                with LifecycleSession(identity,directory=c['result'],rank=1,kind='validation'):
                    value=dict(status='complete',online_latency_seconds=1.25)
                    (c['result']/'result.json').write_text(json.dumps(value))
                    return value
            with patch('riblet.cli._activate_deployment'),patch.dict('sys.modules',{'riblet.party':types.SimpleNamespace(run_full=protocol)}),patch('riblet.measurement.os.link',side_effect=OSError('disk quota')),redirect_stderr(io.StringIO()):
                try: result=_run(c)
                except OSError as error:self.fail('completed inference rejected: '+str(error))
            self.assertEqual(result['status'],'complete')
            self.assertEqual(json.loads((c['result']/'paper-input.json').read_text())['execution_status'],'complete')


if __name__ == '__main__': unittest.main()
