import hashlib,json,tempfile,unittest
import contextlib,io,os,re
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]

class ConstructionRoutesTests(unittest.TestCase):
    def test_image_copy_preserves_identity_and_rejects_changed_bytes(self):
        from riblet.development import copy_image
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'raw/c').mkdir(parents=True);(root/'raw/c/a.JPEG').write_bytes(b'image')
            row=dict(relative_id='c/a.JPEG',bytes=5,sha256=hashlib.sha256(b'image').hexdigest())
            copy_image(root/'raw',root/'out',row)
            self.assertEqual((root/'out/c/a.JPEG').read_bytes(),b'image')
            (root/'raw/c/a.JPEG').write_bytes(b'wrong')
            with self.assertRaises(ValueError):copy_image(root/'raw',root/'other',row)
            with self.assertRaises(ValueError):copy_image(root/'raw',root/'other',dict(row,relative_id='../escape'))

    def test_membership_rejects_altered_fold(self):
        from riblet.development import validate_vit_membership
        with self.assertRaises(ValueError):validate_vit_membership({'rows':[dict(request_ordinal=0,relative_id='x',fold_id=0,class_id=0,sha256='a'*64)]})

    def _training_export_example(self):
        doc=(ROOT/'docs/CONSTRUCTION.md').read_text(encoding='utf-8')
        match=re.search(r"\"\$PY\" - <<'PY'\n(# Export the fixed GLUE training splits.*?)(?=\nPY\n~~~)",doc,re.S)
        self.assertIsNotNone(match,'the model-owner training export instructions are missing')
        return compile(match[1],str(ROOT/'docs/CONSTRUCTION.md'),'exec')

    def test_documented_training_export_preserves_rows_and_reuses_equal_outputs(self):
        import pyarrow as pa
        import pyarrow.ipc as ipc
        import pyarrow.parquet as pq
        code=self._training_export_example()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);snapshot=root/'snapshot';files=[];tables={}
            for task in ('cola','sst2','mrpc','stsb','qqp','mnli','qnli','rte'):
                table=pa.table({'idx':pa.array([17,3],type=pa.int32()),'sentence':['First text','Second text'],
                                'label':pa.array([1,0],type=pa.int64())})
                path=snapshot/task/'train-00000-of-00001.parquet';path.parent.mkdir(parents=True)
                pq.write_table(table,path);tables[task]=pq.read_table(path)
                files.append(dict(path=path.relative_to(snapshot).as_posix(),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            cfg=dict(host_role='model',profile='paper',parameters=dict(source='construct'),
                     construction=dict(split='train',dataset_version='fixed-test-revision',data_root='training'))
            (root/'riblet.json').write_text(json.dumps(cfg));(root/'assets').mkdir()
            (root/'assets/paper-inputs.json').write_text(json.dumps(dict(datasets=dict(glue=dict(repo_id='nyu-mll/glue',revision='fixed-test-revision',files=files)))))
            cwd=Path.cwd()
            try:
                os.chdir(root)
                with patch('huggingface_hub.snapshot_download',return_value=str(snapshot)),contextlib.redirect_stdout(io.StringIO()):
                    exec(code,{})
                    for task,expected in tables.items():
                        with (root/'training'/task/'glue-train.arrow').open('rb') as stream:
                            actual=ipc.open_stream(stream).read_all()
                        self.assertTrue(actual.equals(expected,check_metadata=True))
                        self.assertEqual(actual['idx'].to_pylist(),[17,3])
                    before={p:p.read_bytes() for p in (root/'training').glob('*/glue-train.arrow')}
                    exec(code,{})
                    self.assertEqual(before,{p:p.read_bytes() for p in before})
                    target=root/'training/cola/glue-train.arrow'
                    with target.open('wb') as stream,ipc.new_stream(stream,tables['cola'].schema) as writer:
                        writer.write_table(tables['cola'].slice(0,1))
                    changed=target.read_bytes()
                    with self.assertRaises(ValueError):exec(code,{})
                    self.assertEqual(target.read_bytes(),changed)
            finally:os.chdir(cwd)

    def test_documented_training_export_rejects_data_party(self):
        code=self._training_export_example()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'riblet.json').write_text(json.dumps(dict(host_role='data',profile='paper',parameters=dict(source='construct'),construction=dict(split='train'))))
            cwd=Path.cwd()
            try:
                os.chdir(root)
                with patch('huggingface_hub.snapshot_download',side_effect=AssertionError('data party attempted construction download')):
                    with self.assertRaises(PermissionError):exec(code,{})
                self.assertFalse((root/'training').exists())
            finally:os.chdir(cwd)
