import json,re,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from riblet.resource_manifest import resolve_resources
class PublishedResourceRoutes(unittest.TestCase):
 def test_model_frozen_route_has_pinned_downloads_without_local_archive_requirement(self):
  doc=json.loads((ROOT/'assets/resources.json').read_text())
  cfg=json.loads((ROOT/'riblet.example.json').read_text())
  self.assertIsNone(cfg['resources']['local_dir'])
  revisions=set(); assets=set()
  for model in ['bert-'+t for t in ['cola','sst2','mrpc','stsb','qqp','mnli','qnli','rte']]+['vit-base-patch16-224']:
   for r in resolve_resources(doc,model,'frozen','model'):
    if r.get('format')=='bundled' or r.get('source')=='official':continue
    self.assertEqual(r['source'],'huggingface')
    self.assertEqual(len(r['urls']),1)
    match=re.fullmatch(r'https://huggingface\.co/ethanMei/Riblet-artifacts/resolve/([0-9a-f]{40})/(?:v0\.1\.0|public-site-v1)/'+re.escape(r['filename']),r['urls'][0])
    self.assertIsNotNone(match)
    revisions.add(match[1]);assets.add(r['resource_id'])
  self.assertEqual(len(assets),10);self.assertTrue(revisions)
 def test_input_owner_does_not_acquire_model_or_parameter_blobs(self):
  doc=json.loads((ROOT/'assets/resources.json').read_text())
  for model in ('bert-cola','vit-base-patch16-224'):
   rows=resolve_resources(doc,model,'frozen','data')
   self.assertTrue(rows); self.assertTrue(all(r.get('format')=='bundled' and not r['urls'] for r in rows))
if __name__=='__main__':unittest.main()
