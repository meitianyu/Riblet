import unittest
from importlib.util import find_spec

class ResourceManifestTests(unittest.TestCase):
    def test_cycle_and_cross_role_dependency_rejected(self):
        from riblet.resource_manifest import validate_manifest,resolve_resources
        common=dict(version='v1',bytes=3,sha256='a'*64,roles=['model'],routes=['frozen'],models=['bert-cola'],urls=[])
        a=dict(common,resource_id='a',dependencies=['b']);b=dict(common,resource_id='b',dependencies=['a'])
        with self.assertRaises(ValueError):validate_manifest(dict(schema='riblet-resources-v1',resources=[a,b]))
        b.update(dependencies=[],roles=['data'])
        with self.assertRaises(ValueError):resolve_resources(dict(schema='riblet-resources-v1',resources=[a,b]),'bert-cola','frozen','model')

    def test_missing_validator_is_a_release_gap(self):
        self.assertIsNotNone(find_spec('riblet.resource_manifest'))

    def test_invalid_and_duplicate_resources_fail(self):
        from riblet.resource_manifest import validate_manifest
        r=dict(resource_id='model-a',version='v1',bytes=3,sha256='a'*64,
               roles=['model'],routes=['frozen'],models=['bert-cola'],dependencies=[],urls=[])
        self.assertEqual(len(validate_manifest(dict(schema='riblet-resources-v1',resources=[r]))['resources']),1)
        for rows in ([r,r],[dict(r,bytes=-1)],[dict(r,sha256='bad')],[dict(r,dependencies=['missing'])]):
            with self.assertRaises(ValueError):validate_manifest(dict(schema='riblet-resources-v1',resources=rows))

    def test_construct_does_not_select_frozen_parameters(self):
        from riblet.resource_manifest import resolve_resources
        common=dict(version='v1',bytes=3,sha256='a'*64,roles=['model'],models=['bert-cola'],dependencies=[],urls=[])
        doc=dict(schema='riblet-resources-v1',resources=[dict(common,resource_id='model',routes=['construct','frozen']),dict(common,resource_id='coeff',routes=['frozen'])])
        self.assertEqual([r['resource_id'] for r in resolve_resources(doc,'bert-cola','construct','model')],['model'])
        self.assertEqual(resolve_resources(doc,'bert-cola','frozen','data'),[])
