import copy, importlib, json, os, subprocess, sys, tempfile, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VIT = ROOT / "offline/vit"


def numeric_fixture():
    import abi
    from vit_cdi import layout

    counts = {
        c: sum(layout.condition("rsqrt", p) == c for p in range(197))
        for c in layout.conditions("rsqrt")
    }
    rows = {
        c: dict(
            route="SPU_FALLBACK",
            public_Ks=0,
            candidate_id=None,
            source=None,
            a_encoded=None,
            b_encoded=None,
            L_encoded=None,
            H_encoded=None,
            numeric_abi=abi.NUMERIC["rsqrt"],
        )
        for c in counts
    }
    return dict(
        graph=abi.GRAPH_ID,
        point=0,
        operator="rsqrt",
        population="a" * 64,
        input_meta=dict(
            graph=abi.GRAPH_ID,
            point=0,
            population="a" * 64,
            prefix="b" * 64,
            operand_sha256="c" * 64,
        ),
        sites=[
            dict(
                site=0,
                routes=rows,
                normal_events=0,
                fallback_events=197,
                iteration_events=0,
                condition_event_counts=counts,
            )
        ],
        candidate_count=0,
        expected_candidates=0,
        arrived_requests=1,
        construction_policy="model-local-numeric-v1",
    )


class ModelLocalTests(unittest.TestCase):
    def test_vit_large_predicates_at_strict_and_inclusive_boundaries(self):
        import numpy as np
        from unittest.mock import patch

        with patch.object(sys, 'path', [str(VIT), *sys.path]):
            from vit_cdi.reference_numeric import precision_mask
            x = np.full(256, 1 << 18, dtype=np.int64)
            y = x.copy()
            y[1], y[2] = 0, (1 << 18) + 64
            for op in ('reciprocal', 'rsqrt'):
                for bits in (12, 13):
                    expected = np.ones(256, dtype=bool)
                    expected[1], expected[2] = False, bits == 12
                    np.testing.assert_array_equal(precision_mask(op, x, y, bits), expected)



    def test_vit_prepare_and_resume_do_not_run_development_tests(self):
        code = '''
import json,tempfile,sys,types
from pathlib import Path
from unittest.mock import patch
import construct
from riblet.assets import digest
sys.modules['jax']=types.SimpleNamespace(__version__='metadata-only-fixture')
with tempfile.TemporaryDirectory() as folder:
    root=Path(folder); data=root/'data'; data.mkdir()
    doc=dict(profile='custom',rows=[dict(request_ordinal=0,relative_id='n00000001/a.JPEG',fold_id=0,class_id=1,sha256='f'*64)])
    doc['population_id']=digest(doc)
    (data/'manifest.json').write_text(json.dumps(doc))
    with patch.object(construct,'verify_model',return_value={'flax_model.msgpack':'a'*64}),patch('subprocess.run',side_effect=AssertionError('construction must not run the development test suite')):
        prepared=construct.prepare_run(root/'fit',root/'model',data,custom=True)
        admission=root/'fit/admission';admission.mkdir(exist_ok=True)
        (admission/'reference-check.log').write_text('retained old check failure')
        (admission/'cache0.npz').write_bytes(b'unused old file')
        resumed=construct.prepare_run(root/'fit',root/'model',data,custom=True,resume=True)
        assert resumed.identity==prepared.identity
        assert (root/'fit/fitter/population.json').read_bytes()==(data/'manifest.json').read_bytes()
        with patch.object(construct,'construction_sources',return_value={'changed-shared-code':'b'*64}):
            try:construct.prepare_run(root/'fit',root/'model',data,custom=True,resume=True)
            except ValueError:pass
            else:raise AssertionError('changed construction implementation resumed')
'''
        env=dict(os.environ,PYTHONPATH=os.pathsep.join((str(ROOT/'src'),str(VIT))),CDI_ARRAY_BACKEND='numpy',PYTHONDONTWRITEBYTECODE='1')
        result=subprocess.run([sys.executable,'-B','-c',code],env=env,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_construction_identity_covers_shared_code_not_docs(self):
        from unittest.mock import patch
        from riblet import offline
        self.assertTrue(hasattr(offline,'construction_sources'),'shared construction code is not bound')
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for name in ('assets','config','offline','development','preprocessing','bert_tables','vit_parameters'):
                p=root/'src/riblet'/f'{name}.py';p.parent.mkdir(parents=True,exist_ok=True);p.write_text('original')
            for family in ('bert','vit'):
                p=root/'offline'/family/'model.py';p.parent.mkdir(parents=True);p.write_text('model')
            with patch.object(offline,'ROOT',root):
                before=offline.construction_sources('bert')
                (root/'README.md').write_text('documentation only')
                self.assertEqual(offline.construction_sources('bert'),before)
                (root/'src/riblet/offline.py').write_text('changed encoder')
                self.assertNotEqual(offline.construction_sources('bert'),before)
                before=offline.construction_sources('vit')
                (root/'src/riblet/vit_parameters.py').write_text('changed schedule semantics')
                self.assertNotEqual(offline.construction_sources('vit'),before)

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(VIT))
        cls.abi = importlib.import_module("abi")

    def test_offline_configuration_without_counterparty_or_request(self):
        from riblet.config import load_run_config

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "config.json"
            value = dict(
                host_role="model",
                profile="custom",
                model=dict(
                    name="vit-base-patch16-224",
                    checkpoint_dir="model",
                    preprocessor_dir="processor",
                ),
                parameters=dict(source="construct", package_dir="parameters"),
                construction=dict(
                    split="train",
                    data_root="train",
                    dataset_version="owned-train-v1",
                    candidates="candidates.json",
                    selection=dict(mode="sample", count=1000, seed=7),
                    workers=2,
                    output_dir="construction",
                ),
            )
            p.write_text(json.dumps(value))
            c = load_run_config(p, "model", "prepare-model")
            self.assertEqual(c["construction_root"], Path(tmp) / "construction")
            self.assertNotIn("parties", c)
            self.assertNotIn("request_id", c)
            self.assertNotIn("installation", c)

    def test_numerical_selection_finalizes_without_cost_receipt(self):
        doc = numeric_fixture()
        result = self.abi.admit_numeric(doc)
        self.assertEqual(result["sites"], doc["sites"])
        self.assertEqual(result["cost_assessment"], "not_measured")
        self.assertNotIn("cost_admission", result)
        self.abi.validate_admitted(result)
        corrupt = copy.deepcopy(result)
        corrupt["input_meta"]["prefix"] = "d" * 64
        with self.assertRaises(ValueError):
            self.abi.validate_admitted(corrupt)

    def test_qualified_normal_is_not_replaced_by_performance_fallback(self):
        doc = numeric_fixture()
        row = doc["sites"][0]["routes"]["REGION0"]
        row.update(
            route="CDI_NORMAL",
            public_Ks=1,
            candidate_id="candidate",
            source="CHILD",
            a_encoded=-1,
            b_encoded=2,
            L_encoded=1,
            H_encoded=2,
        )
        row["selected_evidence"] = dict(
            qualified=True,
            feasible_Ks=[1],
            scores=[dict(mask=2, visited=12, scored_events=12) for _ in range(6)],
            **{k: row[k] for k in ("a_encoded", "b_encoded", "L_encoded", "H_encoded")},
        )
        n = doc["sites"][0]["condition_event_counts"]["REGION0"]
        doc["sites"][0].update(normal_events=n, fallback_events=197 - n, iteration_events=n)
        doc.update(candidate_count=1, expected_candidates=1)
        result = self.abi.admit_numeric(doc)
        self.assertEqual(result["sites"][0]["routes"]["REGION0"]["route"], "CDI_NORMAL")
        row["selected_evidence"]["scores"][0]["mask"] = 0
        with self.assertRaises(ValueError):
            self.abi.admit_numeric(doc)

    def test_imports_do_not_read_cost_environment_or_start_protocol(self):
        env = {
            **os.environ,
            "PYTHONPATH": os.pathsep.join((str(ROOT / "src"), str(VIT))),
            "CDI_ARRAY_BACKEND": "numpy",
            "RIBLET_COST_ENVIRONMENT": "/does-not-exist/forbidden-cost.json",
        }
        code = "import sys; import local_point, fit_point, audit; assert 'cost_contract' not in sys.modules; assert 'riblet.party' not in sys.modules"
        result = subprocess.run(
            [sys.executable, "-B", "-c", code], env=env, capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_new_public_package_is_numeric_not_cost_certified(self):
        from riblet import packages as p, vit_parameters as v

        with tempfile.TemporaryDirectory() as tmp:
            points = []
            for point in range(37):
                n = 2364 if point % 3 == 1 else 197
                row = dict(point=point, operator=v.operator(point), route=[1] * n, K=[0] * n)
                row["numerical_receipt"] = v.schedule_identity(
                    point, row["operator"], row["route"], row["K"]
                )
                points.append(row)
            doc = dict(
                p._base("vit-base-patch16-224", "constructed-deployment"),
                graph_id=v.GRAPH_ID,
                numeric_abi=v.NUMERIC_ABI,
                schedule=points,
                population_binding="a" * 64,
                private_hashes={"audit.json": "b" * 64},
                construction=dict(
                    origin="local-cpu-construction",
                    evidence_file="audit.json",
                    normal_design_mismatches=0,
                    points=37,
                    construction_policy="model-local-numeric-v1",
                    numerical_complete=True,
                    cost_assessment="not_measured",
                ),
            )
            doc["coefficient_slots"] = 5434
            doc = p._public_descriptor(doc, "d" * 64)
            path = Path(tmp) / "parameters.json"
            path.write_text(json.dumps(doc))
            self.assertEqual(p.load_public(tmp), doc)
            doc["schedule"][0]["K"][0] = 1
            doc["package_id"] = p.digest({k: v for k, v in doc.items() if k != "package_id"})
            path.write_text(json.dumps(doc))
            with self.assertRaises(ValueError):
                p.load_public(tmp)

    def test_local_admission_cannot_claim_a_measured_cost(self):
        doc = numeric_fixture()
        doc["native_cost_confirmed"] = True
        with self.assertRaises(ValueError):
            self.abi.admit_numeric(doc)

    def test_incomplete_enumeration_and_unknown_policy_are_rejected(self):
        for change in ({"candidate_count": 1}, {"construction_policy": "unregistered"}):
            with self.assertRaises(ValueError):
                self.abi.admit_numeric(dict(numeric_fixture(), **change))

    def test_complete_synthetic_chain_audits_and_packages_without_network(self):


        import numpy as np
        from riblet import packages as pack, development
        from vit_cdi import layout
        import audit, fit_point

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fit = root / "fitter"
            fit.mkdir()
            members = dict(
                profile="custom",
                dataset_version="fixture",
                rows=[
                    dict(
                        request_ordinal=0,
                        relative_id="n00000001/a.JPEG",
                        fold_id=0,
                        class_id=1,
                        sha256="f" * 64,
                    )
                ],
            )
            members["population_id"] = pack.digest(members)
            population = members["population_id"]
            (fit / "population.json").write_text(json.dumps(members))
            model_files = pack._base("vit-base-patch16-224", "constructed-deployment")[
                "model_files"
            ]
            binding = dict(
                profile="custom",
                requests=1,
                population=population,
                population_artifact_sha256=pack.file_hash(fit / "population.json"),
                model_files=model_files,
                construction_policy=self.abi.CONSTRUCTION_POLICY,
            )
            (root / "CONSTRUCTION_BINDING.json").write_text(json.dumps(binding))
            (fit / "CONSTRUCTION_POLICY.json").write_text(
                json.dumps({"mode": self.abi.CONSTRUCTION_POLICY})
            )
            prefix = []
            for point in range(37):
                op = "reciprocal" if point % 3 == 1 else "rsqrt"
                heads = 12 if op == "reciprocal" else 1
                inp = fit / "inputs" / f"{point:02d}"
                inp.mkdir(parents=True)
                np.savez(
                    inp / "operands.npz",
                    request_ids=np.array([0]),
                    folds=np.array([0]),
                    operands=np.full((1, heads, 197), 262144, np.int64),
                )
                meta = dict(
                    point=point,
                    graph=self.abi.GRAPH_ID,
                    population=population,
                    prefix=pack.digest(prefix),
                    operand_sha256=pack.file_hash(inp / "operands.npz"),
                )
                (inp / "meta.json").write_text(json.dumps(meta))
                sites = []
                for h in range(heads):
                    site = (
                        point // 3 * 12 + h
                        if heads == 12
                        else (24 if point == 36 else 2 * (point // 3) + int(point % 3 == 2))
                    )
                    counts = {
                        c: sum(layout.condition(op, i) == c for i in range(197))
                        for c in layout.conditions(op)
                    }
                    rows = {
                        c: dict(
                            route="SPU_FALLBACK",
                            public_Ks=0,
                            a_encoded=None,
                            b_encoded=None,
                            L_encoded=None,
                            H_encoded=None,
                            numeric_abi=self.abi.NUMERIC[op],
                        )
                        for c in counts
                    }
                    sites.append(
                        dict(
                            site=site,
                            routes=rows,
                            normal_events=0,
                            fallback_events=197,
                            iteration_events=0,
                            condition_event_counts=counts,
                        )
                    )
                doc = dict(
                    graph=self.abi.GRAPH_ID,
                    population=population,
                    point=point,
                    operator=op,
                    input_meta=meta,
                    sites=sites,
                    candidate_count=0,
                    expected_candidates=0,
                    construction_policy=self.abi.CONSTRUCTION_POLICY,
                )
                folder = fit / "points" / f"{point:02d}"
                folder.mkdir(parents=True)
                (folder / "numerical_selection.json").write_text(json.dumps(doc))
                (folder / "SCORING_DONE.json").write_text(
                    json.dumps(
                        dict(
                            numerical_digest=pack.digest(doc),
                            candidates=0,
                            scoring_seconds=0,
                            workers=1,
                        )
                    )
                )
                done = fit_point.finalize_numeric(fit, point)
                selected = json.loads((folder / "selection.json").read_text())
                out = root / "outputs" / f"{point:02d}"
                out.mkdir(parents=True)
                summary = dict(
                    prefix_before=pack.digest(prefix),
                    selection_digest=pack.digest(selected),
                    attempted_requests=1,
                    prepared_finite_requests=1,
                    prepare_invalid=0,
                    alive_after=1,
                    finish_invalid=0,
                    normal_events=0,
                    fallback_events=heads * 197,
                    executed_inverse_events=heads * 197,
                    unreached_inherited_events=0,
                    invalid_input_unexecuted_events=0,
                    theoretical_events=heads * 197,
                    normal_p12_failures=0,
                )
                evidence = dict(
                    summary=summary,
                    prepared=[dict(prepared_ids=[0])],
                    applied=[
                        dict(
                            request_ids=[0],
                            normal_p12_failures=0,
                            normal_events=0,
                            fallback_events=heads * 197,
                            predictions=[1] if point == 36 else [],
                        )
                    ],
                    request_state=dict(alive=[True]),
                )
                (out / "EVIDENCE.json").write_text(json.dumps(evidence))
                prefix.append(pack.digest(selected))
            (root / "CONSTRUCTION_DONE.json").write_text(
                json.dumps({"points": 37, "prefix": pack.digest(prefix)})
            )
            audit.audit(root)
            parameters = root / "constructed-joint"
            public = pack.pack_constructed_vit(parameters, root / "package")
            owner = pack.load_model_package(root / "package", rank=0)
            self.assertEqual(owner["package_id"], public["package_id"])
            self.assertEqual(owner["construction"]["cost_assessment"], "not_measured")
            self.assertNotIn("construction", public)
            self.assertNotIn("cost_closed", owner["construction"])
            self.assertEqual(pack.load_public(root / "package/public"), public)
            bad = json.loads((parameters / "audit.json").read_text())
            bad["normal_p12_failures"] = 1

            import shutil

            altered = root / "altered"
            shutil.copytree(parameters, altered)
            (altered / "audit.json").chmod(0o600)
            (altered / "audit.json").write_text(json.dumps(bad))
            with self.assertRaises(ValueError):
                pack.pack_constructed_vit(altered, root / "bad-package")
            for p in root.rglob("*"):
                if p.is_file():
                    p.chmod(0o600)
