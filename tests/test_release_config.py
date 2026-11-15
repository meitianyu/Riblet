import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from riblet import config


class ReleaseConfigTest(unittest.TestCase):
    def test_audit_epoch_end_consumer_uses_public_audit_only(self):
        secret = "tensor<33x!pphlo.secret<i64>>"
        code = (
            f"func.func @main(%arg0: {secret}) -> ({secret}, {secret}) {{\n"
            f'  %1 = pphlo.custom_call @"mjv_audit_snapshot:p:1:2"(%arg0) : ({secret}) -> {secret}\n'
            f'  %2 = pphlo.custom_call @"mjv_epoch_end:p:1:2"(%arg0, %1) : ({secret}, {secret}) -> {secret}\n'
            f"  return %2, %1 : {secret}, {secret}\n}}"
        )
        fixed = config.public_audit(code)
        self.assertIn(f'"mjv_epoch_end:p:1:2"(%arg0, %1) : ({secret}, tensor<33xi64>) -> {secret}', fixed)
        self.assertIn(f"%arg0: {secret}", fixed)
        self.assertIn(f"return %2, %1 : {secret}, tensor<33xi64>", fixed)
        self.assertEqual(fixed.count("tensor<33xi64>"), 4)

    def test_audit_typing_does_not_treat_region_returns_as_function_return(self):
        secret = "tensor<33x!pphlo.secret<i64>>"
        code = (
            f"func.func @main(%arg0: {secret}) -> ({secret}, {secret}) {{\n"
            f"  %0 = pphlo.reduce(%arg0) {{\n"
            f"    pphlo.return %arg0 : {secret}\n"
            f"  }} : ({secret}) -> {secret}\n"
            f'  %1 = pphlo.custom_call @"mjv_audit_snapshot:p:1:2"(%0) : ({secret}) -> {secret}\n'
            f"  return %0, %1 : {secret}, {secret}\n}}"
        )
        fixed = config.public_audit(code)
        self.assertIn(f"pphlo.return %arg0 : {secret}", fixed)
        self.assertIn(f"%arg0: {secret}", fixed)
        self.assertIn(f"return %0, %1 : {secret}, tensor<33xi64>", fixed)
        self.assertEqual(fixed.count('tensor<33xi64>'), 3)

    def test_audit_result_does_not_retype_a_secret_argument_or_logit(self):
        secret = "tensor<33x!pphlo.secret<i64>>"
        code = (
            f"func.func @main(%arg0: {secret}) -> ({secret}, {secret}) {{\n"
            f'  %1 = pphlo.custom_call @"mjv_audit_snapshot:p:1:2"(%arg0) : ({secret}) -> {secret}\n'
            f"  return %arg0, %1 : {secret}, {secret}\n}}"
        )
        fixed = config.public_audit(code)
        self.assertIn(f"%arg0: {secret}", fixed)
        self.assertIn(f" -> ({secret}, tensor<33xi64>)", fixed)
        self.assertIn(f"(%arg0) : ({secret}) -> tensor<33xi64>", fixed)
        self.assertIn(f"return %arg0, %1 : {secret}, tensor<33xi64>", fixed)
        with self.assertRaises(ValueError):
            config.public_audit(code.replace("return %arg0, %1", "return %arg0, %arg0"))

    def test_paths_follow_configuration_and_host_role(self):
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            document = json.loads(
                (Path(__file__).resolve().parents[1] / "riblet.example.json").read_text()
            )
            path = folder / "run.json"
            path.write_text(json.dumps(document))
            cfg = config.load_run_config(path, "model")
            self.assertEqual(cfg["work"], folder / "work/model")
            self.assertEqual(cfg["checkpoint"], folder / "resources/model-bert-cola-v1")
            with self.assertRaises(ValueError):
                config.load_run_config(path, "controller")
            document["runtime"]["resource_profile"] = "arbitrary"
            path.write_text(json.dumps(document))
            with self.assertRaises(ValueError):
                config.load_run_config(path, "model")

    def test_selection_is_order_independent_and_folds_follow_selected_order(self):
        from riblet import development

        rows = [{"idx": str(i)} for i in range(17)]
        result = development.select_rows(rows, "idx", 6, 511, "v1/cola")
        self.assertEqual(result, development.select_rows(rows[::-1], "idx", 6, 511, "v1/cola"))
        self.assertEqual(len({r["idx"] for r in result}), 6)
        with self.assertRaises(ValueError):
            development.select_rows(rows + rows[:1], "idx", 6, 511, "v1/cola")

    def test_custom_vit_checks_real_row_identity_and_dense_ordinals(self):
        from riblet.development import image_identity
        from riblet.packages import digest

        rows = [
            dict(
                request_ordinal=0,
                relative_id="n00000001/a.jpg",
                fold_id=0,
                class_id=1,
                sha256="f" * 64,
            )
        ]
        doc = dict(
            profile="custom",
            rows=rows,
            dataset_version="owned-v1",
            selection={"mode": "sample", "count": 1, "seed": 1},
        )
        doc["population_id"] = digest(doc)
        self.assertEqual(image_identity(doc, custom=True), doc["population_id"])
        doc["rows"][0]["request_ordinal"] = 7
        with self.assertRaises(ValueError):
            image_identity(doc, custom=True)

    def test_stock_rejects_an_unadmitted_custom_trace(self):
        from riblet import resources

        with self.assertRaises(ValueError):
            resources.require_schedule("bert-cola", "0" * 64)

    def test_one_config_cannot_select_a_model_only_action_on_data_host(self):
        from riblet.cli import main

        path = Path(__file__).resolve().parents[1] / "riblet.example.json"
        with self.assertRaises(PermissionError):
            main(["prepare-model", "--role", "data", "--config", str(path)])

    def test_paper_sample_override_is_not_ignored(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.json"
            doc = json.loads(
                (Path(__file__).resolve().parents[1] / "riblet.example.json").read_text()
            )
            doc["construction"]["selection"] = {"mode": "sample", "count": 5, "seed": 1}
            path.write_text(json.dumps(doc))
            with self.assertRaises(ValueError):
                config.load_run_config(path, "model")

    def test_custom_images_keep_original_class_and_verify_the_bytes(self):
        import hashlib
        from riblet.development import custom_inputs, image_identity

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "images/n00000100").mkdir(parents=True)
            candidates = []
            for idx in range(17):
                name = f"n00000100/{idx}.JPEG"
                raw = f"opaque fixture {idx}".encode()
                (root / "images" / name).write_bytes(raw)
                candidates.append(
                    dict(
                        relative_id=name,
                        split="train",
                        class_id=89,
                        bytes=len(raw),
                        sha256=hashlib.sha256(raw).hexdigest(),
                    )
                )
            (root / "candidates.json").write_text(json.dumps(candidates))
            cfg = dict(
                directory=root,
                family="vit",
                model="vit-base-patch16-224",
                construction=dict(
                    dataset_version="fixture-v1",
                    data_root=".",
                    candidates="candidates.json",
                    selection=dict(mode="sample", count=17, seed=42),
                ),
            )
            custom_inputs(cfg, root / "out")
            doc = json.loads((root / "out/manifest.json").read_text())
            self.assertEqual(doc["population_id"], image_identity(doc, custom=True))
            self.assertEqual([r["request_ordinal"] for r in doc["rows"]], list(range(17)))
            self.assertEqual({r["class_id"] for r in doc["rows"]}, {89})
            (root / "images" / doc["rows"][0]["relative_id"]).write_bytes(b"changed")
            with self.assertRaises(ValueError):
                custom_inputs(cfg, root / "changed")
            self.assertFalse((root / "changed/manifest.json").exists())

    def test_budget_limits_apply_per_operator_before_stock_reservation(self):
        from riblet.resources import require_schedule

        from riblet.bert_tables import slots
        data=[dict(operator=o,site=s,point=p,enabled=True,K=8) for o,s,p in slots()]
        require_schedule("bert-mrpc", data)
        data[-1]["K"] = 9
        with self.assertRaises(ValueError):
            require_schedule("bert-mrpc", data)

    def test_custom_checkpoint_can_change_training_origin_but_not_forward_shape(self):
        from riblet.assets import architecture_id

        a = {"hidden_size": 768, "_name_or_path": "trained/checkpoint"}
        b = {"hidden_size": 768, "_name_or_path": "new/checkpoint"}
        self.assertEqual(architecture_id(a), architecture_id(b))
        b["hidden_size"] = 1024
        self.assertNotEqual(architecture_id(a), architecture_id(b))

    def test_custom_processor_must_match_the_bound_file(self):
        from riblet.assets import verify_preprocessor
        import hashlib

        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "preprocessor_config.json"
            p.write_bytes(b'{"size":224}')
            expected = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()}
            self.assertEqual(
                verify_preprocessor("vit-base-patch16-224", p.parent, bound=expected), expected
            )
            p.write_bytes(b'{"size":225}')
            with self.assertRaises(ValueError):
                verify_preprocessor("vit-base-patch16-224", p.parent, bound=expected)

    def test_real_request_binding_is_accepted_by_ledger(self):
        from riblet.party import request_bindings
        from riblet.lifecycle import RequestLedger

        graph = {"graphs": {"request": "a" * 64}, "package_id": "p", "model": "bert-cola"}
        material, peer = request_bindings(graph, {"binding_sha256": "b"}, "lan", "q1")
        self.assertEqual(set(material), {"graph", "parameters", "budget", "model", "roles"})
        with TemporaryDirectory() as tmp:
            store = RequestLedger(tmp)
            ticket = store.reserve("same-epoch", material)
            ticket.prepared(*[dict(rank=i, complete=True, binding=material) for i in (0, 1)])
            ticket.begin_online()
            ticket.retire()
            again, other = request_bindings(graph, {"binding_sha256": "b"}, "wan", "q2")
            self.assertEqual(again, material)
            self.assertNotEqual(other, peer)
            with self.assertRaises(FileExistsError):
                store.reserve("same-epoch", again)

    def test_data_run_needs_no_model_or_construction_paths(self):
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "run.json"
            doc = json.loads(
                (Path(__file__).resolve().parents[1] / "riblet.example.json").read_text()
            )
            doc["host_role"] = "data"
            doc.pop("construction")
            doc.pop("resources")
            doc["model"].pop("checkpoint_dir")
            doc["parameters"].pop("frozen_source")
            doc["parameters"]["package_dir"] = "public-only"
            p.write_text(json.dumps(doc))
            c = config.load_run_config(p, "data", "run")
            self.assertEqual(c["parameters"], Path(tmp) / "public-only")
            self.assertNotIn("construction", c)

    def test_package_copy_failure_does_not_publish_partial_directory(self):
        from riblet import packages
        from test_public_privacy import descriptor
        from unittest.mock import patch
        import shutil

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.write_bytes(b"params")
            real = shutil.copy2

            def failing_copy(src, dst):
                real(src, dst)
                raise OSError("interrupted copy")

            with patch.object(packages.shutil, "copy2", side_effect=failing_copy):
                with self.assertRaises(OSError):
                    packages._write(root / "package", descriptor("bert-cola"), {"table": source})
            self.assertFalse((root / "package").exists())

    def test_completed_package_is_readable_and_content_bound(self):
        from riblet.packages import _write, digest
        from test_public_privacy import descriptor

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            p = root / "source"
            p.write_bytes(b"table")
            result = _write(root / "package", descriptor("bert-cola"), {"table": p})
            self.assertEqual(
                json.loads((root / "package/public/parameters.json").read_text()), result
            )
            self.assertEqual((root / "package/private/table").read_bytes(), b"table")
            self.assertEqual(
                result["package_id"], digest({k: v for k, v in result.items() if k != "package_id"})
            )

    def test_input_preparation_failure_leaves_no_final_construction(self):
        from riblet.cli import _prepare_model
        from unittest.mock import patch

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = dict(
                family="bert",
                model="bert-cola",
                parameters=root / "package",
                source="construct",
                profile="paper",
                construction_root=root / "construction",
                directory=root,
                checkpoint=root / "model",
                processor=root / "processor",
                construction=dict(
                    data_root="raw",
                    models_root="models",
                    membership="members.json",
                    selection={"mode": "paper"},
                ),
            )

            def interrupted(family, membership, raw, output, **kw):
                output.write_bytes(b"partial")
                raise OSError("input copy interrupted")

            with patch("riblet.development.prepare_construction_inputs", side_effect=interrupted):
                with self.assertRaises(OSError):
                    _prepare_model(cfg, False)
            self.assertFalse(cfg["construction_root"].exists())

    def test_audit_rejects_missing_duplicate_and_indirect_results(self):
        code = 'func.func @main(%x: tensor<33x!pphlo.secret<i64>>) -> tensor<33x!pphlo.secret<i64>> {\n  %a = pphlo.custom_call @"mjv_audit_snapshot:p:1:2"(%x) : (tensor<33x!pphlo.secret<i64>>) -> tensor<33x!pphlo.secret<i64>>\n  return %a : tensor<33x!pphlo.secret<i64>>\n}'
        for bad in (
            code.replace("mjv_audit_snapshot", "other"),
            code.replace("return %a", "return %x"),
            code.replace("  return", code.splitlines()[1] + "\n  return"),
        ):
            with self.assertRaises(ValueError):
                config.public_audit(bad)

    def test_config_rejects_typo_and_paper_construct_has_processor(self):
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "run.json"
            doc = json.loads(
                (Path(__file__).resolve().parents[1] / "riblet.example.json").read_text()
            )
            doc["parameters"]["source"] = "construct"
            p.write_text(json.dumps(doc))
            cfg = config.load_run_config(p, "model", "prepare-model")
            self.assertIn("processor", cfg)
            doc["construction"]["workres"] = 8
            p.write_text(json.dumps(doc))
            with self.assertRaisesRegex(ValueError, "construction.workres"):
                config.load_run_config(p, "model", "prepare-model")

    def test_construction_semantics_exclude_workers_and_local_paths(self):
        from riblet.config import construction_policy

        c = dict(
            profile="custom",
            model="bert-cola",
            construction=dict(
                dataset_version="v1",
                split="train",
                workers=2,
                data_root="/one",
                selection=dict(mode="sample", count=40, seed=7),
            ),
        )
        first = construction_policy(c)
        c["construction"].update(workers=9, data_root="/two")
        self.assertEqual(first, construction_policy(c))
        c["construction"]["selection"]["seed"] = 8
        self.assertNotEqual(first, construction_policy(c))

    def test_blocking_fitter_requires_a_terminal_result(self):
        import importlib.util, sys
        from unittest.mock import patch
        from types import SimpleNamespace

        folder = Path(__file__).resolve().parents[1] / "offline/vit"
        sys.path.insert(0, str(folder))
        spec = importlib.util.spec_from_file_location(
            "local_fit_contract", folder / "local_point.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            master = root / "master.json"
            master.write_text("{}")
            service = module.LocalPointService(root / "work", master)
            with patch.object(module.subprocess, "run", return_value=SimpleNamespace(returncode=0)):
                with self.assertRaisesRegex(RuntimeError, "terminal"):
                    service.start(0)
            (service.folder(0) / "progress.json").write_text(
                json.dumps(
                    {"status": "fit_complete", "construction_policy": "model-local-numeric-v1"}
                )
            )
            with patch.object(module.subprocess, "run", return_value=SimpleNamespace(returncode=0)):
                self.assertEqual(service.start(0)["status"], "fit_complete")

    def test_model_binding_uses_explicit_processor_directory(self):
        from riblet import assets
        from unittest.mock import patch
        import hashlib

        with TemporaryDirectory() as tmp:
            r = Path(tmp)
            (r / "model").mkdir()
            (r / "processor").mkdir()
            (r / "registry").mkdir()
            blobs = {
                "config.json": b"{}",
                "flax_model.msgpack": b"weights",
                "preprocessor_config.json": b"processor",
            }
            expected = {k: hashlib.sha256(v).hexdigest() for k, v in blobs.items()}
            for k, v in blobs.items():
                (r / ("processor" if k == "preprocessor_config.json" else "model") / k).write_bytes(
                    v
                )
            (r / "registry/model_identities.json").write_text(
                json.dumps({"models": {"vit-base-patch16-224": expected}})
            )
            with patch.object(assets, "ASSETS", r / "registry"):
                actual = assets.verify_model(
                    "vit-base-patch16-224", r / "model", preprocessor=r / "processor"
                )
            self.assertEqual(actual, expected)

    def test_old_runtime_path_is_not_an_installation_binding(self):
        from riblet.environment import resolve

        with self.assertRaisesRegex(ValueError, "installation"):
            resolve({"runtimes": {"bert": "missing"}}, "bert")

    def test_frontend_exposes_joint_selected_schedule_only(self):
        import importlib.util

        p = Path(__file__).resolve().parents[1] / "frontends/bert/cdi_bert_integration.py"
        import sys

        spec = importlib.util.spec_from_file_location("release_bert_adapter", p)
        m = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = m
        spec.loader.exec_module(m)
        self.assertTrue(m.mode_uses_operator("joint", "reciprocal"))
        with self.assertRaises(ValueError):
            m.mode_uses_operator("reciprocal-only", "reciprocal")
