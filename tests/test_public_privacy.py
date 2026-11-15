import json
import tempfile
import unittest
import numpy as np
from pathlib import Path
from unittest.mock import patch

from riblet import packages as p, vit_parameters as v


def descriptor(model, kind="constructed-deployment", custom=False):
    doc = p._base(model, kind)
    if model.startswith("bert-"):
        doc.update(
            graph_id=p.BERT_GRAPH,
            numeric_abi=p.BERT_ABI,
            lineage=p.BERT_LINEAGE,
            schedule=[dict(operator=o,site=t,point=i,enabled=True,K=1) for o,t,i in p.bert_tables.slots()],
        )
    else:
        doc.update(
            graph_id=v.GRAPH_ID,
            numeric_abi=v.NUMERIC_ABI,
            lineage="Joint",
            coefficient_slots=5434,
            schedule=[
                dict(
                    point=i,
                    operator=v.operator(i),
                    route=[1] * int(np.prod(v.point_shape(i))),
                    K=[0] * int(np.prod(v.point_shape(i))),
                    cost_receipt="private-receipt",
                )
                for i in range(37)
            ],
        )
    doc.update(
        table_sha256="a" * 64,
        population_binding="b" * 64,
        fold_binding="c" * 64,
        construction=dict(prefix="private-prefix", requests=123),
    )
    if custom:
        doc["profile"] = "custom"
    return doc


class PublicPrivacyTests(unittest.TestCase):
    def test_real_bert_packers_integrity_and_owner_binding(self):
        for custom in (False, True):
            worlds = []
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                for world in range(2):
                    directory=root/f"source{world}";directory.mkdir()
                    sites=[dict(task="cola",operator=o,site=t,point=i,enabled=True,K=1) for o,t,i in p.bert_tables.slots()]
                    document=dict(schema=p.bert_tables.SCHEMA,layout="pad128",sites=sites)
                    schedule=directory/"public-schedule.json";schedule.write_text(json.dumps(document))
                    path=directory/"private-sites.json"
                    document["sites"]=[dict(r,a_encoded=3,b_encoded=4+world) for r in sites]
                    path.write_text(json.dumps(document))
                    known=p._asset("parameter_identities.json")
                    known["bert"]={f.name:p.file_hash(f) for f in (path,schedule)}
                    real_asset = p._asset
                    folder = root / f"pkg{world}"
                    with patch.object(
                        p,
                        "_asset",
                        side_effect=lambda name: (
                            known if name == "parameter_identities.json" else real_asset(name)
                        ),
                    ), patch("secrets.token_hex", return_value="d" * 64):
                        if custom:
                            evidence = dict(
                                status="causal_construction_complete",deployment_allowed=True,
                                table_sha256=p.file_hash(path),
                                graph_id=p.BERT_GRAPH,
                                profile="custom",
                                task_counts={"cola": 1},
                                model_files={
                                    "cola": p._base("bert-cola", "constructed-deployment")[
                                        "model_files"
                                    ]
                                },
                                population_semantic_sha256=str(world) * 64,
                                task_results=[
                                    dict(
                                        task="cola",
                                        status="causal_task_complete",
                                        mode="joint",
                                        graph_id=p.BERT_GRAPH,
                                        lineage=p.BERT_LINEAGE,
                                        physical_length=128,
                                        sites=169,points=37,
                                        requests=1,
                                        normal_design_mismatches=0,
                                        design_bits=12,
                                        design_inclusive=False,
                                    )
                                ],
                            )
                            ev = root / f"ev{world}.json"
                            ev.write_text(json.dumps(evidence))
                            public = p.pack_constructed_bert(path, "bert-cola", folder, ev)
                        else:
                            public = p.pack_frozen_bert(directory, "bert-cola", folder)
                        owner = p.load_model_package(folder, rank=0)
                        self.assertEqual(owner["private_hashes"]["private-sites.json"], p.file_hash(path))
                        self.assertEqual(p.load_public(folder / "public"), public)
                        self.assertEqual(
                            (folder / "private/private-sites.json").read_bytes(), path.read_bytes()
                        )
                        manifest = folder / "private/manifest.json"
                        original = manifest.read_text()
                        bad = json.loads(original)
                        bad["deployment_id"] = "e" * 64
                        manifest.write_text(json.dumps(bad))
                        with self.assertRaisesRegex(ValueError, "manifest/public"):
                            p.load_model_package(folder, rank=0)
                        manifest.write_text(original)
                        (folder / "private/private-sites.json").write_bytes(b"corrupt")
                        with self.assertRaisesRegex(ValueError, "^private parameter file differs: private-sites\\.json$"):
                            p.load_model_package(folder, rank=0)
                        worlds.append(public)
                self.assertEqual(worlds[0], worlds[1])

    def test_frozen_vit_import_preserves_payload_and_checks_integrity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            points = descriptor("vit-base-patch16-224")["schedule"]
            meta = dict(
                graph_id=v.GRAPH_ID,
                numeric_abi=v.NUMERIC_ABI,
                model_revision=v.MODEL_REVISION,
                lineage="Joint",
                OPS=False,
                status="audited_cost_closed",
                population_binding="a" * 64,
                fold_binding="b" * 64,
                points=points,
            )
            (source / "public-deployment-metadata.json").write_text(json.dumps(meta))
            np.save(source / "coefficients-encoded-private.npy", np.zeros((5434, 2), np.uint64))
            known = p._asset("parameter_identities.json")
            known["vit"] = {f.name: p.file_hash(f) for f in source.iterdir()}
            original = p._asset
            with patch.object(
                p,
                "_asset",
                side_effect=lambda name: (
                    known if name == "parameter_identities.json" else original(name)
                ),
            ):
                public = p.pack_frozen_vit(source, root / "pkg")
                self.assertEqual(
                    p.load_model_package(root / "pkg", rank=0)["package_id"], public["package_id"]
                )
                self.assertEqual(p.load_public(root / "pkg/public"), public)
                file = root / "pkg/private/coefficients-encoded-private.npy"
                self.assertEqual(file.read_bytes(), (source / file.name).read_bytes())
                file.write_bytes(b"corrupt")
                with self.assertRaises(ValueError):
                    p.load_model_package(root / "pkg", rank=0)

    def test_unselected_private_content_does_not_change_public_export(self):
        for model in ("bert-cola", "vit-base-patch16-224"):
            for kind, custom in (
                ("frozen", False),
                ("constructed-deployment", False),
                ("constructed-deployment", True),
            ):
                with self.subTest(
                    model=model, kind=kind, custom=custom
                ), tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    worlds = []
                    for i in range(2):
                        doc = descriptor(model, kind, custom)
                        doc["table_sha256"] = str(i) * 64
                        doc["model_files"]["flax_model.msgpack"] = str(i) * 64
                        doc["population_binding"] = str(i) * 64
                        doc["construction"]["prefix"] = str(i) * 64
                        if not model.startswith("bert-"):
                            doc["schedule"][0]["cost_receipt"] = str(i) * 64
                        source = root / f"source{i}"
                        source.write_bytes(b"selected=7;unselected=" + bytes([i]))

                        with patch("secrets.token_hex", return_value="d" * 64):
                            worlds.append(p._write(root / str(i), doc, {"table": source}))
                        self.assertEqual(
                            (root / str(i) / "private/table").read_bytes(), source.read_bytes()
                        )
                    self.assertEqual(worlds[0], worlds[1])
                    self.assertEqual(p.graph_binding(worlds[0]), p.graph_binding(worlds[1]))
                    text = json.dumps(worlds[0])
                    for key in (
                        "private_hashes",
                        "table_sha256",
                        "model_files",
                        "population_binding",
                        "fold_binding",
                        "cost_receipt",
                        "construction",
                        "prefix",
                    ):
                        self.assertNotIn(key, text)
                    self.assertEqual(p.load_public(root / "0/public"), worlds[0])

    def test_ids_persist_but_new_deployments_are_independent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.write_bytes(b"same")
            a = p._write(root / "a", descriptor("bert-cola"), {"table": source})
            b = p._write(root / "b", descriptor("bert-cola"), {"table": source})
            self.assertNotEqual(a["package_id"], b["package_id"])
            self.assertEqual(p.load_public(root / "a"), p.load_public(root / "a"))
            p.require_graph_binding(dict(a), a)
            with self.assertRaises(ValueError):
                p.require_graph_binding(dict(a), b)

    def test_input_descriptor_has_no_hash(self):
        from riblet.cli import _describe

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            proc = base / "proc"
            proc.mkdir()
            (proc / "preprocessor_config.json").write_text("{}")
            doc = descriptor("vit-base-patch16-224", custom=True)
            doc["model_files"]["preprocessor_config.json"] = p.file_hash(
                proc / "preprocessor_config.json"
            )
            source = base / "table"
            source.write_bytes(b"table")
            p._write(base / "params", doc, {"table": source})
            descriptions = []
            for i in range(2):
                inp = base / f"{i}.npy"
                np.save(inp, np.full((1, 3, 224, 224), i, np.float32))
                config = dict(
                    model="vit-base-patch16-224",
                    profile="custom",
                    parameters=base / "params/public",
                    processor=proc,
                    description=base / f"desc{i}.json",
                    request_id="request",
                )

                with patch("riblet.paper_inputs.resolve_paper_input", return_value=(inp, None)):
                    descriptions.append(_describe(config))
            self.assertEqual(descriptions[0], descriptions[1])
            self.assertEqual(
                set(descriptions[0]),
                {"shape", "tokens", "parameter_package", "model", "layout", "request"},
            )

    def test_unsupported_schema_and_injected_fingerprints_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.write_bytes(b"secret")
            doc = p._write(root / "pkg", descriptor("bert-cola"), {"table": source})
            path = root / "pkg/public/parameters.json"
            for change in (
                {"schema": "riblet-parameter-package-v1"},
                {"private_hashes": {"x": "a" * 64}},
            ):
                bad = dict(doc, **change)
                bad["package_id"] = p.digest({k: v for k, v in bad.items() if k != "package_id"})
                path.write_text(json.dumps(bad))
                with self.assertRaises(ValueError):
                    p.load_public(path.parent)


if __name__ == "__main__":
    unittest.main()
