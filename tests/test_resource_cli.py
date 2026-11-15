import hashlib, json, subprocess, sys, tempfile, unittest
from pathlib import Path


class CLITests(unittest.TestCase):
    def test_repeated_unpack_verifies_existing_contents(self):
        import io, tarfile

        tool = Path(__file__).resolve().parents[1] / "tools/fetch_resources.py"
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / "files").mkdir()
            archive = p / "files/x.tar"
            with tarfile.open(archive, "w") as tar:
                info = tarfile.TarInfo("x")
                info.size = 3
                tar.addfile(info, io.BytesIO(b"abc"))
            r = dict(
                resource_id="x",
                version="v1",
                filename="x.tar",
                bytes=archive.stat().st_size,
                sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                roles=["model"],
                routes=["frozen"],
                models=["bert-cola"],
                dependencies=[],
                urls=[],
                members=[dict(path="x", bytes=3, sha256=hashlib.sha256(b"abc").hexdigest())],
            )
            (p / "manifest.json").write_text(
                json.dumps(dict(schema="riblet-resources-v1", resources=[r]))
            )
            command = [
                sys.executable,
                str(tool),
                "--manifest",
                str(p / "manifest.json"),
                "--model",
                "bert-cola",
                "--role",
                "model",
                "--route",
                "frozen",
                "--cache",
                str(p / "cache"),
                "--local-directory",
                str(p / "files"),
                "--unpack-to",
                str(p / "assets"),
            ]
            for _ in range(2):
                result = subprocess.run(command, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
            (p / "assets/x-v1/x").write_bytes(b"bad")
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)

    def test_local_acquisition_from_real_manifest(self):
        tool = Path(__file__).resolve().parents[1] / "tools/fetch_resources.py"
        self.assertTrue(tool.exists())
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / "files").mkdir()
            (p / "files/x.bin").write_bytes(b"abc")
            r = dict(
                resource_id="x",
                version="v1",
                filename="x.bin",
                bytes=3,
                sha256=hashlib.sha256(b"abc").hexdigest(),
                roles=["model"],
                routes=["frozen"],
                models=["bert-cola"],
                dependencies=[],
                urls=[],
            )
            (p / "manifest.json").write_text(
                json.dumps(dict(schema="riblet-resources-v1", resources=[r]))
            )
            command = [
                sys.executable,
                str(tool),
                "--manifest",
                str(p / "manifest.json"),
                "--model",
                "bert-cola",
                "--role",
                "model",
                "--route",
                "frozen",
                "--cache",
                str(p / "cache"),
                "--local-directory",
                str(p / "files"),
            ]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["resources"][0]["sha256"], r["sha256"])

    def test_weights_and_repository_config_install_together(self):
        from riblet.resource_fetch import install_resource

        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / "weights").write_bytes(b"weights")
            (p / "config.json").write_bytes(b"config")
            members = [
                dict(path=n, bytes=len(v), sha256=hashlib.sha256(v).hexdigest())
                for n, v in [("flax_model.msgpack", b"weights"), ("config.json", b"config")]
            ]
            r = dict(format="file", members=members, bundled={"config.json": "config.json"})
            install_resource(r, p / "weights", p, p / "installed")
            self.assertEqual((p / "installed/flax_model.msgpack").read_bytes(), b"weights")
            install_resource(r, p / "weights", p, p / "installed")
            (p / "config.json").write_bytes(b"corrupt")
            with self.assertRaises(ValueError):
                install_resource(r, p / "weights", p, p / "new")

    def test_repository_only_resource_needs_no_download(self):
        from riblet.resource_fetch import install_resource

        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / "tokenizer.json").write_bytes(b"tokenizer")
            r = dict(
                format="bundled",
                members=[
                    dict(
                        path="tokenizer.json",
                        bytes=9,
                        sha256=hashlib.sha256(b"tokenizer").hexdigest(),
                    )
                ],
                bundled={"tokenizer.json": "tokenizer.json"},
            )
            install_resource(r, None, p, p / "installed")
            self.assertEqual((p / "installed/tokenizer.json").read_bytes(), b"tokenizer")
            with self.assertRaises(ValueError):
                install_resource(
                    dict(r, bundled={"tokenizer.json": "../outside"}), None, p, p / "bad"
                )

    def test_real_data_role_requires_only_repository_preprocessing(self):
        from riblet.resource_manifest import resolve_resources

        root = Path(__file__).resolve().parents[1]
        catalogue = json.loads((root / "assets/resources.json").read_text())
        for model in ("bert-cola", "vit-base-patch16-224"):
            rows = resolve_resources(catalogue, model, "frozen", "data")
            self.assertTrue(rows)
            self.assertTrue(
                all(r["source"] == "repository" and r["format"] == "bundled" for r in rows)
            )
            with tempfile.TemporaryDirectory() as d:
                result = subprocess.run(
                    [
                        sys.executable,
                        str(root / "tools/fetch_resources.py"),
                        "--manifest",
                        str(root / "assets/resources.json"),
                        "--model",
                        model,
                        "--role",
                        "data",
                        "--route",
                        "frozen",
                        "--cache",
                        str(Path(d) / "cache"),
                        "--unpack-to",
                        str(Path(d) / "assets"),
                    ],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertFalse((Path(d) / "cache").exists())
