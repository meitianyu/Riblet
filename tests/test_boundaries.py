import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class BoundaryTests(unittest.TestCase):
    def test_git_checkout_preserves_asset_hashes_and_linux_script_endings(self):
        import shutil

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root / "repo"
            repo.mkdir()
            empty_config = root / "git-config"
            empty_config.write_text("")
            env = dict(os.environ, GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=str(empty_config))
            for name in (".gitignore", ".gitattributes"):
                if (ROOT / name).is_file():
                    shutil.copy2(ROOT / name, repo / name)
            contents = {
                "assets/reproduction/example/lf.json": b'{"value":1}\n',
                "assets/reproduction/example/crlf.json": b'{"value":2}\r\n',
                "frontends/vit/vit_schedule.json": b'{"shape":[197]}\n',
                "tools/example.sh": b'#!/bin/sh\nprintf "ok\\n"\n',
            }
            for name, data in contents.items():
                path = repo / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)

            def git(*args):
                return subprocess.run(
                    ["git", "-C", str(repo), "-c", "core.autocrlf=true",
                     "-c", "core.hooksPath=" + str(root / "hooks"),
                     "-c", "user.name=Riblet test", "-c", "user.email=test@example.invalid", *args],
                    env=env, check=True, capture_output=True,
                )

            git("init", "-q")
            git("add", ".")
            git("commit", "-qm", "fixture")
            checkout = root / "checkout"
            git("checkout-index", "--all", "--prefix=" + checkout.as_posix() + "/")
            for name, expected in contents.items():
                self.assertEqual((checkout / name).read_bytes(), expected, name)

    def minimal_python(self, *args):
        env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), PYTHONDONTWRITEBYTECODE="1")
        return subprocess.run(
            [sys.executable, "-S", *args], env=env, capture_output=True, text=True
        )

    def test_cli_help_works_before_numerical_dependencies_are_installed(self):
        result = self.minimal_python("-m", "riblet.cli", "--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("resources", result.stdout)
        self.assertIn("compile", result.stdout)

    def test_data_resource_install_does_not_need_model_numerical_dependencies(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config = json.loads((ROOT / "riblet.example.json").read_text())
            config["host_role"] = "data"
            config["resources"].update(
                catalogue=str(ROOT / "assets/resources.json"),
                cache=str(root / "cache"),
                directory=str(root / "installed"),
            )
            path = root / "config.json"
            path.write_text(json.dumps(config))
            result = self.minimal_python(
                "-m", "riblet.cli", "resources", "--config", str(path), "--role", "data"
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            installed = root / "installed/preprocess-bert-v1/tokenizer.json"
            self.assertEqual(
                hashlib.sha256(installed.read_bytes()).hexdigest(),
                "79ea220416a57ca8acdf069d71875d5d2ddadef66021aeb6c120a7c546c04344",
            )
            self.assertFalse((root / "cache").exists())

    def test_identity_hashing_works_without_numerical_or_packaging_dependencies(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "payload"
            path.write_bytes(b"abc")
            code = (
                "import sys; from riblet.assets import digest,file_hash; "
                "print(digest({'b':2,'a':1})); print(file_hash(sys.argv[1]))"
            )
            result = self.minimal_python("-c", code, str(path))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                result.stdout.splitlines(),
                [
                    hashlib.sha256(b'{"a":1,"b":2}').hexdigest(),
                    "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
                ],
            )

    def test_unknown_action_is_rejected_without_reading_runtime_fields(self):
        from riblet.config import load_run_config

        with tempfile.TemporaryDirectory() as folder:
            config = dict(
                host_role="model",
                profile="paper",
                model={"name": "bert-cola"},
                parameters={"source": "frozen", "package_dir": "parameters"},
            )
            path = Path(folder) / "config.json"
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "unknown configuration action"):
                load_run_config(path, "model", "mistyped-action")

    def test_malformed_share_is_rejected_before_backend_loading(self):
        code = (
            "from riblet.transport import decode_share\n"
            "for packet in (b'bad', b'RBS1'+b'\\x00'*8+b'trailing'):\n"
            " try: decode_share(packet)\n"
            " except ValueError: pass\n"
            " else: raise AssertionError('invalid frame accepted')\n"
        )
        result = self.minimal_python("-c", code)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_dot_archive_member_rejected_without_publishing_destination(self):
        from riblet.resource_fetch import unpack_resource

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            archive = root / "resource.tar"
            with tarfile.open(archive, "w") as stream:
                member = tarfile.TarInfo(".")
                member.size = 3
                stream.addfile(member, io.BytesIO(b"abc"))
            members = [{"path": ".", "bytes": 3, "sha256": hashlib.sha256(b"abc").hexdigest()}]
            with self.assertRaises(ValueError):
                unpack_resource(archive, members, root / "installed")
            self.assertFalse((root / "installed").exists())


if __name__ == "__main__":
    unittest.main()
