import json, os, subprocess, sys
from pathlib import Path
import abi


class LocalPointService:
    def __init__(self, work, master, *, workers=28):
        self.work = Path(work)
        self.fit_root = self.work / "fitter"
        self.work.mkdir(parents=True, exist_ok=True)
        self.fit_root.mkdir(exist_ok=True)
        self.workers = workers
        if type(workers) is not int or not 1 <= workers <= 64:
            raise ValueError("invalid fitter CPU worker budget")
        self._immutable(self.fit_root / "population.json", Path(master).read_bytes())
        self._immutable(
            self.fit_root / "CONSTRUCTION_POLICY.json",
            json.dumps({"mode": abi.CONSTRUCTION_POLICY}, sort_keys=True).encode(),
        )

    @staticmethod
    def _immutable(path, data):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if path.read_bytes() != data:
                raise ValueError("immutable local point artifact differs")
        else:
            with path.open("xb") as stream:
                stream.write(data)

    def folder(self, point):
        if type(point) is not int or not 0 <= point < 37:
            raise ValueError("unknown causal point")
        return self.fit_root / "points" / f"{point:02d}"

    def status(self, point):
        folder = self.folder(point)
        for name in ("DONE.json", "FAILED.json", "progress.json"):
            if (folder / name).exists():
                return json.loads((folder / name).read_bytes())
        return {}

    def submit(self, point, path):
        if path.name not in ("operands.npz", "meta.json"):
            raise ValueError("unexpected fitter input")
        self._immutable(
            self.fit_root / "inputs" / self.folder(point).name / path.name, path.read_bytes()
        )

    def publish(self, point, path):
        self._immutable(
            self.fit_root / "evidence" / (self.folder(point).name + ".json"), path.read_bytes()
        )

    def selection(self, point):
        folder = self.folder(point)
        if not (folder / "DONE.json").exists():
            raise ValueError("point has not passed numerical admission")
        selected = json.loads((folder / "selection.json").read_bytes())
        abi.validate_admitted(selected)
        return selected

    def start(self, point):
        folder = self.folder(point)
        env = {
            **os.environ,
            "CDI_ARRAY_BACKEND": "numpy",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "PYTHONPATH": os.pathsep.join(
                [str(Path(__file__).parent), str(Path(__file__).resolve().parents[2] / "src")]
            ),
        }
        env.pop("RIBLET_COST_ENVIRONMENT", None)
        if (folder / "FAILED.json").exists():
            raise ValueError("failed local point needs explicit recovery")
        folder.mkdir(parents=True, exist_ok=True)
        with (folder / "local-fit.log").open("ab") as log:
            result = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(Path(__file__).with_name("fit_point.py")),
                    "--root",
                    str(self.fit_root),
                    "--point",
                    str(point),
                    "--workers",
                    str(self.workers),
                ],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        if result.returncode:
            raise RuntimeError("local point fit failed; see " + str(folder / "local-fit.log"))
        status = self.status(point)
        if (
            status.get("status") != "fit_complete"
            or status.get("construction_policy") != abi.CONSTRUCTION_POLICY
        ):
            raise RuntimeError("fitter exited without a terminal result; inspect local-fit.log")
        return status
