"""Fetch a fixed resource selection before any compilation or online work."""

import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from riblet.resource_manifest import resolve_resources
from riblet.resource_fetch import fetch_resource, install_resource


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "model", "role", "route", "cache"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--local-directory")
    p.add_argument("--offline", action="store_true")
    p.add_argument("--unpack-to")
    a = p.parse_args()
    document = json.loads(Path(a.manifest).read_text())
    selected = resolve_resources(document, a.model, a.route, a.role)
    if not selected:
        raise ValueError("no resources matched; check model/route/role")
    receipts = []
    for r in selected:
        if r.get("format") == "bundled":
            if not a.unpack_to:
                raise ValueError("repository resources require --unpack-to")
            destination = Path(a.unpack_to) / (r["resource_id"] + "-" + r["version"])
            install_resource(r, None, Path(a.manifest).resolve().parent, destination)
            receipts.append(
                dict(
                    resource_id=r["resource_id"],
                    status="repository-verified",
                    installed_path=str(destination),
                )
            )
            continue
        name = r.get("filename", "")
        if Path(name).name != name or not name or "\\" in name:
            raise ValueError("invalid bundle filename")
        local = None if a.local_directory is None else Path(a.local_directory) / name
        if r.get("source") == "official" and local is not None and not local.is_file():
            local = None
        receipt = fetch_resource(r, a.cache, local_file=local, offline=a.offline)
        if a.unpack_to:
            destination = Path(a.unpack_to) / (r["resource_id"] + "-" + r["version"])
            install_resource(r, receipt["path"], Path(a.manifest).resolve().parent, destination)
            receipt["installed_path"] = str(destination)
        receipts.append(receipt)
    print(json.dumps(dict(status="verified", resources=receipts)))


if __name__ == "__main__":
    main()
