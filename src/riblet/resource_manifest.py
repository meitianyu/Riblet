import re, json, hashlib
from urllib.parse import urlsplit


def validate_manifest(document):
    if document.get("schema") != "riblet-resources-v1":
        raise ValueError("unsupported resource schema")
    records = document.get("resources")
    if not isinstance(records, list):
        raise ValueError("resource list required")
    indexed = {}
    for r in records:
        key = r.get("resource_id", "")
        if not re.fullmatch(r"[a-z0-9][a-z0-9.-]*", key) or key in indexed:
            raise ValueError("invalid or duplicate resource ID")
        if not re.fullmatch(r"v[1-9][0-9]*", r.get("version", "")):
            raise ValueError("fixed version required")
        if type(r.get("bytes")) is not int or r["bytes"] < 0:
            raise ValueError("invalid size")
        if not re.fullmatch("[0-9a-f]{64}", r.get("sha256", "")):
            raise ValueError("invalid SHA256")
        for field, allowed in [("roles", {"model", "data"}), ("routes", {"frozen", "construct"})]:
            if not isinstance(r.get(field), list) or not r[field] or not set(r[field]) <= allowed:
                raise ValueError("invalid " + field)
        if not isinstance(r.get("models"), list) or not r["models"]:
            raise ValueError("models required")
        if not isinstance(r.get("dependencies"), list) or not isinstance(r.get("urls"), list):
            raise ValueError("dependencies/urls required")
        for url in r["urls"]:
            p = urlsplit(url)
            if p.scheme != "https" or not p.netloc or p.username or p.password:
                raise ValueError("public HTTPS URL required")
        form = r.get("format", "tar")
        if form not in ("tar", "file", "bundled"):
            raise ValueError("unknown resource format")
        if form != "tar":
            members = r.get("members", [])
            supplied = r.get("bundled", {})
            if (
                not members
                or len({m["path"] for m in members}) != len(members)
                or not set(supplied) <= {m["path"] for m in members}
            ):
                raise ValueError("incomplete resource members")
            remote = [m for m in members if m["path"] not in supplied]
            if form == "file" and (
                len(remote) != 1
                or remote[0]["sha256"] != r["sha256"]
                or remote[0]["bytes"] != r["bytes"]
            ):
                raise ValueError("file content identity differs")
            if form == "bundled":
                identity = hashlib.sha256(
                    json.dumps(members, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest()
                if (
                    remote
                    or r["urls"]
                    or r["bytes"] != sum(m["bytes"] for m in members)
                    or r["sha256"] != identity
                ):
                    raise ValueError("repository member identity differs")
        indexed[key] = r
    done = set()
    active = set()

    def visit(key):
        if key not in indexed or key in active:
            raise ValueError("missing/cyclic dependency")
        if key in done:
            return
        active.add(key)
        for dep in indexed[key]["dependencies"]:
            visit(dep)
        active.remove(key)
        done.add(key)

    for key in indexed:
        visit(key)
    return document


def resolve_resources(document, model, route, role):
    validate_manifest(document)
    if route not in ("frozen", "construct") or role not in ("model", "data"):
        raise ValueError("invalid route/role")
    indexed = {r["resource_id"]: r for r in document["resources"]}
    selected = {}

    def visit(r):
        if role not in r["roles"] or route not in r["routes"]:
            raise ValueError("dependency crosses role/route boundary")
        if r["resource_id"] in selected:
            return
        for key in r["dependencies"]:
            visit(indexed[key])
        selected[r["resource_id"]] = r

    for r in indexed.values():
        if model in r["models"] and role in r["roles"] and route in r["routes"]:
            visit(r)
    return list(selected.values())
