import re, shutil, tarfile, tempfile, os
from pathlib import Path, PurePosixPath
from urllib.request import Request, build_opener, HTTPRedirectHandler
from urllib.parse import urlsplit
from .assets import file_hash


def _canonical_member(name):
    path = PurePosixPath(name)
    return (
        bool(path.parts)
        and not path.is_absolute()
        and ".." not in path.parts
        and "\\" not in name
        and ":" not in name
        and str(path) == name
    )


def _url(url):
    p = urlsplit(url)
    if p.username or p.password or not p.hostname:
        raise ValueError("invalid public URL")
    if p.scheme != "https" and not (p.scheme == "http" and p.hostname in ("127.0.0.1", "::1")):
        raise ValueError("HTTPS required outside loopback tests")


class _Redirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _verify(path, resource):
    if path.is_symlink() or not path.is_file() or path.stat().st_size != resource["bytes"]:
        raise ValueError("resource size/type mismatch")
    if file_hash(path) != resource["sha256"]:
        raise ValueError("resource checksum mismatch")


def _import_local(resource, root, target, source):
    source = Path(source)
    _verify(source, resource)
    if shutil.disk_usage(root).free < resource["bytes"]:
        raise OSError("insufficient cache space")
    partial = root / (resource["sha256"] + ".importing")
    if partial.is_symlink():
        raise ValueError("unsafe import temporary path")
    if source.resolve() == partial.resolve():
        raise ValueError("import source aliases temporary path")
    with source.open("rb") as src, partial.open("wb") as dst:
        shutil.copyfileobj(src, dst, 1048576)
    _verify(partial, resource)
    partial.rename(target)


def _download(resource, root, target):
    partial = root / (resource["sha256"] + ".part")
    if partial.is_symlink():
        raise ValueError("unsafe partial path")
    if shutil.disk_usage(root).free < resource["bytes"]:
        raise OSError("insufficient cache space")
    errors = []
    for url in resource.get("urls", []):
        _url(url)
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            if offset >= resource["bytes"]:
                offset = 0
            headers = {"Accept-Encoding": "identity"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            with build_opener(_Redirect()).open(
                Request(url, headers=headers), timeout=60
            ) as response:
                _url(response.url)
                if "text/html" in response.headers.get("Content-Type", ""):
                    raise ValueError("HTML is not a resource")
                if response.status == 206:
                    expected = f'bytes {offset}-{resource["bytes"]-1}/{resource["bytes"]}'
                    if response.headers.get("Content-Range") != expected:
                        raise ValueError("invalid resume range")
                elif response.status == 200:
                    offset = 0
                else:
                    raise ValueError("unsupported download response")
                with partial.open("ab" if offset else "wb") as stream:
                    total = offset
                    while chunk := response.read(1048576):
                        total += len(chunk)
                        if total > resource["bytes"]:
                            raise ValueError("resource exceeds declared size")
                        stream.write(chunk)
            _verify(partial, resource)
            partial.rename(target)
            return
        except (OSError, ValueError) as error:
            errors.append(type(error).__name__)
    raise ValueError(
        "no verified resource obtained; check source permission/availability: " + ",".join(errors)
    )


def fetch_resource(resource, cache_root, *, local_file=None, offline=False):
    sha = resource.get("sha256", "")
    if (
        not re.fullmatch("[0-9a-f]{64}", sha)
        or type(resource.get("bytes")) is not int
        or resource["bytes"] < 0
    ):
        raise ValueError("invalid resource identity")
    root = Path(cache_root)
    root.mkdir(parents=True, exist_ok=True)
    target = root / sha
    receipt = dict(
        resource_id=resource["resource_id"],
        version=resource["version"],
        sha256=sha,
        bytes=resource["bytes"],
        path=str(target),
    )
    if target.exists():
        _verify(target, resource)
        return dict(receipt, status="cached")
    if offline:
        message = (
            "verified offline cache missing"
            if local_file is None
            else "offline mode permits verified cache only"
        )
        raise ValueError(message)


    from filelock import FileLock, Timeout

    try:
        with FileLock(str(root / (sha + ".lock")), timeout=0):
            if target.exists():
                _verify(target, resource)
                return dict(receipt, status="cached")
            if local_file is None:
                _download(resource, root, target)
                return dict(receipt, status="downloaded")
            _import_local(resource, root, target, local_file)
            return dict(receipt, status="imported")
    except Timeout as error:
        raise FileExistsError("another process is preparing resource " + sha) from error


def verify_installed(destination, members):

    destination = Path(destination)
    if destination.is_symlink() or not destination.is_dir():
        raise ValueError("invalid installed resource root")
    expected = {r["path"]: r for r in members}
    if len(expected) != len(members):
        raise ValueError("duplicate member identity")
    for name in expected:
        if not _canonical_member(name):
            raise ValueError("unsafe installed member")
    files = set()
    for path in destination.rglob("*"):
        if path.is_symlink():
            raise ValueError("installed resource contains link")
        if path.is_file():
            files.add(path.relative_to(destination).as_posix())
        elif not path.is_dir():
            raise ValueError("unexpected installed resource type")
    if files != set(expected):
        raise ValueError("installed resource member set differs")
    for name, row in expected.items():
        _verify(destination / name, row)
    return destination


def unpack_resource(archive, members, destination):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    expected = {r["path"]: r for r in members}
    if len(expected) != len(members):
        raise ValueError("duplicate declared members")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:*") as tar:
        infos = tar.getmembers()
        seen = set()
        for info in infos:
            if (
                not info.isfile()
                or not _canonical_member(info.name)
                or info.name in seen
                or info.name not in expected
            ):
                raise ValueError("unsafe archive member")
            if info.size != expected[info.name]["bytes"]:
                raise ValueError("member size differs")
            seen.add(info.name)
        if seen != set(expected):
            raise ValueError("missing archive members")
        if shutil.disk_usage(destination.parent).free < sum(i.size for i in infos):
            raise OSError("insufficient extraction space")
        with tempfile.TemporaryDirectory(prefix=".riblet-unpack-", dir=destination.parent) as tmp:
            stage = Path(tmp) / "data"
            stage.mkdir()
            for info in infos:
                output = stage / info.name
                output.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(info) as src, output.open("xb") as dst:
                    shutil.copyfileobj(src, dst, 1048576)
                _verify(output, expected[info.name])
                output.chmod(0o600)
            os.rename(stage, destination)
    return destination


def install_resource(resource, downloaded, repository, destination):

    destination = Path(destination)
    if destination.exists():
        return verify_installed(destination, resource["members"])
    repository = Path(repository).resolve()
    bundled = resource.get("bundled", {})
    members = {m["path"]: m for m in resource["members"]}
    if len(members) != len(resource["members"]) or not set(bundled) <= set(members):
        raise ValueError("invalid assembled member set")
    for name in members:
        if not _canonical_member(name):
            raise ValueError("unsafe resource member")
    remote = [m for name, m in members.items() if name not in bundled]
    form = resource.get("format", "tar")
    if form not in ("file", "tar", "bundled"):
        raise ValueError("invalid resource format")
    if (form == "file" and len(remote) != 1) or (form == "bundled" and remote):
        raise ValueError("invalid resource format")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".riblet-resource-", dir=destination.parent) as temp:
        stage = Path(temp) / "data"
        if form == "tar":
            unpack_resource(downloaded, remote, stage)
        else:
            stage.mkdir()
            if remote:
                _verify(Path(downloaded), remote[0])
                target = stage / remote[0]["path"]
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(downloaded, target)
        for name, relative in bundled.items():
            path = PurePosixPath(relative)
            source = repository / relative
            if (
                path.is_absolute()
                or ".." in path.parts
                or "\\" in relative
                or ":" in relative
                or source.is_symlink()
                or not source.resolve().is_relative_to(repository)
            ):
                raise ValueError("unsafe repository resource")
            _verify(source, members[name])
            target = stage / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        verify_installed(stage, resource["members"])
        os.rename(stage, destination)
    return destination
