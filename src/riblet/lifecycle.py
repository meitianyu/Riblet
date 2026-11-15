import json
from contextlib import contextmanager
from pathlib import Path
import re
import sqlite3
import time
import os
from functools import wraps


@contextmanager
def host_lease(path="/tmp/riblet-mpc-host.lock"):

    import fcntl

    fd = os.open(str(path), os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another Riblet workload owns this host") from error
        yield
    finally:
        os.close(fd)


def exclusive_host(function):
    @wraps(function)
    def guarded(*args, **kwargs):
        with host_lease():
            return function(*args, **kwargs)

    return guarded


def _identity(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}", value):
        raise ValueError("invalid request namespace")
    return value


class RequestLedger:
    def __init__(self, root):
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        self.path = root / "requests.sqlite"
        with self._db() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS requests (id TEXT PRIMARY KEY, binding TEXT NOT NULL, state TEXT NOT NULL, updated REAL NOT NULL, reason TEXT)"
            )

    @contextmanager
    def _db(self):
        db = sqlite3.connect(self.path, timeout=30)
        try:
            db.execute("PRAGMA synchronous=FULL")
            with db:
                yield db
        finally:
            db.close()

    def reserve(self, identity, binding):
        _identity(identity)
        if set(binding) != {"graph", "parameters", "budget", "model", "roles"} or binding[
            "roles"
        ] != [0, 1]:
            raise ValueError("complete fixed-role material binding required")
        if any(
            not isinstance(binding[k], str) or not binding[k]
            for k in ("graph", "parameters", "budget", "model")
        ):
            raise ValueError("empty material binding")
        encoded = json.dumps(binding, sort_keys=True, allow_nan=False)
        try:
            with self._db() as db:
                db.execute(
                    "INSERT INTO requests VALUES(?,?,?,?,?)",
                    (identity, encoded, "reserved", time.time(), None),
                )
        except sqlite3.IntegrityError as error:
            raise FileExistsError("request/material namespace was already reserved") from error
        return Ticket(self, identity)

    def read(self, identity):
        with self._db() as db:
            row = db.execute(
                "SELECT binding,state,reason FROM requests WHERE id=?", (_identity(identity),)
            ).fetchone()
        if row is None:
            raise KeyError(identity)
        return dict(binding=json.loads(row[0]), state=row[1], reason=row[2])

    def transition(self, identity, previous, target, reason=None):
        with self._db() as db:
            cursor = db.execute(
                "UPDATE requests SET state=?,updated=?,reason=? WHERE id=? AND state=?",
                (target, time.time(), reason, identity, previous),
            )
            if cursor.rowcount != 1:
                raise ValueError("invalid or concurrent lifecycle transition")


class Ticket:
    def __init__(self, store, identity):
        self.store, self.identity = store, identity

    def prepared(self, model_receipt, data_receipt):
        binding = self.store.read(self.identity)["binding"]
        for rank, receipt in enumerate((model_receipt, data_receipt)):
            if (
                receipt.get("rank") != rank
                or receipt.get("complete") is not True
                or receipt.get("binding") != binding
            ):
                raise ValueError("both matching complete backend receipts required")
        self.store.transition(self.identity, "reserved", "prepared")

    def begin_online(self):
        self.store.transition(self.identity, "prepared", "online")

    def retire(self):
        self.store.transition(self.identity, "online", "retired")

    def abandon(self, reason):
        state = self.store.read(self.identity)["state"]
        if state not in ("reserved", "prepared", "online"):
            raise ValueError("terminal material state cannot be reused")
        self.store.transition(self.identity, state, "abandoned", str(reason))
