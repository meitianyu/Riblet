from concurrent.futures import Future
import json
import os
import sys
from pathlib import Path
import time
import uuid


class LifecycleSession:
    def __init__(self, identity, *, directory, rank, kind, backend=None):
        if not isinstance(identity, dict) or any(
            not isinstance(k, str) or not isinstance(v, (str, int, float, bool, type(None)))
            for k, v in identity.items()
        ):
            raise ValueError("only public scalar identity metadata is accepted")
        for key in (
            "model",
            "method",
            "implementation",
            "configuration",
            "request_id",
            "session_id",
            "layout",
            "network",
            "parameter",
        ):
            if not identity.get(key):
                raise ValueError("missing public lifecycle identity: " + key)
        if rank not in (0, 1) or kind not in (
            "formal",
            "warm-up",
            "profiling",
            "diagnostic",
            "validation",
        ):
            raise ValueError("actual party and run kind required")
        self.identity = dict(identity)
        self.rank, self.kind, self.backend = rank, kind, backend
        self.phase, self.online_entered = "setup", False
        self.events, self.initializations, self.materials = [], [], {}
        self.active = 0
        self._stack = []
        self.requests = []
        self.auxiliary_seconds = {}
        self.request_ref, self.material_epoch = None, None
        self.status = "running"
        self.phase_errors = []
        self.directory = Path(directory)
        if not self.directory.is_dir():
            raise ValueError("owned existing result directory required")
        self.output_path = self.directory / ("lifecycle-party%d.json" % rank)
        if self.output_path.exists():
            raise FileExistsError("lifecycle result already exists")

    def __enter__(self):
        if self.backend is not None:
            self.backend.lifecycle_begin(
                json.dumps(self.identity, sort_keys=True), self.rank, self.kind
            )
        return self

    def mark_phase(self, phase):
        allowed = {
            "setup": {"preprocessing", "online", "closed", "failed"},
            "preprocessing": {"online", "closed", "failed"},
            "online": {"retirement", "failed"},
            "retirement": {"closed", "failed"},
            "failed": {"closed"},
            "closed": set(),
        }
        if self.active or phase not in allowed[self.phase]:
            raise ValueError("invalid actual lifecycle phase transition")
        if self.backend is not None:
            self.backend.lifecycle_mark_phase(phase)
        self.phase = phase
        self.online_entered |= phase == "online"

    def start_request(self, request_id, *, material_epoch):
        if (
            self.active
            or self.phase not in ("setup", "preprocessing", "retirement")
            or not isinstance(request_id, str)
            or not request_id
            or not isinstance(material_epoch, str)
            or not material_epoch
            or any(r["request_id"] == request_id for r in self.requests)
            or any(v not in ("retired", "abandoned") for v in self.materials.values())
        ):
            raise ValueError("new request needs distinct identity and no live materials")
        if self.backend is not None:
            self.backend.lifecycle_mark_request(request_id, material_epoch)
        self.request_ref, self.material_epoch = request_id, material_epoch
        self.requests.append(dict(request_id=request_id, material_epoch=material_epoch))
        self.phase = "preprocessing"

    def observe(self, scope, category, operation, *, sent_snapshot=None):
        if self.phase not in ("setup", "preprocessing", "retirement"):
            raise ValueError("preparation scope cannot observe Online or closed phase")
        if category not in ("R-M", "R-S", "F", "A", "unclassified"):
            raise ValueError("unknown preparation category")
        parent = self._stack[-1] if self._stack else None
        row = dict(
            event_id=len(self.events) + 1,
            parent_event=parent,
            scope=scope,
            category=category,
            original_phase=self.phase,
            status="failed",
            measurement_status="complete",
            local_sent_bytes=None,
            bytes_basis="actual-own-sent-inclusive",
            time_additive=False,
            bytes_additive=False,
            request_ref=self.request_ref,
            material_epoch=self.material_epoch,
        )
        self.events.append(row)
        self._stack.append(row["event_id"])
        before = None
        if sent_snapshot is not None:
            try:
                value = sent_snapshot()
                before = (
                    [int(v) for v in value]
                    if isinstance(value, (list, tuple)) and len(value) == 2
                    else int(value)
                )
            except Exception:
                row["measurement_status"] = "counter-unavailable"
        started = time.monotonic_ns()
        self.active += 1
        primary = None
        try:
            value = operation()
            if isinstance(value, Future):
                value = value.result()
            row["status"] = "complete"
            return value
        except BaseException as error:
            primary = error
            raise
        finally:
            row["local_elapsed_ns"] = time.monotonic_ns() - started
            self.active -= 1
            self._stack.pop()
            if sent_snapshot is not None:
                try:
                    value = sent_snapshot()
                    after = (
                        [int(v) for v in value]
                        if isinstance(value, (list, tuple)) and len(value) == 2
                        else int(value)
                    )
                    if before is not None:
                        reset = (
                            any(b < a for a, b in zip(before, after))
                            if isinstance(before, list) and isinstance(after, list)
                            else after < before
                        )
                        if reset:
                            row["measurement_status"] = "counter-reset"
                            if primary is None:
                                raise ValueError("lifecycle sender counter reset")
                        else:
                            if isinstance(before, list):
                                row["party_sent_bytes"] = [b - a for a, b in zip(before, after)]
                                row["bytes_basis"] = "actual-two-senders-inclusive"
                            else:
                                row["local_sent_bytes"] = after - before
                except ValueError:
                    if primary is None:
                        raise
                except Exception:
                    row["measurement_status"] = "counter-unavailable"

    def material(self, material_id, event):
        previous = self.materials.get(material_id)
        if (
            event
            not in {
                None: {"prepared"},
                "prepared": {"consumed", "abandoned"},
                "consumed": {"retired"},
                "retired": set(),
                "abandoned": set(),
            }[previous]
        ):
            raise ValueError("material lifecycle reuse or impossible transition")
        self.materials[material_id] = event

    def initialization(self, name, *, created, lifetime):
        if lifetime not in ("R-M", "R-S"):
            raise ValueError("persistent initialization lifetime required")
        self.initializations.append(
            dict(
                name=name,
                category=lifetime,
                event="initialized" if created else "cache-hit",
                original_phase=self.phase,
            )
        )

    def snapshot(self):
        backend = None
        if self.backend is not None:
            try:
                backend = json.loads(self.backend.lifecycle_snapshot())
            except Exception as error:
                backend = dict(status="unavailable", error_type=type(error).__name__)
        return dict(
            schema="lifecycle-fine-v1",
            identity=dict(self.identity),
            rank=self.rank,
            kind=self.kind,
            phase=self.phase,
            status=self.status,
            online_entered=self.online_entered,
            python_events=list(self.events),
            generator_events=backend,
            initializations=list(self.initializations),
            materials=dict(self.materials),
            retained_model_values=False,
            online_timer_changed=False,
            requests=list(self.requests),
            existing_auxiliary_seconds=dict(self.auxiliary_seconds),
            auxiliary_additive=False,
            phase_errors=list(self.phase_errors),
        )

    def export(self):
        if self.active:
            raise ValueError("cannot export unfinished work")
        payload = json.dumps(self.snapshot(), ensure_ascii=False, allow_nan=False, indent=2)
        temporary = self.directory / (".lifecycle-" + uuid.uuid4().hex + ".tmp")
        try:
            with temporary.open("x", encoding="utf8") as stream:
                stream.write(payload)
                stream.flush()

            os.link(temporary, self.output_path)
        finally:
            if temporary.exists():
                temporary.unlink()
        return self.output_path

    def __exit__(self, exc_type, exc, traceback):
        self.status = "failed" if exc_type is not None else "complete"
        phase_error = None
        try:
            if exc_type is not None and self.phase not in ("failed", "closed"):
                self.mark_phase("failed")
            if self.phase != "closed":
                self.mark_phase("closed")
        except Exception as error:
            phase_error = error
            self.status = "failed"
            self.phase_errors.append(dict(phase=self.phase, error_type=type(error).__name__))
        try:
            self.export()
        except Exception as export_error:
            print(f"Warning: lifecycle record not saved to {self.output_path}: {export_error}. "
                  "Protocol outcome is unchanged; lifecycle measurements are unavailable.",
                  file=sys.stderr)
        if phase_error is not None and exc_type is None:
            raise phase_error
        return False
