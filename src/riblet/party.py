import gc, json, time
from pathlib import Path
import numpy as np
from .transport import encode_share, decode_share, require_owner
from .config import validate_parties, model_spec
from .packages import load_public, load_model_package, require_graph_binding
from .assets import verify_model, digest, file_hash
from .coefficient_shares import split
from .vit_parameters import slot_indices
from .resources import bind_budget, require_headroom
from .lifecycle import RequestLedger, exclusive_host
from .measurement import LifecycleSession


def validate_owner_arguments(rank, *, model, private_parameters, client_input, preprocessor):
    if rank == 0:
        if (
            not model
            or not private_parameters
            or client_input is not None
            or preprocessor is not None
        ):
            raise PermissionError("model0 accepts only owned model/parameter assets")
    elif rank == 1:
        if (
            model is not None
            or private_parameters is not None
            or not client_input
            or not preprocessor
        ):
            raise PermissionError("data1 accepts only owned input and public preprocessing assets")
    else:
        raise PermissionError("fixed model0/data1 roles required")


def validate_graph_identity(graph):
    spec = model_spec(graph["model"])
    expected = dict(
        method="riblet-full",
        family=spec["family"],
        layout=spec["layout"],
        model_rank=0,
        data_rank=1,
    )
    if graph.get("OPS") or "ops_package_id" in graph:
        raise ValueError("incompatible graph format")
    if spec["family"] == "bert":
        expected["input_dtype"] = "int64"
    if any(type(graph.get(k)) is not type(v) or graph.get(k) != v for k, v in expected.items()):
        raise ValueError(
            "graph must bind the supported full model, fixed roles, layout and graph policy"
        )


from .config import runtime_config


def raw_fixed_share(words):
    from spu import libspu, spu_pb2 as pb

    value = np.asarray(words)
    if value.dtype != np.uint64:
        raise ValueError("encoded uint64 arithmetic shares required")
    meta = pb.ValueMetaProto(
        data_type=pb.DT_F32, visibility=pb.VIS_SECRET, storage_type="cheetah.AShr<FM64>"
    )
    meta.shape.dims.extend(value.shape)
    raw = np.ascontiguousarray(value, dtype="<u8").tobytes()
    chunk = pb.ValueChunkProto(total_bytes=len(raw), chunk_offset=0, content=raw)
    share = libspu.Share()
    share.meta = meta.SerializeToString()
    share.share_chunks = [chunk.SerializeToString()]
    return share


class Party:
    def __init__(self, rank, parties, session, config=None, timeout_ms=3600000):
        from spu import api, libspu, spu_pb2 as pb

        self.pb, self.backend, self.rank = pb, libspu, rank
        if rank not in (0, 1):
            raise ValueError("rank")
        endpoints = validate_parties(parties)
        if type(timeout_ms) is not int or not 1000 <= timeout_ms <= 3600000:
            raise ValueError("bounded positive link timeout required")
        desc = libspu.link.Desc()
        desc.id = session
        desc.recv_timeout_ms = timeout_ms
        desc.http_timeout_ms = timeout_ms
        desc.throttle_window_size = 16
        desc.add_party("model", endpoints[0])
        desc.add_party("data", endpoints[1])
        self.link = libspu.link.create_brpc(desc, rank)
        self.config = config or runtime_config()
        self.runtime = api.Runtime(self.link, self.config)
        self.io = api.Io(2, self.config)

    def sent(self):
        return int(self.link.get_stats()["sent_bytes"])

    def exchange(self, value):
        raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        if self.rank == 0:
            self.link.send(1, raw)
            other = self.link.recv(1)
        else:
            other = self.link.recv(0)
            self.link.send(0, raw)
        return json.loads(bytes(other))

    def place(self, name, value, owner):
        if self.rank == owner:
            shares = self.io.make_shares(np.asarray(value), self.pb.VIS_SECRET, owner)
            self.runtime.set_var(name, shares[owner])
            self.link.send(1 - owner, encode_share(shares[1 - owner]))
        else:
            self.runtime.set_var(name, decode_share(self.link.recv(owner)))

    def public(self, name, value):
        self.runtime.set_var(
            name, self.io.make_shares(np.asarray(value), self.pb.VIS_PUBLIC)[self.rank]
        )

    def execute(self, executable):
        self.runtime.run(executable)

    def public_value(self, name):
        share = self.runtime.get_var(name)
        meta = self.pb.ValueMetaProto()
        meta.ParseFromString(share.meta)
        if meta.visibility != self.pb.VIS_PUBLIC:
            raise PermissionError("only existing public values may be read on both parties")
        return np.asarray(self.io.reconstruct([share, share]))

    def reveal_to_data(self, name):
        share = self.runtime.get_var(name)
        if self.rank == 0:
            self.link.send(1, encode_share(share))
            return None
        return np.asarray(self.io.reconstruct([decode_share(self.link.recv(0)), share]))

    def custom(self, target, value, output_shape):
        array = np.asarray(value, np.int64)
        self.public("__control", array)
        arg = "tensor<" + "x".join(map(str, array.shape)) + "xi64>"
        result = "tensor<" + "x".join(map(str, output_shape)) + "xi64>"
        code = (
            f'module attributes {{pphlo.version="{self.backend._get_version()}"}} {{func.func @main(%q:{arg})->{result} {{%s=pphlo.custom_call @"{target}"(%q):({arg})->{result} return %s:{result}'
            + "}}"
        )
        executable = self.pb.ExecutableProto(
            name="control", input_names=["__control"], output_names=["__status"], code=code.encode()
        )
        self.execute(executable)
        return self.public_value("__status")

    def material_prepare(self, family, quotas, epoch):
        if family == "bert":
            return self.custom("cdi_material_prepare:" + str(epoch), quotas, (1,))
        return self.custom("ours_vit_material_prepare_v2", quotas, (12,))

    def material_retire(self, family, epoch):
        if family == "bert":
            return self.custom(f"cdi_scope:materials:{epoch}:release", [0], (1,))
        return self.custom("ours_vit_material_retire_v1", [0], (12,))

    def release_runtime(self):
        if self.runtime is not None:
            self.runtime = None
            gc.collect()

    def close(self):
        self.release_runtime()
        self.link.stop_link()


def _model_values(rank, graph, model_path, parameter_package, preprocessor=None):
    require_owner(rank, "model")
    package = load_model_package(parameter_package, rank=rank)
    verify_model(
        graph["model"], model_path, bound=package["model_files"], preprocessor=preprocessor
    )
    require_graph_binding(graph, package)
    import jax
    from flax.serialization import msgpack_restore

    params = msgpack_restore((Path(model_path) / "flax_model.msgpack").read_bytes())
    leaves, _ = jax.tree_util.tree_flatten(params)
    if len(leaves) != len(graph["model_schema"]):
        raise ValueError("model parameter leaf count changed")
    values = {row["name"]: v for row, v in zip(graph["model_schema"], leaves)}
    kernels = []
    for path in graph["kernel_paths"]:
        value = params
        for key in path.split("/"):
            value = value[key]
        kernels.append(value["kernel"])
    for row, indices in zip(graph["weight_schema"], graph["groups"]):
        values[row["name"]] = np.concatenate([kernels[i] for i in indices], axis=1)
    if graph["family"] == "bert":
        values[graph["weight_schema"][-1]["name"]] = params["bert"]["embeddings"][
            "word_embeddings"
        ]["embedding"]
        from .bert_tables import select, coefficient_limbs
        doc=json.loads((Path(parameter_package)/'private/private-sites.json').read_bytes())
        _,rows=select(doc,graph['model'][5:])
        values['coeff_hi'],values['coeff_lo']=coefficient_limbs(rows)
    for row in graph["model_schema"] + graph["weight_schema"]:
        if (
            list(np.asarray(values[row["name"]]).shape) != row["shape"]
            or str(np.asarray(values[row["name"]]).dtype) != row["dtype"]
        ):
            raise ValueError("model schema differs from the public graph")
    return values


def _available():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("memory census unavailable")


def initialize_session(life, factory):
    holder = {}

    def create():
        try:
            holder["party"] = factory()
        except BaseException:
            holder["failed"] = True
            raise
        return holder["party"]

    def sent():
        if holder.get("failed"):
            raise RuntimeError("failed initialization traffic unavailable")
        return holder["party"].sent() if "party" in holder else 0

    begin = time.monotonic_ns()
    value = life.observe("session-initialization", "R-S", create, sent_snapshot=sent)
    life.initialization("protocol-session", created=True, lifetime="R-S")
    return value, dict(
        phase="setup",
        scope="session-initialization",
        seconds=(time.monotonic_ns() - begin) / 1e9,
        sent_bytes=value.sent(),
    )


def phase_total(begin_ns, before_sent, party):
    sent = party.sent() - before_sent
    if sent < 0:
        raise ValueError("phase sender counter reset")
    return dict(
        seconds=(time.monotonic_ns() - begin_ns) / 1e9,
        sent_bytes=sent,
        basis="continuous-local-phase-including-handshakes",
        add_to_nested_scopes=False,
    )


@exclusive_host
def run_full(
    *,
    rank,
    parties,
    graph_dir,
    public_parameters,
    state_dir,
    result_dir,
    model=None,
    private_parameters=None,
    client_input=None,
    preprocessor=None,
    network="unshaped",
    request_alias=None,
    model_preprocessor=None,
):
    validate_owner_arguments(
        rank,
        model=model,
        private_parameters=private_parameters,
        client_input=client_input,
        preprocessor=preprocessor,
    )
    if rank == 1 and model_preprocessor is not None:
        raise PermissionError("data party cannot load model preprocessing assets")
    from spu import libspu, spu_pb2 as pb
    from .inputs import load_input, save_output

    graph_dir = Path(graph_dir)
    graph = json.loads((graph_dir / "GRAPH.json").read_text())
    public = load_public(public_parameters)
    validate_graph_identity(graph)
    require_graph_binding(graph, public)
    from .resources import require_schedule

    require_schedule(graph["model"], public["schedule"])
    config = runtime_config()
    if (graph_dir / "config.pb").read_bytes() != config.SerializeToString():
        raise ValueError("runtime numerical/resource configuration changed")
    exes = _read_programs(graph_dir, graph, pb)
    runtime_sha = file_hash(libspu.__file__)
    budget = bind_budget(graph, runtime_sha)
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=False)
    execution = (
        "q-"
        + digest(
            dict(graphs=graph["graphs"], epoch=graph["request_epoch"], package=graph["package_id"])
        )[:32]
    )
    binding, peer_binding = request_bindings(graph, budget, network, request_alias)
    ticket = RequestLedger(state_dir).reserve(execution, binding)
    data = {}
    values = {}
    aux = {}
    party = None
    armed = False
    started = time.monotonic()
    try:
        if rank == 1:
            data, input_meta = load_input(
                rank, graph["model"], client_input, preprocessor, bound=public["preprocessor_files"]
            )
            if (
                graph["family"] == "bert"
                and input_meta["effective_length"] != graph["effective_length"]
            ):
                raise ValueError("input public length differs from prepared graph")
            aux["input_prepare_seconds"] = time.monotonic() - started
        else:
            values = _model_values(rank, graph, model, private_parameters, model_preprocessor)
            aux["model_load_seconds"] = time.monotonic() - started
    except BaseException as error:
        ticket.abandon(type(error).__name__)
        (result_dir / "FAILED.json").write_text(
            json.dumps(
                dict(
                    status="failed",
                    phase="owned-asset-loading",
                    error_type=type(error).__name__,
                    message=str(error),
                    secure_computation_started=False,
                ),
                indent=2,
            )
        )
        raise
    identity = dict(
        model=graph["model"],
        method="Riblet",
        implementation="riblet-" + __import__("riblet").__version__,
        configuration=digest(graph["graphs"]),
        request_id=execution,
        request_alias=request_alias,
        session_id="model-" + str(graph["model_epoch"]),
        layout=graph["layout"],
        network=network,
        parameter=graph["package_id"],
        profile=public.get("profile", "paper"),
        parameter_source=public["kind"],
    )
    phases = []
    online = None
    output = None
    phase_totals = {}
    with LifecycleSession(
        identity, directory=result_dir, rank=rank, kind="validation", backend=libspu
    ) as life:
        try:
            setup_begin = time.monotonic_ns()
            party, initialization = initialize_session(
                life, lambda: Party(rank, parties, execution, config)
            )
            phases.append(initialization)
            if party.exchange(peer_binding) != peer_binding:
                raise ValueError("peer graph/parameter/runtime/budget identity differs")

            def phase(name, category, operation):
                party.link.barrier()
                before = party.sent()
                begin = time.monotonic_ns()
                value = life.observe(name, category, operation, sent_snapshot=party.sent)
                phases.append(
                    dict(
                        phase=life.phase,
                        scope=name,
                        seconds=(time.monotonic_ns() - begin) / 1e9,
                        sent_bytes=party.sent() - before,
                    )
                )
                party.link.barrier()
                return value

            coeff = (values.pop("coeff_hi"), values.pop("coeff_lo")) if rank == 0 and graph["family"] == "bert" else (None,None)

            def placement():
                for row in graph["model_schema"] + graph["weight_schema"]:
                    party.place(row["name"], values.get(row["name"]), 0)
                if graph["family"] == "vit":
                    if rank == 0:
                        encoded = np.load(
                            Path(private_parameters) / "private/coefficients-encoded-private.npy",
                            allow_pickle=False,
                        )
                        own, peer = split(encoded, owner_rank=0)
                        party.link.send(1, np.ascontiguousarray(peer, dtype="<u8").tobytes())
                    else:
                        raw = bytes(party.link.recv(0))
                        if len(raw) != 5434 * 2 * 8:
                            raise ValueError("coefficient share payload size")
                        own = np.frombuffer(raw, dtype="<u8").copy().reshape(5434, 2)
                    for p, name in enumerate(graph["coefficients"]):
                        party.runtime.set_var(name, raw_fixed_share(own[slot_indices(p)]))

            phase("static-model-and-parameter-placement", "R-M", placement)
            values.clear()
            gc.collect()
            import ctypes

            ctypes.CDLL("libc.so.6").malloc_trim(0)
            phase("MJV-model-setup", "R-M", lambda: party.execute(exes["setup"]))
            life.initialization("model-context", created=True, lifetime="R-M")
            if graph["family"] == "bert":
                party.public("anchor", [0])
                phase("public-linear-registration", "A", lambda: party.execute(exes["register"]))
            phase_totals["setup"] = phase_total(setup_begin, 0, party)
            prep_sent = party.sent()
            prep_begin = time.monotonic_ns()
            life.start_request(execution, material_epoch=str(graph["request_epoch"]))
            phase("MJV-request-correlation", "F", lambda: party.execute(exes["preprocess"]))
            gc.collect()
            available = _available()
            peer = party.exchange({"available_bytes": available})
            require_headroom([available, int(peer["available_bytes"])], budget)
            before_stock = phase(
                "complete-fresh-inventory",
                "F",
                lambda: party.material_prepare(
                    graph["family"], budget["quotas"], graph["request_epoch"]
                ),
            )
            armed = True
            life.material(execution, "prepared")
            receipt = {"rank": rank, "complete": True, "binding": binding}
            other = party.exchange(receipt)
            ticket.prepared(receipt if rank == 0 else other, other if rank == 0 else receipt)
            ticket.begin_online()
            phase_totals["preprocessing"] = phase_total(prep_begin, prep_sent, party)
            life.mark_phase("online")
            before = party.sent()
            begin = time.monotonic_ns()
            party.link.barrier()
            names = (
                ["input_ids", "attention_mask", "token_type_ids"]
                if graph["family"] == "bert"
                else ["image"]
            )
            for name in names:
                party.place(name, data.get(name), 1)
            if graph["family"] == "bert":
                for name,value in zip(("coeff_hi","coeff_lo"),coeff):party.place(name,value,0)
            party.execute(exes["request"])
            output = party.reveal_to_data("logits")
            elapsed = (time.monotonic_ns() - begin) / 1e9
            online = dict(
                seconds=elapsed,
                sent_bytes=party.sent() - before,
                clock="local-monotonic-continuous",
            )
            life.material(execution, "consumed")
            life.mark_phase("retirement")
            party.link.barrier()
            audit = party.public_value("audit").astype(np.int64).tolist()
            retired = phase(
                "fresh-inventory-retirement",
                "A",
                lambda: party.material_retire(graph["family"], graph["request_epoch"]),
            )
            armed = False
            life.material(execution, "retired")
            ticket.retire()
            phase("runtime-release", "A", party.release_runtime)
            close_started = time.monotonic_ns()
            close_before = party.sent()
            life.observe("session-close", "A", party.close)
            try:
                close_sent = party.sent() - close_before
            except Exception:
                close_sent = None
            phases.append(
                dict(
                    phase="retirement",
                    scope="session-close",
                    seconds=(time.monotonic_ns() - close_started) / 1e9,
                    sent_bytes=close_sent,
                )
            )
            party = None
            quality = (
                save_output(rank, graph["model"], output, result_dir / "client-output")
                if rank == 1
                else None
            )
            record = dict(
                status="complete",
                kind="inference",
                rank=rank,
                identity=identity,
                online_latency_seconds=online["seconds"] if rank == 1 else None,
                participant_online=online,
                phases=phases,
                phase_totals=phase_totals,
                owned_auxiliary=aux,
                auxiliary_cross_party_additive=False,
                budget=budget,
                before_stock=np.asarray(before_stock).tolist(),
                retired_stock=np.asarray(retired).tolist(),
                audit=audit,
                output=quality,
                plaintext_reexecution=False,
                actual_setup_requests=1,
                setup_amortization_claimed=False,
                observer_data_excluded_from_online=True,
            )
            (result_dir / "result.json").write_text(json.dumps(record, indent=2))
            return record
        except BaseException as error:
            try:
                ticket.abandon(type(error).__name__)
            except ValueError:
                pass
            if armed and party is not None:
                try:
                    party.material_retire(graph["family"], graph["request_epoch"])
                except Exception:
                    pass
            (result_dir / "FAILED.json").write_text(
                json.dumps(
                    dict(
                        status="failed",
                        error_type=type(error).__name__,
                        message=str(error),
                        phases=phases,
                        completed_phase_totals=phase_totals,
                        online=online,
                    ),
                    indent=2,
                )
            )
            raise
        finally:
            if party is not None:
                try:
                    party.close()
                except Exception:
                    pass


def _read_programs(graph_dir, graph, pb):
    expected = {"setup", "preprocess", "request"} | (
        {"register"} if graph["family"] == "bert" else set()
    )
    if set(graph["graphs"]) != expected:
        raise ValueError("incomplete full-inference programs")
    exes = {}
    for name, wanted in graph["graphs"].items():
        path = graph_dir / (name + ".exec")
        if file_hash(path) != wanted:
            raise ValueError("compiled program changed")
        exe = pb.ExecutableProto()
        exe.ParseFromString(path.read_bytes())
        exes[name] = exe
    return exes


def request_bindings(graph, budget, network, alias):
    material = dict(
        graph=digest(graph["graphs"]),
        parameters=graph["package_id"],
        budget=budget["binding_sha256"],
        model=graph["model"],
        roles=[0, 1],
    )
    return material, dict(material, network=network, request_alias=alias)
