from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, replace
import math


@dataclass(frozen=True)
class DenseCall:
    ordinal: int
    scope_path: str
    input_key: int
    input_shape: tuple[int, ...]
    kernel_shape: tuple[int, int]
    output_width: int


@dataclass(frozen=True)
class DenseGroup:
    group_ordinal: int
    input_key: int
    input_shape: tuple[int, ...]
    scope_paths: tuple[str, ...]
    output_widths: tuple[int, ...]
    consumer_ordinals: tuple[int, ...]


@dataclass(frozen=True)
class DenseExecutionTrace:
    calls: tuple[DenseCall, ...]
    kernels: tuple[object, ...]


def _validate_calls(calls):
    seen_scopes = set()
    for expected_ordinal, call in enumerate(calls):
        if call.ordinal != expected_ordinal:
            raise ValueError("Dense call ordinals must be contiguous and trace ordered")
        if not call.scope_path or call.scope_path in seen_scopes:
            raise ValueError(
                f"Dense scope {call.scope_path!r} must appear exactly once"
            )
        if not call.input_shape or len(call.kernel_shape) != 2:
            raise ValueError("Dense calls require tensor inputs and rank-2 kernels")
        if call.kernel_shape[0] != call.input_shape[-1]:
            raise ValueError("Dense input width does not match kernel height")
        if call.output_width != call.kernel_shape[1] or call.output_width <= 0:
            raise ValueError("Dense output width does not match kernel width")
        seen_scopes.add(call.scope_path)


def group_dense_calls(calls):
    calls = tuple(calls)
    _validate_calls(calls)
    builders = {}
    order = []
    for call in calls:
        key = (call.input_key, call.input_shape)
        if key not in builders:
            builders[key] = {
                "scope_paths": [],
                "output_widths": [],
                "consumer_ordinals": [],
            }
            order.append(key)
        builders[key]["scope_paths"].append(call.scope_path)
        builders[key]["output_widths"].append(call.output_width)
        builders[key]["consumer_ordinals"].append(call.ordinal)

    return tuple(
        DenseGroup(
            group_ordinal=group_ordinal,
            input_key=input_key,
            input_shape=input_shape,
            scope_paths=tuple(builders[key]["scope_paths"]),
            output_widths=tuple(builders[key]["output_widths"]),
            consumer_ordinals=tuple(builders[key]["consumer_ordinals"]),
        )
        for group_ordinal, key in enumerate(order)
        for input_key, input_shape in (key,)
    )


class DensePlanCursor:
    def __init__(self, calls):
        self._calls = tuple(calls)
        _validate_calls(self._calls)
        self._position = 0

    @property
    def next_call(self):
        if self._position >= len(self._calls):
            return None
        return self._calls[self._position]

    def consume(self, scope_path, input_key, input_shape):
        if self._position >= len(self._calls):
            raise ValueError("Dense execution plan was already consumed")
        expected = self._calls[self._position]
        if scope_path != expected.scope_path:
            raise ValueError(f"expected {expected.scope_path} before {scope_path}")
        if input_key != expected.input_key:
            raise ValueError(f"Dense input identity changed at {scope_path}")
        if tuple(input_shape) != expected.input_shape:
            raise ValueError(f"Dense input shape changed at {scope_path}")
        self._position += 1
        return expected

    def finish(self):
        if self._position != len(self._calls):
            expected = self._calls[self._position]
            raise ValueError(f"Dense execution stopped before {expected.scope_path}")


class DensePlanExecutor:
    def __init__(self, calls, *, handles, grouped_projection):
        self._calls = tuple(calls)
        self._groups = group_dense_calls(self._calls)
        self._cursor = DensePlanCursor(self._calls)
        self._handles = tuple(handles)
        if len(self._handles) != len(self._groups):
            raise ValueError("Dense group and handle counts must match")
        self._grouped_projection = grouped_projection
        self._scope_consumers = {
            scope_path: (group, consumer)
            for group in self._groups
            for consumer, scope_path in enumerate(group.scope_paths)
        }
        self._outputs = {}
        self._runtime_input_keys = {}

    @property
    def groups(self):
        return self._groups

    def call(self, scope_path, inputs, input_shape=None):
        if input_shape is None:
            input_shape = tuple(int(dimension) for dimension in inputs.shape)
        next_call = self._cursor.next_call
        if next_call is None:
            raise ValueError("Dense execution plan was already consumed")
        expected = self._cursor.consume(
            scope_path,
            next_call.input_key,
            input_shape,
        )
        group, consumer = self._scope_consumers[expected.scope_path]
        group_id = group.group_ordinal
        if group_id not in self._outputs:
            outputs = tuple(
                self._grouped_projection(
                    inputs,
                    self._handles[group_id],
                    output_widths=group.output_widths,
                )
            )
            if len(outputs) != len(group.scope_paths):
                raise ValueError("Dense group returned the wrong output count")
            self._outputs[group_id] = outputs
            self._runtime_input_keys[group_id] = id(inputs)
        elif self._runtime_input_keys[group_id] != id(inputs):
            raise ValueError(f"Dense shared input identity changed at {scope_path}")
        return self._outputs[group_id][consumer]

    def finish(self):
        self._cursor.finish()
        if len(self._outputs) != len(self._groups):
            raise ValueError("not every Dense group was executed")


def combine_group_kernels(calls, kernels):
    import numpy as np

    calls = tuple(calls)
    kernels = tuple(kernels)
    if len(calls) != len(kernels):
        raise ValueError("Dense call and kernel counts must match")
    groups = group_dense_calls(calls)
    combined = []
    for group in groups:
        group_kernels = []
        for ordinal in group.consumer_ordinals:
            kernel = np.asarray(kernels[ordinal])
            if tuple(kernel.shape) != calls[ordinal].kernel_shape:
                raise ValueError("Dense kernel changed after trace capture")
            group_kernels.append(kernel)
        combined.append(np.concatenate(group_kernels, axis=1))
    return tuple(combined)


def _scope_path(module):
    path = tuple(getattr(module.scope, "path", ()))
    if not path:
        raise ValueError("Dense module has no stable scope path")
    return "/".join(str(component) for component in path)


def capture_dense_trace(dense_class, execute_model):
    original_call = dense_class.__call__
    calls = []
    kernels = []
    captured_inputs = []

    def stable_input_key(value):
        for key, captured in enumerate(captured_inputs):
            if captured is value:
                return key
        captured_inputs.append(value)
        return len(captured_inputs) - 1

    def traced_call(module, inputs, *args, **kwargs):
        output = original_call(module, inputs, *args, **kwargs)
        kernel = module.variables["params"]["kernel"]
        kernel_shape = tuple(int(dimension) for dimension in kernel.shape)
        input_shape = tuple(int(dimension) for dimension in inputs.shape)
        calls.append(
            DenseCall(
                ordinal=len(calls),
                scope_path=_scope_path(module),
                input_key=stable_input_key(inputs),
                input_shape=input_shape,
                kernel_shape=kernel_shape,
                output_width=kernel_shape[1],
            )
        )
        kernels.append(kernel)
        return output

    dense_class.__call__ = traced_call
    try:
        execute_model()
    finally:
        dense_class.__call__ = original_call
    trace = DenseExecutionTrace(tuple(calls), tuple(kernels))
    _validate_calls(trace.calls)
    return trace


def capture_dense_execution(dense_class, execute_model):
    return capture_dense_trace(dense_class, execute_model).calls


def _canonical_dot_shapes(equation):
    lhs_shape = tuple(int(size) for size in equation.invars[0].aval.shape)
    rhs_shape = tuple(int(size) for size in equation.invars[1].aval.shape)
    dimensions = equation.params.get("dimension_numbers")
    if dimensions is None:
        return None
    (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = dimensions
    if len(lhs_contract) != 1 or len(rhs_contract) != 1:
        return None
    lhs_batch = tuple(lhs_batch)
    rhs_batch = tuple(rhs_batch)
    if len(lhs_batch) != len(rhs_batch):
        return None
    lhs_excluded = set(lhs_batch) | set(lhs_contract)
    rhs_excluded = set(rhs_batch) | set(rhs_contract)
    lhs_outer = tuple(
        index for index in range(len(lhs_shape)) if index not in lhs_excluded
    )
    rhs_outer = tuple(
        index for index in range(len(rhs_shape)) if index not in rhs_excluded
    )
    if not lhs_outer or not rhs_outer:
        return None
    batch = math.prod(lhs_shape[index] for index in lhs_batch)
    if batch != math.prod(rhs_shape[index] for index in rhs_batch):
        return None
    rows = math.prod(lhs_shape[index] for index in lhs_outer)
    inner = math.prod(lhs_shape[index] for index in lhs_contract)
    if inner != math.prod(rhs_shape[index] for index in rhs_contract):
        return None
    columns = math.prod(rhs_shape[index] for index in rhs_outer)
    if lhs_batch:
        return (batch, rows, inner), (batch, inner, columns), True
    return (rows, inner), (inner, columns), False


def collect_dynamic_linear_calls(closed_jaxpr):
    from linear_object_plan import DynamicLinearCall

    calls = []
    source_ordinal = 0

    def is_dynamic_integer(variable):
        dtype = getattr(getattr(variable, "aval", None), "dtype", None)
        dtype_name = getattr(dtype, "name", getattr(dtype, "__name__", str(dtype)))
        return getattr(dtype, "kind", None) in ("i", "u") or dtype_name.startswith(
            ("int", "uint")
        )

    def dependency_of(dependencies, variable):
        try:
            return bool(dependencies.get(variable, False))
        except TypeError:
            return False

    def visit(value, input_dependencies=None):
        nonlocal source_ordinal
        jaxpr = getattr(value, "jaxpr", value)
        invars = tuple(getattr(jaxpr, "invars", ()))
        if input_dependencies is None:
            input_dependencies = tuple(is_dynamic_integer(var) for var in invars)
        dependencies = {
            variable: False for variable in getattr(jaxpr, "constvars", ())
        }
        dependencies.update(zip(invars, input_dependencies))

        for equation in getattr(jaxpr, "eqns", ()):
            equation_ordinal = source_ordinal
            source_ordinal += 1
            equation_dependencies = tuple(
                dependency_of(dependencies, variable) for variable in equation.invars
            )
            primitive = getattr(equation.primitive, "name", "")
            if primitive == "gather" and len(equation.invars) >= 2:
                operand_shape = tuple(
                    int(size) for size in equation.invars[0].aval.shape
                )
                index_shape = tuple(
                    int(size) for size in equation.invars[1].aval.shape
                )
                dimensions = equation.params.get("dimension_numbers")
                slice_sizes = tuple(equation.params.get("slice_sizes", ()))
                is_private_row_gather = (
                    equation_dependencies[1]
                    and len(operand_shape) == 2
                    and bool(index_shape)
                    and index_shape[-1] == 1
                    and dimensions is not None
                    and tuple(dimensions.collapsed_slice_dims) == (0,)
                    and tuple(dimensions.start_index_map) == (0,)
                    and slice_sizes == (1, operand_shape[1])
                )
                if is_private_row_gather:
                    for row in range(math.prod(index_shape[:-1])):
                        calls.append(
                            DynamicLinearCall(
                                ordinal=len(calls),
                                source_primitive="gather",
                                source_ordinal=equation_ordinal,
                                kind="embedding_row",
                                logical_object_id="model.word_embeddings",
                                lhs_shape=(1, operand_shape[0]),
                                rhs_shape=operand_shape,
                                batched=False,
                                object_operand="rhs",
                                role=f"row.{row}",
                            )
                        )
            elif primitive == "dot_general":
                canonical = _canonical_dot_shapes(equation)
                if canonical is not None:
                    lhs_shape, rhs_shape, batched = canonical
                    calls.append(
                        DynamicLinearCall(
                            ordinal=len(calls),
                            source_primitive="dot_general",
                            source_ordinal=equation_ordinal,
                            kind="attention_matmul",
                            logical_object_id=(
                                f"request.dot_source.{equation_ordinal}.lhs"
                            ),
                            lhs_shape=lhs_shape,
                            rhs_shape=rhs_shape,
                            batched=batched,
                            object_operand="lhs",
                            role=f"dot.{equation_ordinal}",
                        )
                    )
            for parameter in equation.params.values():
                nested = (
                    parameter if isinstance(parameter, (tuple, list)) else (parameter,)
                )
                for child in nested:
                    if hasattr(child, "jaxpr") or hasattr(child, "eqns"):
                        child_jaxpr = getattr(child, "jaxpr", child)
                        child_count = len(getattr(child_jaxpr, "invars", ()))
                        visit(
                            child_jaxpr,
                            equation_dependencies[:child_count]
                            + (False,)
                            * max(0, child_count - len(equation_dependencies)),
                        )
            output_dependency = any(equation_dependencies)
            for variable in getattr(equation, "outvars", ()):
                try:
                    dependencies[variable] = output_dependency
                except TypeError:
                    pass

    visit(closed_jaxpr)
    return tuple(calls)


def select_complete_bert_dynamic_consumers(
    dense_calls,
    dynamic_calls,
    *,
    embedding_row_count=None,
    embedding_table_shape=None,
):


    dense_calls = tuple(dense_calls)
    dynamic_calls = tuple(dynamic_calls)
    has_row_count = embedding_row_count is not None
    has_table_shape = embedding_table_shape is not None
    if has_row_count != has_table_shape:
        raise ValueError("embedding source facts must be provided together")
    has_embedding_source = has_row_count and has_table_shape
    if has_embedding_source:
        embedding_row_count = int(embedding_row_count)
        embedding_table_shape = tuple(int(size) for size in embedding_table_shape)
        if embedding_row_count <= 0:
            raise ValueError("embedding row count must be positive")
        if len(embedding_table_shape) != 2:
            raise ValueError("embedding table shape must have rank two")
    expected_dense_dots = Counter(
        (
            (
                math.prod(call.input_shape[:-1]),
                call.input_shape[-1],
            ),
            tuple(call.kernel_shape),
        )
        for call in dense_calls
    )
    observed_dense_dots = Counter()
    selected = []
    embedding_rows = []
    for call in dynamic_calls:
        if call.kind == "embedding_row":
            if call.source_primitive != "gather" or call.batched:
                raise ValueError("embedding source role is inconsistent")
            selected.append(call)
            embedding_rows.append(call)
        elif call.kind == "attention_matmul":
            if call.source_primitive != "dot_general":
                raise ValueError("matrix source role is inconsistent")
            if call.batched:
                selected.append(call)
            else:
                key = (call.lhs_shape, call.rhs_shape)
                observed_dense_dots[key] += 1
        else:
            raise ValueError(f"unsupported complete-BERT dynamic role: {call.kind}")

    if has_embedding_source:
        vocabulary_size, embedding_width = embedding_table_shape
        embedding_batch = (
            (embedding_row_count, vocabulary_size),
            embedding_table_shape,
        )
        matching_physical = tuple(
            call
            for call in dynamic_calls
            if (call.lhs_shape, call.rhs_shape) == embedding_batch
        )
        if len(matching_physical) > 1:
            raise ValueError("embedding source has duplicate physical batch dots")
        if matching_physical:
            physical = matching_physical[0]
            if (
                physical.source_primitive != "dot_general"
                or physical.kind != "attention_matmul"
                or physical.batched
                or physical.object_operand != "lhs"
            ):
                raise ValueError("embedding physical batch provenance is inconsistent")

        if embedding_rows:
            expected_roles = tuple(
                f"row.{row}" for row in range(embedding_row_count)
            )
            if (
                len(embedding_rows) != embedding_row_count
                or tuple(row.role for row in embedding_rows) != expected_roles
                or any(
                    row.source_primitive != "gather"
                    or row.batched
                    or row.logical_object_id != "model.word_embeddings"
                    or row.object_operand != "rhs"
                    or row.lhs_shape != (1, vocabulary_size)
                    or row.rhs_shape != embedding_table_shape
                    for row in embedding_rows
                )
            ):
                raise ValueError(
                    "gather-derived embedding rows do not match source facts"
                )
        else:
            if len(matching_physical) != 1:
                raise ValueError(
                    "embedding source without gathers requires one physical batch dot"
                )
            physical = matching_physical[0]
            synthetic_rows = [
                replace(
                    physical,
                    ordinal=row,
                    source_primitive="source_bound_embedding",
                    kind="embedding_row",
                    logical_object_id="model.word_embeddings",
                    lhs_shape=(1, vocabulary_size),
                    rhs_shape=(vocabulary_size, embedding_width),
                    batched=False,
                    object_operand="rhs",
                    role=f"row.{row}",
                )
                for row in range(embedding_row_count)
            ]
            insertion = sum(
                call.source_ordinal < physical.source_ordinal for call in selected
            )
            selected[insertion:insertion] = synthetic_rows

        if matching_physical:
            del observed_dense_dots[embedding_batch]
    elif embedding_rows:
        first_row = embedding_rows[0]
        consistent_rows = (
            len(first_row.lhs_shape) == 2
            and first_row.lhs_shape[0] == 1
            and len(first_row.rhs_shape) == 2
            and first_row.lhs_shape[1] == first_row.rhs_shape[0]
            and all(
                row.lhs_shape == first_row.lhs_shape
                and row.rhs_shape == first_row.rhs_shape
                for row in embedding_rows
            )
        )
        if consistent_rows:
            embedding_batch = (
                (len(embedding_rows), first_row.lhs_shape[1]),
                first_row.rhs_shape,
            )
            if observed_dense_dots[embedding_batch] == 1:
                del observed_dense_dots[embedding_batch]


    if observed_dense_dots and observed_dense_dots != expected_dense_dots:
        expected_evidence = [
            (lhs_shape, rhs_shape, count)
            for (lhs_shape, rhs_shape), count in sorted(
                expected_dense_dots.items()
            )
        ]
        observed_evidence = [
            (lhs_shape, rhs_shape, count)
            for (lhs_shape, rhs_shape), count in sorted(
                observed_dense_dots.items()
            )
        ]
        raise ValueError(
            "non-batched dynamic dots do not match the captured Dense trace; "
            f"expected_dense_dots={expected_evidence!r}; "
            f"observed_nonbatched_dots={observed_evidence!r}"
        )
    return tuple(
        replace(call, ordinal=ordinal)
        for ordinal, call in enumerate(selected)
    )


def capture_linear_object_plan(dense_class, closed_jaxpr, execute_model):
    from linear_object_plan import compile_linear_object_plan

    dense_trace = capture_dense_trace(dense_class, execute_model)
    dynamic_calls = select_complete_bert_dynamic_consumers(
        dense_trace.calls,
        collect_dynamic_linear_calls(closed_jaxpr),
    )
    return compile_linear_object_plan(dense_trace.calls, dynamic_calls)


@contextmanager
def hijack_dense_plan(dense_class, executor):
    original_call = dense_class.__call__

    def planned_call(module, inputs, *args, **kwargs):
        if args or kwargs:
            raise ValueError("planned Dense calls accept only the input tensor")
        output = executor.call(_scope_path(module), inputs)
        bias = module.variables["params"].get("bias")
        return output if bias is None else output + bias

    dense_class.__call__ = planned_call
    try:
        yield executor
        executor.finish()
    finally:
        dense_class.__call__ = original_call
