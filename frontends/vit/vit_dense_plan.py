from __future__ import annotations

from dataclasses import dataclass


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
    ordinal: int
    input_key: int
    input_shape: tuple[int, ...]
    scope_paths: tuple[str, ...]
    output_widths: tuple[int, ...]
    consumer_ordinals: tuple[int, ...]


@dataclass(frozen=True)
class DenseExecutionTrace:
    calls: tuple[DenseCall, ...]
    kernels: tuple[object, ...]


def _validate_calls(calls: tuple[DenseCall, ...]) -> None:
    scopes = set()
    for ordinal, call in enumerate(calls):
        if call.ordinal != ordinal:
            raise ValueError("Dense call ordinals must be contiguous")
        if not call.scope_path or call.scope_path in scopes:
            raise ValueError("Dense scope paths must be nonempty and unique")
        if not call.input_shape or len(call.kernel_shape) != 2:
            raise ValueError("Dense calls require tensor inputs and rank-2 kernels")
        if call.input_shape[-1] != call.kernel_shape[0]:
            raise ValueError("Dense input and kernel contracting widths differ")
        if call.output_width != call.kernel_shape[1] or call.output_width <= 0:
            raise ValueError("Dense output width differs from its kernel")
        scopes.add(call.scope_path)


def group_dense_calls(calls) -> tuple[DenseGroup, ...]:
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
            ordinal=ordinal,
            input_key=key[0],
            input_shape=key[1],
            scope_paths=tuple(builders[key]["scope_paths"]),
            output_widths=tuple(builders[key]["output_widths"]),
            consumer_ordinals=tuple(builders[key]["consumer_ordinals"]),
        )
        for ordinal, key in enumerate(order)
    )


def validate_vit_dense_schedule(calls, groups) -> dict[str, int]:
    calls = tuple(calls)
    groups = tuple(groups)
    _validate_calls(calls)
    if len(calls) != 73 or len(groups) != 49:
        raise ValueError("full ViT requires 73 Dense calls and 49 physical Dense families")
    qvk = tuple(group for group in groups if len(group.scope_paths) == 3)
    singletons = tuple(group for group in groups if len(group.scope_paths) == 1)
    if len(qvk) != 12 or len(singletons) != 37:
        raise ValueError("full ViT requires twelve Q/V/K groups and thirty-seven singletons")
    for layer, group in enumerate(qvk):
        suffixes = tuple(path.rsplit("/", 1)[-1] for path in group.scope_paths)
        if suffixes != ("query", "value", "key"):
            raise ValueError(f"layer {layer} Q/V/K order differs")
        if group.output_widths != (768, 768, 768):
            raise ValueError(f"layer {layer} Q/V/K widths differ")
    if any(len(group.scope_paths) not in (1, 3) for group in groups):
        raise ValueError("only Q/V/K direct sums may contain multiple Dense consumers")
    return {"qvk_groups": len(qvk), "singleton_groups": len(singletons)}


def combine_group_kernels(calls, kernels):
    import numpy as np

    calls = tuple(calls)
    kernels = tuple(kernels)
    if len(calls) != len(kernels):
        raise ValueError("Dense calls and kernels must align")
    combined = []
    for group in group_dense_calls(calls):
        members = []
        for ordinal in group.consumer_ordinals:
            kernel = np.asarray(kernels[ordinal])
            if tuple(kernel.shape) != calls[ordinal].kernel_shape:
                raise ValueError("Dense kernel shape changed after trace capture")
            members.append(kernel)
        combined.append(np.concatenate(members, axis=1))
    return tuple(combined)


def _scope_path(module) -> str:
    path = tuple(getattr(module.scope, "path", ()))
    if not path:
        raise ValueError("Dense module has no stable scope path")
    return "/".join(str(component) for component in path)


def capture_dense_trace(dense_class, execute_model) -> DenseExecutionTrace:
    original_call = dense_class.__call__
    calls = []
    kernels = []
    captured_inputs = []

    def input_key(value):
        for index, captured in enumerate(captured_inputs):
            if captured is value:
                return index
        captured_inputs.append(value)
        return len(captured_inputs) - 1

    def traced_call(module, inputs, *args, **kwargs):
        output = original_call(module, inputs, *args, **kwargs)
        kernel = module.variables["params"]["kernel"]
        kernel_shape = tuple(int(size) for size in kernel.shape)
        calls.append(
            DenseCall(
                ordinal=len(calls),
                scope_path=_scope_path(module),
                input_key=input_key(inputs),
                input_shape=tuple(int(size) for size in inputs.shape),
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
    validate_vit_dense_schedule(trace.calls, group_dense_calls(trace.calls))
    return trace


class DensePlanExecutor:
    def __init__(self, calls, *, handles, grouped_projection):
        self._calls = tuple(calls)
        self._groups = group_dense_calls(self._calls)
        self._handles = tuple(handles)
        if len(self._groups) != len(self._handles):
            raise ValueError("Dense group and handle counts differ")
        self._group_by_scope = {
            scope: (group, consumer)
            for group in self._groups
            for consumer, scope in enumerate(group.scope_paths)
        }
        self._projection = grouped_projection
        self._position = 0
        self._outputs = {}
        self._runtime_inputs = {}

    def call(self, scope_path, inputs):
        if self._position >= len(self._calls):
            raise ValueError("Dense execution exceeded the certified plan")
        expected = self._calls[self._position]
        if scope_path != expected.scope_path:
            raise ValueError(f"expected {expected.scope_path} before {scope_path}")
        if tuple(int(size) for size in inputs.shape) != expected.input_shape:
            raise ValueError(f"Dense input shape changed at {scope_path}")
        self._position += 1
        group, consumer = self._group_by_scope[scope_path]
        if group.ordinal not in self._outputs:
            outputs = tuple(
                self._projection(
                    inputs,
                    self._handles[group.ordinal],
                    output_widths=group.output_widths,
                )
            )
            if len(outputs) != len(group.scope_paths):
                raise ValueError("grouped Dense projection returned the wrong width")
            self._outputs[group.ordinal] = outputs
            self._runtime_inputs[group.ordinal] = id(inputs)
        elif self._runtime_inputs[group.ordinal] != id(inputs):
            raise ValueError(f"shared Dense input identity changed at {scope_path}")
        return self._outputs[group.ordinal][consumer]

    def finish(self):
        if self._position != len(self._calls):
            raise ValueError("Dense execution stopped before consuming the plan")
        if len(self._outputs) != len(self._groups):
            raise ValueError("Dense execution did not invoke every family")


def hijack_dense_plan(dense_class, executor):
    from contextlib import contextmanager

    @contextmanager
    def context():
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

    return context()
