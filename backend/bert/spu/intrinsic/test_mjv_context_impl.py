import importlib
import unittest

class MJVContextIntrinsicTest(unittest.TestCase):
    def _api(self):
        try:
            api = importlib.import_module("mjv_context_impl")
        except ModuleNotFoundError as exc:
            if exc.name != "mjv_context_impl":
                raise
            api = None
        self.assertIsNotNone(api, "mjv_context_impl production API is missing")
        return api

    def test_context_targets_carry_plan_epochs_and_no_per_op_identity(self):
        api = self._api()
        digest = "a" * 64
        self.assertEqual(
            api.epoch_begin_target(digest, 7, 19),
            f"mjv_epoch_begin:{digest}:7:19",
        )
        self.assertEqual(
            api.epoch_end_target(
                digest, 7, 19, release_model_epoch=True
            ),
            f"mjv_epoch_end:{digest}:7:19:release_model",
        )
        targets = (
            api.plan_register_target(digest),
            api.model_setup_batch_target(digest, 7),
            api.request_preprocess_batch_target(digest, 7, 19),
            api.epoch_begin_target(digest, 7, 19),
            api.epoch_end_target(digest, 7, 19),
            api.audit_snapshot_target(digest, 7, 19),
        )
        for target in targets:
            self.assertNotIn("op_id", target)

    def test_epoch_anchor_survives_tracing_with_plan_and_epochs(self):
        try:
            import jax
            import jax.numpy as jnp
        except ModuleNotFoundError:
            self.skipTest("JAX is validated by the Bazel/Linux target")
        api = self._api()
        digest = "b" * 64

        def scoped(value):
            value = api.mjv_epoch_begin(
                value,
                plan_sha256=digest,
                model_epoch=7,
                request_epoch=19,
            )
            audit = api.mjv_audit_snapshot(
                value,
                plan_sha256=digest,
                model_epoch=7,
                request_epoch=19,
            )
            return api.mjv_epoch_end(
                value,
                audit,
                plan_sha256=digest,
                model_epoch=7,
                request_epoch=19,
                release_model_epoch=True,
            )

        traced = jax.make_jaxpr(scoped)(jnp.ones((2, 3)))
        self.assertEqual(
            [equation.primitive.name for equation in traced.jaxpr.eqns],
            ["mjv_epoch_begin", "mjv_audit_snapshot", "mjv_epoch_end"],
        )
        self.assertEqual(
            traced.jaxpr.eqns[-1].params["target"],
            f"mjv_epoch_end:{digest}:7:19:release_model",
        )

    def test_epoch_end_depends_on_runtime_audit_snapshot(self):
        try:
            import jax
            import jax.numpy as jnp
        except ModuleNotFoundError:
            self.skipTest("JAX is validated by the Bazel/Linux target")
        api = self._api()
        metadata = {
            "plan_sha256": "d" * 64,
            "model_epoch": 7,
            "request_epoch": 19,
        }

        def ordered_scope(anchor):
            audit = api.mjv_audit_snapshot(anchor, **metadata)
            ended = api.mjv_epoch_end(anchor, audit, **metadata)
            return ended, audit

        traced = jax.make_jaxpr(ordered_scope)(
            jnp.ones((2, 3), dtype=jnp.float32)
        )
        snapshot, epoch_end = traced.jaxpr.eqns
        self.assertEqual(snapshot.primitive.name, "mjv_audit_snapshot")
        self.assertEqual(epoch_end.primitive.name, "mjv_epoch_end")
        self.assertEqual(len(epoch_end.invars), 2)
        self.assertEqual(epoch_end.invars[1], snapshot.outvars[0])

    def test_runtime_audit_has_one_versioned_fixed_width_result(self):
        try:
            import jax
            import jax.numpy as jnp
        except ModuleNotFoundError:
            self.skipTest("JAX is validated by the Bazel/Linux target")
        api = self._api()
        expected_fields = (
            "schema_version",
            "plan_matches",
            "model_epoch",
            "request_epoch",
            "logical_consumer_count",
            "physical_family_count",
            "model_required_shared_object_contexts",
            "model_shared_side_logical_openings",
            "model_shared_side_opened_elements",
            "model_shared_side_physical_collectives",
            "model_setup_right_logical_openings",
            "model_setup_right_opened_elements",
            "model_setup_right_physical_collectives",
            "request_required_shared_object_contexts",
            "request_shared_side_logical_openings",
            "request_shared_side_opened_elements",
            "request_shared_side_physical_collectives",
            "request_online_right_logical_openings",
            "request_online_right_opened_elements",
            "request_online_right_physical_collectives",
            "embedding_physical_executions",
            "model_setup_rpc_calls",
            "request_preprocess_rpc_calls",
            "consumed_logical_consumers",
            "expected_logical_consumers",
            "fresh_request_masks",
            "static_object_reused",
            "secure_invoke_rpc_calls",
            "protocol_matrix_calls",
            "dot_ole_calls",
            "batch_dot_ole_calls",
            "live_object_contexts",
            "live_consumer_contexts",
        )
        self.assertEqual(api.RUNTIME_OBJECT_AUDIT_V1_FIELDS, expected_fields)

        def snapshot(anchor):
            return api.mjv_audit_snapshot(
                anchor,
                plan_sha256="c" * 64,
                model_epoch=7,
                request_epoch=19,
            )

        traced = jax.make_jaxpr(snapshot)(jnp.ones((1,), dtype=jnp.int32))
        self.assertEqual(
            [equation.primitive.name for equation in traced.jaxpr.eqns],
            ["mjv_audit_snapshot"],
        )
        self.assertEqual(
            traced.jaxpr.outvars[0].aval.shape, (len(expected_fields),)
        )
        self.assertEqual(str(traced.jaxpr.outvars[0].aval.dtype), "int64")

    def test_rejects_invalid_hash_and_epoch(self):
        api = self._api()
        with self.assertRaisesRegex(AssertionError, "plan"):
            api.epoch_begin_target("", 1, 2)
        with self.assertRaisesRegex(AssertionError, "model_epoch"):
            api.epoch_begin_target("a" * 64, -1, 2)
        with self.assertRaisesRegex(AssertionError, "request_epoch"):
            api.epoch_begin_target("a" * 64, 1, -2)


if __name__ == "__main__":
    unittest.main()
