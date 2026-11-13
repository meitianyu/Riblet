import unittest

import jax
import jax.numpy as jnp

import mjv_linear_impl as linear


class MJVLinearIntrinsicTest(unittest.TestCase):
    def test_serializes_scope_and_consumer_preprocess_targets(self):
        manifest = "a" * 64
        self.assertEqual(
            linear.linear_begin_target(7, manifest, "ours"),
            f"mjv_linear_begin:ours:7:{manifest}:0:0:0",
        )
        self.assertEqual(
            linear.linear_begin_target(
                7,
                manifest,
                "ours",
                common_fallback_prefix_calls=1,
                allowed_online_dot_ole_calls=1,
                allowed_online_batch_dot_ole_calls=0,
            ),
            f"mjv_linear_begin:ours:7:{manifest}:1:1:0",
        )
        self.assertEqual(linear.linear_end_target(7), "mjv_linear_end:7")
        self.assertEqual(
            linear.consumer_preprocess_target(
                manifest,
                7,
                "attention.17.dot.17",
                (2, 4, 8),
                (2, 8, 5),
            ),
            (
                f"mjv_consumer_preprocess:{manifest}:7:attention.17.dot.17:"
                "b:2x4x8:2x8x5"
            ),
        )
        self.assertEqual(
            linear.consumer_preprocess_target(
                manifest,
                7,
                "attention.17.dot.17",
                (2, 4, 8),
                (2, 8, 5),
                model_epoch=70,
                physical_family_id="attention.family.17",
                logical_object_id="request.attention.17",
                logical_members=1,
                object_lifetime="request-immutable",
                object_operand="lhs",
                family_kind="attention_matmul",
            ),
            (
                f"mjv_consumer_preprocess:{manifest}:7:attention.17.dot.17:"
                "b:2x4x8:2x8x5:70:attention.family.17:"
                "request.attention.17:1:r:l:attention_matmul"
            ),
        )

    def test_rejects_invalid_ids_manifest_and_shapes(self):
        with self.assertRaisesRegex(AssertionError, "request_id"):
            linear.mjv_linear_begin(-1, "a" * 64)
        with self.assertRaisesRegex(AssertionError, "manifest"):
            linear.mjv_linear_begin(1, "")
        with self.assertRaisesRegex(AssertionError, "common_fallback"):
            linear.linear_begin_target(
                1,
                "a" * 64,
                common_fallback_prefix_calls=-1,
            )
        with self.assertRaisesRegex(AssertionError, "rank"):
            linear.mjv_consumer_preprocess(
                plan_sha256="a" * 64,
                request_epoch=1,
                consumer_id="attention.0",
                lhs_shape=(2,),
                rhs_shape=(2,),
            )
        with self.assertRaisesRegex(AssertionError, "contract"):
            linear.mjv_consumer_preprocess(
                plan_sha256="a" * 64,
                request_epoch=1,
                consumer_id="attention.0",
                lhs_shape=(2, 3),
                rhs_shape=(4, 2),
            )

    def test_scope_anchor_survives_jax_tracing(self):
        manifest = "b" * 64

        def scoped(value):
            value = linear.mjv_linear_begin(
                value,
                request_id=9,
                manifest_hash=manifest,
                backend="ours",
            )
            return linear.mjv_linear_end(value, request_id=9)

        traced = jax.make_jaxpr(scoped)(jnp.ones((2, 3)))
        self.assertEqual(
            [equation.primitive.name for equation in traced.jaxpr.eqns],
            ["mjv_linear_begin", "mjv_linear_end"],
        )


if __name__ == "__main__":
    unittest.main()
