import unittest

import mjv_impl as mjv


class MJVRuntimeFamilyTargetTest(unittest.TestCase):
    def test_static_targets_carry_stable_object_family_and_epochs(self):
        metadata = {
            "plan_sha256": "a" * 64,
            "model_epoch": 70,
            "physical_family_id": "dense.family.0",
            "logical_object_id": "request.hidden.0",
            "logical_members": 3,
            "object_lifetime": "request-immutable",
            "object_operand": "lhs",
            "family_kind": "qkv_direct_sum",
            "shared_shape": (2, 3),
            "right_shape": (3, 6),
        }
        self.assertEqual(
            mjv.model_setup_target(
                model_id=7,
                sum_r=9,
                output_widths=(2, 2, 2),
                linear_family="matmul_aa",
                **metadata,
            ),
            (
                "mjv_model_setup:7:9:2,2,2:matmul_aa:"
                + "a" * 64
                + ":70:dense.family.0:request.hidden.0:3:r:l:"
                "qkv_direct_sum:2x3:3x6"
            ),
        )
        self.assertEqual(
            mjv.request_preprocess_target(
                model_id=7,
                request_id=19,
                rows=2,
                plan_sha256="a" * 64,
                model_epoch=70,
                request_epoch=190,
                physical_family_id="dense.family.0",
            ),
            (
                "mjv_request_preprocess:7:19:2:"
                + "a" * 64
                + ":70:190:dense.family.0"
            ),
        )

    def test_rejects_partial_or_append_only_runtime_metadata(self):
        with self.assertRaisesRegex(AssertionError, "complete"):
            mjv.model_setup_target(
                model_id=7,
                sum_r=3,
                output_widths=(2,),
                linear_family="matmul_aa",
                plan_sha256="a" * 64,
            )
        with self.assertRaisesRegex(AssertionError, "lifetime"):
            mjv.model_setup_target(
                model_id=7,
                sum_r=3,
                output_widths=(2,),
                linear_family="matmul_aa",
                plan_sha256="a" * 64,
                model_epoch=70,
                physical_family_id="dense.family.0",
                logical_object_id="request.hidden.0",
                logical_members=1,
                object_lifetime="append-only",
                object_operand="lhs",
                family_kind="other_dense",
                shared_shape=(2, 3),
                right_shape=(3, 2),
            )


if __name__ == "__main__":
    unittest.main()
