import unittest

import spu.intrinsic as intrinsic


class MJVPublicApiTest(unittest.TestCase):
    def test_exports_runtime_audit_schema_used_by_packaged_harness(self):
        fields = intrinsic.RUNTIME_OBJECT_AUDIT_V1_FIELDS

        self.assertEqual(len(fields), 33)
        self.assertEqual(fields[0], "schema_version")
        self.assertEqual(fields[-1], "live_consumer_contexts")


if __name__ == "__main__":
    unittest.main()
