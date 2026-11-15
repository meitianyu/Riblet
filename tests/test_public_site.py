import unittest
import numpy as np
from riblet.inputs import bert_runtime_inputs


class PublicSite(unittest.TestCase):
    def test_no_private_role_input(self):
        v = {
            k: np.zeros((1, 128), np.int32)
            for k in ("input_ids", "attention_mask", "token_type_ids")
        }
        result = bert_runtime_inputs(v)
        self.assertEqual(set(result), set(v))
        self.assertTrue(all(x.dtype == np.int64 for x in result.values()))

    def test_only_valid_public_positions_are_calibrated(self):
        from riblet.bert_tables import slots, runtime_arguments

        records = [
            {"site": s, "operator": op, "point": p, "enabled": True, "K": 4} for op, s, p in slots()
        ]
        hi = np.ones((169, 2))
        lo = np.zeros_like(hi)
        r, ab, l, cd = runtime_arguments(np, records, hi, lo, 3)
        self.assertEqual(r[:, 0].tolist(), [1, 0, 0])
        self.assertEqual(l[:, 0].tolist(), [0, 0, 0])
        self.assertEqual(ab.shape, (3, 144, 2))
        self.assertEqual(cd.shape, (3, 25, 2))

    def test_custom_total_and_paper_population_are_distinct(self):
        from riblet.development import task_counts

        counts = task_counts(40000)
        self.assertEqual(sum(counts.values()), 40000)
        self.assertEqual(len(counts), 8)
        self.assertEqual(task_counts(50000)["mrpc"], 1779)
        with self.assertRaises(ValueError):
            task_counts(1)

    def test_inactive_site_is_not_an_enabled_zero_iteration_site(self):
        from riblet.bert_tables import slots, validate, runtime_arguments

        rows = [dict(operator=o, site=s, point=p, enabled=True, K=0) for o, s, p in slots()]
        rows[0].update(enabled=False, K=None)
        x = np.zeros((169, 2))
        r, *_ = runtime_arguments(np, validate(rows), x, x, 3)
        self.assertEqual(r[:, 0].tolist(), [1, 1, 1])
        self.assertEqual(r[:, 1].tolist(), [1, 0, 0])
        rows[0]["K"] = 0
        with self.assertRaises(ValueError):
            validate(rows)
