from __future__ import annotations

import unittest

import numpy as np

from g22_common import downstream_gate_decision, stable_top_k, validity_gate_decision


class G22UtilityTests(unittest.TestCase):
    def test_stable_top_k_tie_break(self) -> None:
        selected = stable_top_k(np.asarray([0.5, 0.8, 0.8, 0.1]), 2)
        self.assertEqual(selected.tolist(), [False, True, True, False])

    def test_validity_gate_failure(self) -> None:
        decision = validity_gate_decision(
            {
                "marginal_auc_mean": 0.90,
                "global_swap_auc_mean": 0.98,
                "near_constant_knockoff_columns_max": 0,
            },
            {"marginal_auc_max": 0.75, "global_swap_auc_max": 0.75},
        )
        self.assertFalse(decision.passed)
        self.assertIn("global_swap_c2st", decision.reasons)

    def test_downstream_gate(self) -> None:
        decision = downstream_gate_decision(
            {
                "fdp_delta": 0.08,
                "average_precision_delta": 0.01,
                "matched_power_delta": 0.00,
                "draw_jaccard_delta": 0.00,
            },
            max_fdp_delta=0.05,
            min_average_precision_delta=-0.02,
            min_matched_power_delta=-0.02,
            min_draw_jaccard_delta=-0.05,
        )
        self.assertFalse(decision.passed)
        self.assertEqual(decision.reasons, ("fdp_delta",))


if __name__ == "__main__":
    unittest.main()
