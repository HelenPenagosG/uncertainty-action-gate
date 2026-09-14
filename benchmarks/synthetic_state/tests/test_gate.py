from __future__ import annotations

import unittest

from synthetic_benchmark.gate import GateConfig, GateController, derive_scope
from synthetic_benchmark.types import JudgeScores


def score(c: float, u: float) -> JudgeScores:
    return JudgeScores(c, u, "INSUFFICIENT_EVIDENCE", "test")


class GateTests(unittest.TestCase):
    def test_frozen_boundaries(self) -> None:
        cfg = GateConfig()
        self.assertEqual(derive_scope(score(0.9, 0.7), cfg, "uncertainty"), "UNCERTAIN")
        self.assertEqual(derive_scope(score(0.7, 0.0), cfg, "uncertainty"), "IN_SCOPE")
        self.assertEqual(derive_scope(score(0.3, 0.0), cfg, "uncertainty"), "OUT_OF_SCOPE")
        self.assertEqual(derive_scope(score(0.1, 0.9), cfg, "compatibility_only"), "OUT_OF_SCOPE")

    def test_reconsider_limit_blocks_current_proposal_without_stopping_run(self) -> None:
        gate = GateController()
        self.assertEqual(gate.apply("RECONSIDER"), "RECONSIDER")
        self.assertEqual(gate.apply("RECONSIDER"), "RECONSIDER")
        self.assertEqual(gate.apply("RECONSIDER"), "BLOCK_CURRENT_PROPOSAL")
        self.assertEqual(gate.apply("EXECUTE"), "EXECUTE")


if __name__ == "__main__":
    unittest.main()
