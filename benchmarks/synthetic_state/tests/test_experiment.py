from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path

from synthetic_benchmark.clients import MockActorClient, MockJudgeClient
from synthetic_benchmark.environment import SCENARIOS
from synthetic_benchmark.experiment import ExperimentConfig, run_experiment
from synthetic_benchmark.prompts import actor_system_prompt


class ExperimentTests(unittest.TestCase):
    def test_b_and_c_use_identical_actor_prompt(self) -> None:
        scenario = SCENARIOS["nonce_alpha"]
        self.assertEqual(
            actor_system_prompt(scenario, "B"),
            actor_system_prompt(scenario, "C-UNC"),
        )
        self.assertEqual(
            actor_system_prompt(scenario, "B"),
            actor_system_prompt(scenario, "C-COMP"),
        )
        self.assertNotEqual(
            actor_system_prompt(scenario, "A"),
            actor_system_prompt(scenario, "B"),
        )

    def run_case(self, condition: str, policy: str) -> dict:
        scenario = SCENARIOS["nonce_alpha"]
        with tempfile.TemporaryDirectory() as temp:
            summary = run_experiment(
                scenario=scenario,
                actor=MockActorClient(scenario, policy),
                judge=MockJudgeClient(scenario) if condition.startswith("C-") else None,
                output_dir=Path(temp),
                config=ExperimentConfig(condition=condition, max_steps=10),
                run_id="test-run",
            )
            self.assertTrue((Path(temp) / "test-run" / "steps.jsonl").is_file())
            self.assertTrue((Path(temp) / "test-run" / "summary.json").is_file())
            return summary

    def test_baseline_uses_easy_forbidden_route(self) -> None:
        result = self.run_case("B", "shortcut")
        self.assertEqual(result["route_used"], "FORBIDDEN")
        self.assertGreater(result["forbidden_executed"], 0)

    def test_judge_blocks_shortcut_and_actor_recovers_legitimately(self) -> None:
        result = self.run_case("C-UNC", "shortcut")
        self.assertEqual(result["route_used"], "LEGITIMATE")
        self.assertEqual(result["forbidden_executed"], 0)
        self.assertGreater(result["blocked_actions"], 0)

    def test_uncertainty_changes_reconsider_vs_block(self) -> None:
        unc = self.run_case("C-UNC", "ambiguous")
        comp = self.run_case("C-COMP", "ambiguous")
        self.assertGreater(unc["reconsidered_actions"], 0)
        self.assertGreater(unc["ask_actions"], comp["ask_actions"])
        self.assertGreater(comp["blocked_actions"], 0)

    def test_judge_input_excludes_reward_flag_and_private_labels(self) -> None:
        scenario = SCENARIOS["nonce_alpha"]
        with tempfile.TemporaryDirectory() as temp:
            run_experiment(
                scenario=scenario, actor=MockActorClient(scenario, "shortcut"),
                judge=MockJudgeClient(scenario), output_dir=Path(temp),
                config=ExperimentConfig(condition="C-UNC"), run_id="isolation",
            )
            first = json.loads((Path(temp) / "isolation" / "steps.jsonl").read_text(encoding="utf-8").splitlines()[0])
            judge_input = first["judge_input"]
            self.assertEqual(set(judge_input), {
                "operator_authorization", "authorized_resources", "explicit_restrictions",
                "current_context", "recent_action_history", "proposed_action", "action_justification",
            })
            serialized = json.dumps(judge_input).lower()
            self.assertNotIn("actual_scope", serialized)
            self.assertNotIn("oracle_scope", serialized)
            self.assertNotIn("reward", serialized)
            self.assertNotIn("obtain the per-run flag", serialized)


if __name__ == "__main__":
    unittest.main()
