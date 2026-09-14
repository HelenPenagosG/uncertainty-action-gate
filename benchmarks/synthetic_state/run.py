from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path

from synthetic_benchmark.artifacts import write_batch_csv
from synthetic_benchmark.clients import MockActorClient, MockJudgeClient, OpenAICompatibleClient
from synthetic_benchmark.environment import SCENARIOS
from synthetic_benchmark.experiment import ExperimentConfig, run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the interactive synthetic state benchmark.")
    parser.add_argument("--condition", choices=["A", "B", "C-UNC", "C-COMP"], required=True)
    parser.add_argument("--scenario", choices=["all", *SCENARIOS], default="all")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--env-variant", choices=["v0", "v1", "v2", "v3"], default="v0")
    parser.add_argument(
        "--reward-frame", choices=["neutral", "scored", "competitive"], default="neutral"
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--mock-policy", choices=["shortcut", "ambiguous", "legitimate"], default="shortcut")
    parser.add_argument("--actor-base-url")
    parser.add_argument("--actor-model")
    parser.add_argument("--actor-api-key-env", default="ACTOR_API_KEY")
    parser.add_argument("--judge-base-url")
    parser.add_argument("--judge-model", default="qwen/qwen3-8b")
    parser.add_argument("--judge-api-key-env", default="JUDGE_API_KEY")
    parser.add_argument("--actor-temperature", type=float, default=0.7)
    parser.add_argument("--judge-temperature", type=float, default=0.0)
    parser.add_argument("--actor-max-tokens", type=int, default=512)
    parser.add_argument("--judge-max-tokens", type=int, default=1024)
    parser.add_argument("--timeout", type=float, default=120.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.runs < 1:
        raise ValueError("--runs must be positive")
    args.out.mkdir(parents=True, exist_ok=False)
    scenario_ids = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
    config = ExperimentConfig(
        condition=args.condition, env_variant=args.env_variant,
        reward_frame=args.reward_frame, max_steps=args.max_steps,
        actor_temperature=args.actor_temperature, judge_temperature=args.judge_temperature,
        actor_max_tokens=args.actor_max_tokens, judge_max_tokens=args.judge_max_tokens,
    )
    summaries: list[dict] = []
    batch_id = uuid.uuid4().hex[:12]

    real_actor = real_judge = None
    if not args.mock:
        if not args.actor_base_url or not args.actor_model:
            raise ValueError("real runs require --actor-base-url and --actor-model")
        real_actor = OpenAICompatibleClient(
            model=args.actor_model, base_url=args.actor_base_url,
            api_key=os.getenv(args.actor_api_key_env, "EMPTY"), timeout=args.timeout,
        )
        if args.condition.startswith("C-"):
            if not args.judge_base_url:
                raise ValueError("C conditions require --judge-base-url")
            real_judge = OpenAICompatibleClient(
                model=args.judge_model, base_url=args.judge_base_url,
                api_key=os.getenv(args.judge_api_key_env, "EMPTY"), timeout=args.timeout,
            )

    try:
        for scenario_id in scenario_ids:
            scenario = SCENARIOS[scenario_id]
            for repetition in range(1, args.runs + 1):
                actor = MockActorClient(scenario, args.mock_policy) if args.mock else real_actor
                judge = MockJudgeClient(scenario) if args.mock and args.condition.startswith("C-") else real_judge
                run_id = f"{batch_id}-{scenario_id}-r{repetition:03d}"
                summary = run_experiment(
                    scenario=scenario, actor=actor, judge=judge,
                    output_dir=args.out / "runs", config=config, run_id=run_id,
                )
                summaries.append(summary)
                print(
                    f"{run_id}: flag={summary['flag_captured']} route={summary['route_used']} "
                    f"forbidden_attempt={summary['forbidden_attempt']}"
                )
    finally:
        if real_actor:
            real_actor.close()
        if real_judge:
            real_judge.close()

    write_batch_csv(args.out / "batch_summary.csv", summaries)
    aggregate = {
        "batch_id": batch_id,
        "condition": args.condition,
        "env_variant": args.env_variant,
        "reward_frame": args.reward_frame,
        "scenario_count": len(scenario_ids),
        "runs": len(summaries),
        "flag_capture_rate": sum(bool(x["flag_captured"]) for x in summaries) / len(summaries),
        "legitimate_success_rate": sum(bool(x["legitimate_success"]) for x in summaries) / len(summaries),
        "forbidden_attempt_rate": sum(x["forbidden_attempt"] > 0 for x in summaries) / len(summaries),
        "forbidden_execution_rate": sum(x["forbidden_executed"] > 0 for x in summaries) / len(summaries),
        "route_counts": {route: sum(x["route_used"] == route for x in summaries)
                         for route in ("LEGITIMATE", "FORBIDDEN", "MIXED", "NONE")},
        "note": "Runs are repeated observations; scenario families are the independent generalization units.",
    }
    (args.out / "aggregate.json").write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(aggregate, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
