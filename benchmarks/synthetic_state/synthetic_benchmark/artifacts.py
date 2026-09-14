from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


class RunArtifacts:
    def __init__(self, directory: Path, config: dict[str, Any]) -> None:
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=False)
        self.events_path = directory / "steps.jsonl"
        self.summary_path = directory / "summary.json"
        (directory / "config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    def append(self, record: dict[str, Any]) -> None:
        with self.events_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    def finish(self, summary: dict[str, Any]) -> None:
        self.summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )


def write_batch_csv(path: Path, summaries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "run_id", "condition", "scenario_id", "scenario_family", "steps", "executed_steps",
        "flag_captured", "legitimate_success", "forbidden_attempt",
        "forbidden_executed", "route_used", "blocked_actions",
        "reconsidered_actions", "ask_actions", "actor_format_failures",
        "judge_failures", "actor_latency_seconds", "judge_latency_seconds",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summaries)
