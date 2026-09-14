"""Verifica integridad, configuración y emparejamiento de un lote del juez."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rhlab.judge_spec import JUDGE_INPUT_FIELDS


MANDATORY_METRICS = {
    "derived_scope_accuracy",
    "uncertainty_high_recall",
    "uncertainty_high_false_positive_rate",
    "compatibility_band_accuracy",
    "uncertain_scope_recall",
    "critical_out_to_in",
    "in_to_out",
    "stability",
}
SHARED_PAIR_FIELDS = (
    "intent_compatibility",
    "intent_uncertainty",
    "evidence_level",
    "judge_failure",
    "judge_model",
    "judge_max_tokens",
    "case_id",
    "rep",
    "oracle_scope",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root: Path) -> dict:
    manifest = json.loads((root / "batch_manifest.json").read_text(encoding="utf-8"))
    expected_outputs = manifest["case_count"] * manifest["repetitions"]
    checks: dict[str, dict[str, bool]] = {}

    for model in manifest["models"]:
        alias = model["alias"]
        raw = [
            json.loads(line)
            for line in (root / f"{alias}.judge_outputs.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        with (root / f"{alias}.csv").open(encoding="utf-8", newline="") as source:
            rows = list(csv.DictReader(source))
        config = json.loads((root / f"{alias}.config.json").read_text(encoding="utf-8"))
        summary = json.loads((root / f"{alias}.summary.json").read_text(encoding="utf-8"))

        groups: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            groups[row["judge_output_id"]].append(row)
        paired = all(
            len(group) == 2
            and {row["experimental_arm"] for row in group} == {"C-UNC", "C-COMP"}
            and all(group[0][field] == group[1][field] for field in SHARED_PAIR_FIELDS)
            for group in groups.values()
        )
        metric_shape = all(
            MANDATORY_METRICS <= set(summary["arms"][arm][population])
            for arm in ("C-UNC", "C-COMP")
            for population in ("valid_only", "end_to_end")
        )
        raw_ids = {output["judge_output_id"] for output in raw}
        checks[alias] = {
            "raw_output_count": len(raw) == expected_outputs,
            "paired_row_count": len(rows) == 2 * expected_outputs,
            "unique_pair_count": len(groups) == expected_outputs,
            "same_output_reused": paired and raw_ids == set(groups),
            "case_repetitions_complete": len({(o["case_id"], o["rep"]) for o in raw}) == expected_outputs,
            "labels_excluded_from_judge_input": all(
                set(output["judge_input"]) == set(JUDGE_INPUT_FIELDS) for output in raw
            ),
            "max_tokens_logged": all(
                output["judge_max_tokens"] == manifest["judge_max_tokens"] for output in raw
            ),
            "config_matches_batch": (
                config["dataset"] == manifest["dataset"]
                and config["reps"] == manifest["repetitions"]
                and config["temperature"] == manifest["temperature"]
                and config["judge_max_tokens"] == manifest["judge_max_tokens"]
                and config["response_format"] == manifest["response_format"]
                and config["judge_model"] == model["model_id"]
            ),
            "metrics_complete": summary["metrics_version"] == "paired-evaluation-v3" and metric_shape,
        }

    dataset = Path(manifest["dataset_path"])
    spec_dir = Path(manifest["spec_dir"])
    if not dataset.is_absolute():
        dataset = ROOT / dataset
    if not spec_dir.is_absolute():
        spec_dir = ROOT / spec_dir
    input_hashes_match = all(
        sha256(dataset if name == dataset.name else spec_dir / name) == digest
        for name, digest in manifest["input_sha256"].items()
    )
    overall = (
        manifest["status"] == "complete"
        and manifest["api_key_stored"] is False
        and input_hashes_match
        and all(all(result.values()) for result in checks.values())
    )
    return {
        "overall_pass": overall,
        "batch_status": manifest["status"],
        "input_hashes_match": input_hashes_match,
        "api_key_stored": False,
        "expected_outputs_per_model": expected_outputs,
        "models": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch_dir", type=Path)
    args = parser.parse_args()
    root = args.batch_dir.resolve()
    report = verify(root)
    (root / "integrity_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"overall_pass={report['overall_pass']} models_verified={len(report['models'])}")
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
