"""Verify counts, pairing, hashes, and raw/public linkage for a deception batch."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.eval_deception import audit
from scripts.run_deception_batch import refresh_artifacts, sha256, write_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args()
    out_dir = args.out_dir.resolve()
    manifest_path = out_dir / "batch_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors = []
    model_reports = []
    expected = manifest["logical_outputs_per_model"]

    for model in manifest["models"]:
        alias = model["alias"]
        csv_path = out_dir / f"{alias}.paired_results.csv"
        raw_path = out_dir / f"{alias}.raw_attempts.jsonl"
        frame = pd.read_csv(csv_path)
        try:
            audit(frame)
        except SystemExit as exc:
            errors.append(f"{alias}: {exc}")
        if len(frame) != expected:
            errors.append(f"{alias}: {len(frame)} public rows, expected {expected}")
        if frame["judge_output_id"].nunique() != expected:
            errors.append(f"{alias}: judge_output_id is not unique per logical evaluation")
        if frame["conversation_id"].nunique() != expected:
            errors.append(f"{alias}: conversation_id is not unique per variant")
        expected_order = frame.apply(
            lambda row: 1 if ((row["rep"] % 2 == 1) == (row["variant"] == "factual")) else 2,
            axis=1,
        )
        if not frame["eval_order"].eq(expected_order).all():
            errors.append(f"{alias}: variant order does not alternate")

        raw_count = None
        if raw_path.is_file():
            raw = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines() if line]
            raw_count = len(raw)
            if raw_count != expected:
                errors.append(f"{alias}: {raw_count} raw outputs, expected {expected}")
            if {row["judge_output_id"] for row in raw} != set(frame["judge_output_id"]):
                errors.append(f"{alias}: raw/public judge_output_id mismatch")
        model_reports.append({
            "alias": alias,
            "public_rows": len(frame),
            "raw_rows": raw_count,
            "valid_rows": int(frame["json_valid_final"].astype(str).str.lower().isin(["true", "1", "yes"]).sum()),
            "unique_outputs": int(frame["judge_output_id"].nunique()),
            "unique_conversations": int(frame["conversation_id"].nunique()),
        })

    for name, expected_hash in manifest.get("artifact_sha256", {}).items():
        path = out_dir / name
        if not path.is_file() or sha256(path) != expected_hash:
            errors.append(f"public artifact hash mismatch: {name}")
    for item in manifest.get("private_artifacts", []):
        path = out_dir / item["name"]
        if path.is_file() and sha256(path) != item["sha256"]:
            errors.append(f"private artifact hash mismatch: {item['name']}")

    report = {
        "overall_pass": not errors,
        "errors": errors,
        "models_verified": len(model_reports),
        "expected_outputs_per_model": expected,
        "models": model_reports,
    }
    report_path = out_dir / "integrity_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    refresh_artifacts(manifest, out_dir, manifest_path)
    write_manifest(manifest_path, manifest)
    print(f"overall_pass={report['overall_pass']} models_verified={len(model_reports)}")
    if errors:
        print("\n".join(errors))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
