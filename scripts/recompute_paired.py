"""Recalcula la ablación desde los CSV C-UNC históricos, sin red ni llamadas LLM.

Los CSV no conservaron respuestas crudas: los outputs migrados indican
raw_available=false. Nunca se presenta un JSON reconstruido como respuesta cruda.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rhlab.judge_output import JudgeOutput
from rhlab.judge_spec import JUDGE_INPUT_FIELDS, load_judge_spec, validate_output
from scripts.eval_judge import rows_for_case, write_results

ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_ROOT = ROOT / "results" / "judge_eval_v1_2" / "calibration_3rep"
DEFAULT_SOURCE = CALIBRATION_ROOT / "source_c_unc"


def recompute(source_dir: Path, destination: Path, spec_dir: Path) -> list[dict]:
    spec = load_judge_spec(spec_dir)
    cases = {c["case_id"]: c for c in (json.loads(line) for line in
             (spec_dir / "pilot_calibration.jsonl").read_text(encoding="utf-8").splitlines() if line.strip())}
    destination.mkdir(parents=True, exist_ok=True)
    comparison = []
    for path in sorted(source_dir.glob("*.csv")):
        with path.open(encoding="utf-8-sig", newline="") as fh:
            source_rows = list(csv.DictReader(fh))
        if not source_rows or "derived_scope" not in source_rows[0]:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        run_id = f"reanalysis-{path.stem}-{digest[:12]}"
        raw_path = destination / f"{path.stem}.judge_outputs.jsonl"
        rows, seen = [], set()
        with raw_path.open("w", encoding="utf-8") as raw_file:
            for legacy in source_rows:
                if legacy["prompt_version"] != spec.version:
                    raise ValueError(f"Spec version mismatch: {path}")
                case = cases[legacy["case_id"]]
                if legacy["oracle_scope"] != case["oracle_scope"]:
                    raise ValueError(f"Oracle changed: {case['case_id']}")
                rep = int(legacy["rep"])
                key = (case["case_id"], rep)
                if key in seen:
                    raise ValueError(f"Duplicate case/repetition: {key}")
                seen.add(key)
                valid = legacy["json_valid_final"] == "True"
                scores = None
                if valid:
                    scores = {"intent_compatibility": float(legacy["intent_compatibility"]),
                              "intent_uncertainty": float(legacy["intent_uncertainty"]),
                              "evidence_level": legacy["evidence_level"], "reason": legacy["reason"]}
                    ok, error = validate_output(scores)
                    if not ok:
                        raise ValueError(f"Invalid historical scores {key}: {error}")
                output = JudgeOutput(
                    judge_output_id=hashlib.sha256(f"{digest}:{case['case_id']}:{rep}".encode()).hexdigest(),
                    scores=scores, judge_model=legacy["model_id"], prompt_version=legacy["prompt_version"],
                    judge_max_tokens=None,
                    latency=float(legacy["latency_s"]), json_valid_first=legacy["json_valid_first"] == "True",
                    json_valid_final=valid, error=None if valid else legacy["reason"],
                    raw_available=False, source="historical_csv",
                )
                raw_file.write(json.dumps({**output.to_dict(), "case_id": case["case_id"], "rep": rep,
                    "run_id": run_id, "source_file": path.name, "source_sha256": digest,
                    "legacy_record": legacy,
                    "input_reconstructed_from_calibration": {k: case[k] for k in JUDGE_INPUT_FIELDS}},
                    ensure_ascii=False) + "\n")
                rows.extend(rows_for_case(output, spec, case, rep, run_id))
        summary = write_results(destination / path.name, rows, spec, metadata={
            "judge_model": source_rows[0]["model_id"], "dataset": "calibration",
            "source_file": path.name, "source_sha256": digest,
            "raw_outputs": raw_path.name, "raw_available": False,
            "effective_derivation_config": spec.effective_config(),
            "historical_generation_config": {"temperature": 0.0, "judge_max_tokens": None,
                                               "complete_config_available": False},
            "token_note": "Legacy CSV retained only last-attempt usage, not total usage; migrated totals are unknown.",
        })
        unc, comp = summary["arms"]["C-UNC"], summary["arms"]["C-COMP"]
        item = {"model_id": source_rows[0]["model_id"],
                "n_outputs": summary["n_unique_judge_outputs"], "n_valid": unc["n_valid"],
                "c_unc_accuracy_valid_only_pct": unc["valid_only"]["derived_scope_accuracy"]["pct"],
                "c_comp_accuracy_valid_only_pct": comp["valid_only"]["derived_scope_accuracy"]["pct"],
                "c_unc_accuracy_end_to_end_pct": unc["end_to_end"]["derived_scope_accuracy"]["pct"],
                "c_comp_accuracy_end_to_end_pct": comp["end_to_end"]["derived_scope_accuracy"]["pct"],
                "delta_pp": summary["delta_compat_minus_unc_pp"]["end_to_end"],
                "uncertainty_high_correct": unc["end_to_end"]["uncertainty_high_recall"]["numerator"],
                "uncertainty_cases": unc["end_to_end"]["uncertainty_high_recall"]["denominator"],
                "uncertainty_high_false_positive_rate_pct": unc["end_to_end"]["uncertainty_high_false_positive_rate"]["pct"],
                "uncertain_scope_correct": unc["end_to_end"]["uncertain_scope_recall"]["numerator"],
                "compatibility_band_accuracy_valid_only_pct": unc["valid_only"]["compatibility_band_accuracy"]["pct"],
                "invalid_json_rate_pct": unc["end_to_end"]["invalid_json_rate"]["pct"],
                "critical_out_to_in": unc["end_to_end"]["critical_out_to_in"]["numerator"],
                "in_to_out": unc["end_to_end"]["in_to_out"]["numerator"]}
        comparison.append(item)
    comparison.sort(key=lambda r: r["c_unc_accuracy_end_to_end_pct"], reverse=True)
    with (destination / "model_comparison.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out-dir", type=Path, default=CALIBRATION_ROOT / "paired")
    parser.add_argument("--spec-dir", type=Path, default=ROOT / "spec" / "judge_v1_2_pilot")
    args = parser.parse_args()
    for row in recompute(args.source_dir, args.out_dir, args.spec_dir):
        print(row["model_id"], row["c_unc_accuracy_end_to_end_pct"],
              row["c_comp_accuracy_end_to_end_pct"], row["delta_pp"],
              "high uncertainty", row["uncertainty_high_correct"], "/", row["uncertainty_cases"])


if __name__ == "__main__":
    main()
