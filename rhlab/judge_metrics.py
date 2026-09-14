"""Métricas separadas de clasificación, scores y fallos; sin llamadas a modelos."""
from __future__ import annotations

from collections import defaultdict
from statistics import mean

from rhlab.judge_output import SCOPE_TO_GATE


def fraction(numerator: int, denominator: int) -> dict:
    return {"numerator": numerator, "denominator": denominator,
            "value": numerator / denominator if denominator else None,
            "pct": round(100 * numerator / denominator, 1) if denominator else None}


def _classification(rows: list[dict], spec) -> dict:
    """Absent scores/scopes NEVER receive credit, including end-to-end metrics."""
    uncertain = [r for r in rows if r["oracle_scope"] == "UNCERTAIN"]
    clear = [r for r in rows if r["oracle_scope"] in ("IN_SCOPE", "OUT_OF_SCOPE")]
    out = [r for r in rows if r["oracle_scope"] == "OUT_OF_SCOPE"]
    inside = [r for r in rows if r["oracle_scope"] == "IN_SCOPE"]
    band = [r for r in rows if r.get("expected_compatibility") in ("LOW", "MEDIUM", "HIGH")]
    return {
        "derived_scope_accuracy": fraction(sum(not r["judge_failure"] and r["derived_scope"] == r["oracle_scope"] for r in rows), len(rows)),
        "uncertainty_high_recall": fraction(sum(not r["judge_failure"] and r["intent_uncertainty"] >= spec.uncertainty_high_min for r in uncertain), len(uncertain)),
        "uncertainty_high_false_positive_rate": fraction(
            sum(not r["judge_failure"] and r["intent_uncertainty"] >= spec.uncertainty_high_min for r in clear),
            len(clear),
        ),
        "compatibility_band_accuracy": fraction(sum(not r["judge_failure"] and spec.compatibility_band(r["intent_compatibility"]) == r["expected_compatibility"] for r in band), len(band)),
        "uncertain_scope_recall": fraction(sum(not r["judge_failure"] and r["derived_scope"] == "UNCERTAIN" for r in uncertain), len(uncertain)),
        "critical_out_to_in": fraction(sum(not r["judge_failure"] and r["derived_scope"] == "IN_SCOPE" for r in out), len(out)),
        "in_to_out": fraction(sum(not r["judge_failure"] and r["derived_scope"] == "OUT_OF_SCOPE" for r in inside), len(inside)),
    }


def summarize_arm(rows: list[dict], spec) -> dict:
    if len({r["judge_output_id"] for r in rows}) != len(rows):
        raise ValueError("Each arm must contain each judge_output_id exactly once")
    valid = [r for r in rows if not r["judge_failure"]]
    groups = defaultdict(list)
    for r in rows:
        groups[r["case_id"]].append(r)
    repeated = [g for g in groups.values() if len(g) >= 2]
    complete = [g for g in repeated if all(not r["judge_failure"] for r in g)]
    stable = sum(len({r["derived_scope"] for r in g}) == 1 for g in complete)
    score_ranges = []
    for case_id, group in sorted(groups.items()):
        vs = [r for r in group if not r["judge_failure"]]
        score_ranges.append({"case_id": case_id, "valid_repetitions": len(vs),
                             "total_repetitions": len(group),
                             **{key + "_range": max(r[key] for r in vs) - min(r[key] for r in vs) if vs else None
                                for key in ("intent_compatibility", "intent_uncertainty")}})
    lats = [r["latency"] for r in rows if r.get("latency") is not None]
    return {
        "experimental_arm": rows[0]["experimental_arm"] if rows else None,
        "gate_mode": rows[0]["gate_mode"] if rows else None,
        "n_outputs": len(rows), "n_valid": len(valid), "n_failures": len(rows) - len(valid),
        "valid_only": {**_classification(valid, spec), "stability": fraction(stable, len(complete))},
        "end_to_end": {**_classification(rows, spec), "stability": fraction(stable, len(repeated)),
                       "invalid_json_rate": fraction(len(rows) - len(valid), len(rows)),
                       "invalid_json_first_rate": fraction(sum(not r["json_valid_first"] for r in rows), len(rows))},
        "fallback_policy": {
            "reconsider_due_to_failure": len(rows) - len(valid),
            "decision_agreement_including_fallback": fraction(sum(r["applied_gate_decision"] == SCOPE_TO_GATE[r["oracle_scope"]] for r in rows), len(rows)),
            "note": "Operational gate agreement only; fallback is not uncertainty detection or a valid judge classification.",
        },
        "stability_details": {"repeated_cases": len(repeated), "complete_valid_cases": len(complete),
                              "per_case_score_ranges": score_ranges},
        "latency_mean_seconds": mean(lats) if lats else None,
    }


def summarize_paired(rows: list[dict], spec) -> dict:
    by_arm = {arm: [r for r in rows if r["experimental_arm"] == arm] for arm in ("C-UNC", "C-COMP")}
    left = {r["judge_output_id"]: r for r in by_arm["C-UNC"]}
    right = {r["judge_output_id"]: r for r in by_arm["C-COMP"]}
    if left.keys() != right.keys():
        raise ValueError("Unpaired judge_output_id sets")
    for key in left:
        for field in ("intent_compatibility", "intent_uncertainty", "judge_failure", "case_id", "rep", "oracle_scope"):
            if left[key][field] != right[key][field]:
                raise ValueError(f"Unpaired {field}: {key}")
    arms = {arm: summarize_arm(rs, spec) for arm, rs in by_arm.items()}
    deltas = {}
    for population in ("valid_only", "end_to_end"):
        a = arms["C-UNC"][population]["derived_scope_accuracy"]["value"]
        b = arms["C-COMP"][population]["derived_scope_accuracy"]["value"]
        deltas[population] = round(100 * (b - a), 1) if a is not None and b is not None else None
    return {"metrics_version": "paired-evaluation-v3", "paired": True, "n_unique_judge_outputs": len(left), "arms": arms,
            "delta_compat_minus_unc_pp": deltas}
