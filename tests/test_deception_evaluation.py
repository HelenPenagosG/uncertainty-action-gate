from pathlib import Path

import pandas as pd

from rhlab.judge_output import JudgeOutput, paired_derivations
from rhlab.judge_spec import load_judge_spec
from scripts.run_deception_batch import load_dataset, parse_metrics, public_row, variant_order
from scripts.eval_deception import audit


ROOT = Path(__file__).resolve().parents[1]


def test_deception_development_dataset_has_complete_pairs():
    records, pairs = load_dataset(ROOT / "spec" / "deception_eval_v1" / "development.jsonl")
    assert len(records) == 20
    assert len(pairs) == 10
    assert {record["provenance"]["origin"] for record in records} == {"prepared"}
    assert all({record["variant"] for record in pair} == {"factual", "deceptive"}
               for pair in pairs.values())


def test_deception_variant_order_alternates():
    assert variant_order(1) == ("factual", "deceptive")
    assert variant_order(2) == ("deceptive", "factual")
    assert variant_order(3) == ("factual", "deceptive")


def test_deception_public_row_uses_one_output_for_both_gates():
    records, _ = load_dataset(ROOT / "spec" / "deception_eval_v1" / "development.jsonl")
    spec = load_judge_spec(ROOT / "spec" / "judge_v1_2_pilot")
    output = JudgeOutput(
        judge_output_id="shared-output",
        scores={
            "intent_compatibility": 0.9,
            "intent_uncertainty": 0.9,
            "evidence_level": "INSUFFICIENT_EVIDENCE",
            "reason": "Authorization remains uncertain.",
        },
        judge_model="mock",
        prompt_version=spec.version,
        judge_max_tokens=1024,
        json_valid_first=True,
        json_valid_final=True,
    )
    derivations = paired_derivations(output, spec)
    row = public_row(records[0], 1, 1, "conversation", "run", output, derivations)

    assert row["judge_output_id"] == "shared-output"
    assert row["scope_c_unc"] == "UNCERTAIN"
    assert row["scope_c_comp"] == "IN_SCOPE"
    assert row["gate_c_unc"] == "RECONSIDER"
    assert row["gate_c_comp"] == "APPROVE"
    assert "action_justification" not in row
    assert "reason" not in row


def test_deception_evaluator_accepts_alternating_independent_pairs():
    records, _ = load_dataset(ROOT / "spec" / "deception_eval_v1" / "development.jsonl")
    rows = []
    scope_to_gate = {
        "IN_SCOPE": "APPROVE",
        "OUT_OF_SCOPE": "BLOCK",
        "UNCERTAIN": "RECONSIDER",
    }
    for rep in (1, 2):
        order = variant_order(rep)
        for record in records:
            scope = record["private_labels"]["oracle_scope"]
            rows.append({
                "scenario_id": record["scenario_id"],
                "case_id": record["case_id"],
                "pair_id": record["pair_id"],
                "variant": record["variant"],
                "rep": rep,
                "oracle_scope": scope,
                "json_valid_final": True,
                "judge_output_id": f"{record['case_id']}-{rep}",
                "conversation_id": f"conversation-{record['case_id']}-{rep}",
                "eval_order": order.index(record["variant"]) + 1,
                "gate_c_unc": scope_to_gate[scope],
                "gate_c_comp": scope_to_gate[scope],
            })

    audit(pd.DataFrame(rows))


def test_deception_batch_parses_metric_report(tmp_path):
    report = tmp_path / "metrics.txt"
    report.write_text(
        "[END_TO_END]\n"
        "scope_agreement_C-UNC                                        18/20          0.9000  "
        "95% cluster CI [  0.8000,   1.0000]\n",
        encoding="utf-8",
    )

    parsed = parse_metrics(report)

    assert parsed[("END_TO_END", "scope_agreement_C-UNC")] == {
        "numerator": 18,
        "denominator": 20,
        "value": 0.9,
        "ci_low": 0.8,
        "ci_high": 1.0,
    }
