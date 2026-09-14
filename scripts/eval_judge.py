"""Evalúa una vez cada caso/repetición y deriva C-UNC y C-COMP del mismo output.

No existe --gate-mode: la ablación es siempre pareada. Un retry solo se permite
para recuperar una respuesta inválida, nunca para evaluar el segundo brazo.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rhlab.judge_spec import JUDGE_INPUT_FIELDS, load_judge_spec
from rhlab.judge_output import collect_judge_output, paired_derivations
from rhlab.judge_metrics import summarize_paired
from rhlab.llm_client import OpenAICompatibleClient

ROOT = Path(__file__).resolve().parents[1]


def build_messages(prompt: str, case: dict) -> list[dict]:
    return [{"role": "system", "content": prompt},
            {"role": "user", "content": json.dumps({k: case.get(k) for k in JUDGE_INPUT_FIELDS}, ensure_ascii=False)}]


async def eval_once(client, spec, case, use_rf=False, max_tokens=1024):
    return await collect_judge_output(
        client, spec, {k: case.get(k) for k in JUDGE_INPUT_FIELDS},
        json_mode=use_rf, max_tokens=max_tokens,
    )


def rows_for_case(output, spec, case, rep: int, run_id: str) -> list[dict]:
    return [{**r, "run_id": run_id, "case_id": case["case_id"], "rep": rep,
             "category": case.get("category", ""), "oracle_scope": case["oracle_scope"],
             "expected_compatibility": case.get("expected_compatibility"),
             "expected_uncertainty": case.get("expected_uncertainty")}
            for r in paired_derivations(output, spec)]


def write_results(path: Path, rows: list[dict], spec, *, metadata: dict) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows.sort(key=lambda r: (r["case_id"], r["rep"], r["experimental_arm"]))
    # `reason` puede revelar el input retenido. Se conserva en el JSONL crudo
    # privado, pero no en el CSV publicable; no participa en ninguna métrica.
    csv_rows = [{key: value for key, value in row.items() if key != "reason"} for row in rows]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(csv_rows[0]) if csv_rows else [])
        writer.writeheader()
        writer.writerows(csv_rows)
    summary = {**metadata, **summarize_paired(rows, spec)}
    path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


async def run(args) -> int:
    spec = load_judge_spec(Path(args.spec_dir))
    dataset = Path(args.spec_dir) / ("pilot_calibration.jsonl" if args.dataset == "calibration" else "held_out_evaluation.jsonl")
    if not dataset.is_file():
        print(f"Dataset absent: {dataset}")
        return 2
    if args.dataset == "heldout":
        print("HELD-OUT: one evaluation after freezing; do not tune using these results.")
    cases = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines() if line.strip()]
    out = Path(args.out)
    raw_path = out.with_suffix(".judge_outputs.jsonl")
    config_path = out.with_suffix(".config.json")
    for path in (out, raw_path, config_path, out.with_suffix(".summary.json")):
        if path.exists():
            raise ValueError(f"Output already exists; choose a fresh --out: {path}")
    out.parent.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex
    config = {"run_id": run_id, "dataset": args.dataset, "reps": args.reps,
              "judge_model": args.judge_model, "temperature": 0.0,
              "judge_max_tokens": args.judge_max_tokens,
              "response_format": args.response_format, "timeout": args.timeout,
              "concurrency": args.concurrency, "spec": spec.effective_config(), "paired": True}
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    client = OpenAICompatibleClient(args.judge_model, args.base_url, api_key=args.api_key,
                                   timeout=args.timeout, use_response_format=args.response_format, label="judge-eval")
    semaphore = asyncio.Semaphore(args.concurrency)
    rows = []

    async def worker(case, rep):
        async with semaphore:
            output = await eval_once(client, spec, case, args.response_format, args.judge_max_tokens)
        return case, rep, output

    tasks = [asyncio.create_task(worker(case, rep)) for case in cases for rep in range(1, args.reps + 1)]
    try:
        with raw_path.open("w", encoding="utf-8") as raw_file:
            for task in asyncio.as_completed(tasks):
                case, rep, output = await task
                raw_file.write(json.dumps({**output.to_dict(), "run_id": run_id,
                                           "case_id": case["case_id"], "rep": rep}, ensure_ascii=False) + "\n")
                raw_file.flush()
                rows.extend(rows_for_case(output, spec, case, rep, run_id))
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await client.aclose()
    summary = write_results(out, rows, spec, metadata={"effective_config": config, "raw_outputs": raw_path.name})
    for arm, metrics in summary["arms"].items():
        print(arm, "valid-only", metrics["valid_only"]["derived_scope_accuracy"],
              "end-to-end", metrics["end_to_end"]["derived_scope_accuracy"])
    print(f"Paired CSV: {out}\nShared raw outputs: {raw_path}")
    return 0


def positive_int(text):
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["calibration", "heldout"], default="calibration")
    parser.add_argument("--reps", type=positive_int, default=3)
    parser.add_argument("--spec-dir", default=str(ROOT / "spec" / "judge_v1_2_pilot"))
    parser.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--judge-max-tokens", type=positive_int, default=1024)
    parser.add_argument("--api-key", default=os.environ.get("OPENROUTER_API_KEY", "EMPTY"))
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--concurrency", type=positive_int, default=3)
    parser.add_argument("--response-format", action="store_true")
    parser.add_argument("--out", default=str(ROOT / "results" / "judge_eval_paired" / "judge_evaluation.csv"))
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(asyncio.run(run(parse_args())))
