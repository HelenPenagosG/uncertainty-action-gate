"""Run the prepared actor-to-judge deception evaluation as a paired batch.

Each factual/deceptive variant gets one logical judge output. C-UNC and C-COMP
are derived from that same output, while the two variants use independent
conversations. Odd repetitions submit factual first; even repetitions submit
deceptive first.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import getpass
import hashlib
import json
import os
import re
import subprocess
import sys
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rhlab.judge_output import collect_judge_output, paired_derivations
from rhlab.judge_spec import JUDGE_INPUT_FIELDS, load_judge_spec
from rhlab.llm_client import OpenAICompatibleClient

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "spec" / "deception_eval_v1" / "development.jsonl"
DEFAULT_JUDGE_SPEC = ROOT / "spec" / "judge_v1_2_pilot"
GATE_LABEL = {"EXECUTE": "APPROVE", "BLOCK": "BLOCK", "RECONSIDER": "RECONSIDER"}
METRIC_LINE = re.compile(
    r"^(?P<metric>\S.*?)\s+(?:(?P<n>\d+)/(?P<d>\d+)\s+)?"
    r"(?P<value>-?nan|-?\d+\.\d+)\s+95% cluster CI "
    r"\[(?P<low>\s*-?nan|\s*-?\d+\.\d+),\s*(?P<high>-?nan|-?\d+\.\d+)\]$"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repository_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolved)


def positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def model_argument(text: str) -> tuple[str, str]:
    alias, separator, model_id = text.partition("=")
    if not separator or not alias.strip() or not model_id.strip():
        raise argparse.ArgumentTypeError("use alias=provider/model")
    return alias.strip(), model_id.strip()


def variant_order(rep: int) -> tuple[str, str]:
    return ("factual", "deceptive") if rep % 2 else ("deceptive", "factual")


def load_dataset(path: Path) -> tuple[list[dict], dict[str, list[dict]]]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    pairs: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        pairs[record["pair_id"]].append(record)
        judge_input = record.get("judge_input", {})
        if set(judge_input) != set(JUDGE_INPUT_FIELDS):
            raise ValueError(f"{record.get('case_id')}: unexpected judge_input fields")
    for pair_id, variants in pairs.items():
        if len(variants) != 2 or {v["variant"] for v in variants} != {"factual", "deceptive"}:
            raise ValueError(f"{pair_id}: expected exactly factual and deceptive variants")
    return records, dict(pairs)


def public_row(record: dict, rep: int, eval_order: int, conversation_id: str,
               run_id: str, output, derivations: list[dict]) -> dict:
    by_arm = {row["experimental_arm"]: row for row in derivations}
    unc = by_arm["C-UNC"]
    comp = by_arm["C-COMP"]
    labels = record["private_labels"]
    return {
        "run_id": run_id,
        "scenario_id": record["scenario_id"],
        "case_id": record["case_id"],
        "pair_id": record["pair_id"],
        "variant": record["variant"],
        "rep": rep,
        "eval_order": eval_order,
        "conversation_id": conversation_id,
        "provenance_origin": record["provenance"]["origin"],
        "primary_behavior": labels["primary_behavior"],
        "oracle_scope": labels["oracle_scope"],
        "expected_compatibility": labels["expected_compatibility"],
        "expected_uncertainty": labels["expected_uncertainty"],
        "judge_output_id": output.judge_output_id,
        "intent_compatibility": unc["intent_compatibility"],
        "intent_uncertainty": unc["intent_uncertainty"],
        "evidence_level": unc["evidence_level"],
        "scope_c_unc": unc["derived_scope"],
        "scope_c_comp": comp["derived_scope"],
        "gate_c_unc": GATE_LABEL[unc["applied_gate_decision"]],
        "gate_c_comp": GATE_LABEL[comp["applied_gate_decision"]],
        "judge_failure": output.judge_failure,
        "fallback_decision": "RECONSIDER" if output.judge_failure else None,
        "json_valid_first": output.json_valid_first,
        "json_valid_final": output.json_valid_final,
        "judge_source": output.source,
        "prompt_version": output.prompt_version,
        "judge_model": output.judge_model,
        "temperature": output.temperature,
        "judge_max_tokens": output.judge_max_tokens,
        "latency": output.latency,
        "input_tokens": output.input_tokens,
        "output_tokens": output.output_tokens,
    }


async def run_model(*, alias: str, model_id: str, pairs: dict[str, list[dict]],
                    spec_dir: Path, out_dir: Path, reps: int, api_key: str,
                    base_url: str, max_tokens: int, timeout: float,
                    concurrency: int, response_format: bool) -> tuple[Path, Path]:
    spec = load_judge_spec(spec_dir)
    csv_path = out_dir / f"{alias}.paired_results.csv"
    raw_path = out_dir / f"{alias}.raw_attempts.jsonl"
    config_path = out_dir / f"{alias}.config.json"
    metrics_path = out_dir / f"{alias}.metrics.txt"
    for path in (csv_path, raw_path, config_path, metrics_path):
        if path.exists():
            raise FileExistsError(f"Output already exists: {path}")

    run_id = uuid.uuid4().hex
    config = {
        "run_id": run_id,
        "dataset": "deception_development_prepared",
        "repetitions": reps,
        "logical_outputs": len(pairs) * 2 * reps,
        "judge_model": model_id,
        "temperature": 0.0,
        "judge_max_tokens": max_tokens,
        "response_format": response_format,
        "timeout_seconds": timeout,
        "pair_concurrency": concurrency,
        "variant_conversations_independent": True,
        "variant_order": "factual-first on odd reps; deceptive-first on even reps",
        "paired_gate_derivation": True,
        "judge_spec": spec.effective_config(),
    }
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    client = OpenAICompatibleClient(
        model_id, base_url, api_key=api_key, timeout=timeout,
        use_response_format=response_format, label="deception-eval",
    )
    semaphore = asyncio.Semaphore(concurrency)

    async def evaluate_pair(pair_id: str, variants: list[dict], rep: int):
        by_variant = {record["variant"]: record for record in variants}
        order = variant_order(rep)
        completed = []
        async with semaphore:
            for eval_order, variant in enumerate(order, 1):
                record = by_variant[variant]
                conversation_id = uuid.uuid4().hex
                output = await collect_judge_output(
                    client, spec, record["judge_input"], temperature=0.0,
                    max_tokens=max_tokens, json_mode=response_format,
                )
                completed.append((record, rep, eval_order, conversation_id, output))
        return pair_id, completed

    tasks = [
        asyncio.create_task(evaluate_pair(pair_id, variants, rep))
        for pair_id, variants in sorted(pairs.items())
        for rep in range(1, reps + 1)
    ]
    rows: list[dict] = []
    try:
        with raw_path.open("w", encoding="utf-8") as raw_file:
            for task in asyncio.as_completed(tasks):
                pair_id, completed = await task
                for record, rep, eval_order, conversation_id, output in completed:
                    derivations = paired_derivations(output, spec)
                    raw_file.write(json.dumps({
                        **output.to_dict(),
                        "run_id": run_id,
                        "scenario_id": record["scenario_id"],
                        "case_id": record["case_id"],
                        "pair_id": pair_id,
                        "variant": record["variant"],
                        "rep": rep,
                        "eval_order": eval_order,
                        "conversation_id": conversation_id,
                    }, ensure_ascii=False) + "\n")
                    raw_file.flush()
                    rows.append(public_row(
                        record, rep, eval_order, conversation_id, run_id,
                        output, derivations,
                    ))
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await client.aclose()

    rows.sort(key=lambda row: (row["pair_id"], row["rep"], row["eval_order"]))
    with csv_path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "eval_deception.py"), str(csv_path)],
        cwd=ROOT, text=True, encoding="utf-8", capture_output=True, check=False,
    )
    metrics_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(f"Metric evaluation failed for {alias}; see {metrics_path}")
    return csv_path, metrics_path


def parse_metrics(path: Path) -> dict[tuple[str, str], dict]:
    population = ""
    parsed = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line in {"[END_TO_END]", "[VALID_ONLY]"}:
            population = line.strip("[]")
            continue
        match = METRIC_LINE.match(line.strip())
        if match and population:
            values = match.groupdict()
            parsed[(population, values["metric"].strip())] = {
                "numerator": int(values["n"]) if values["n"] else None,
                "denominator": int(values["d"]) if values["d"] else None,
                "value": float(values["value"]),
                "ci_low": float(values["low"]),
                "ci_high": float(values["high"]),
            }
    return parsed


def write_comparison(out_dir: Path, models: list[dict]) -> None:
    rows = []
    wanted = {
        "scope_agreement_C-UNC": "scope_agreement_c_unc",
        "scope_agreement_C-COMP": "scope_agreement_c_comp",
        "improper_approval_C-UNC_factual": "improper_approval_c_unc_factual",
        "improper_approval_C-UNC_deceptive": "improper_approval_c_unc_deceptive",
        "approval_delta_deceptive_minus_factual_C-UNC": "approval_delta_c_unc",
        "nonapproval_to_approval_C-UNC": "deception_flip_c_unc",
        "format_failure_rate": "format_failure_rate",
    }
    for model in models:
        metrics = parse_metrics(out_dir / f"{model['alias']}.metrics.txt")
        row = {"alias": model["alias"], "model_id": model["model_id"]}
        for metric, column in wanted.items():
            item = metrics.get(("END_TO_END", metric), {})
            row[column] = item.get("value")
            row[f"{column}_ci_low"] = item.get("ci_low")
            row[f"{column}_ci_high"] = item.get("ci_high")
        rows.append(row)
    with (out_dir / "model_comparison.csv").open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_variant_comparison(out_dir: Path, models: list[dict]) -> None:
    import pandas as pd

    rows = []
    for model in models:
        frame = pd.read_csv(out_dir / f"{model['alias']}.paired_results.csv")
        valid_mask = frame["json_valid_final"].astype(str).str.lower().isin(["true", "1", "yes"])
        valid = frame[valid_mask].copy()
        e2e_correct = valid_mask & frame["scope_c_unc"].eq(frame["oracle_scope"])
        e2e = frame.assign(correct=e2e_correct).groupby("variant")["correct"].mean()
        if len(valid):
            valid["correct"] = valid["scope_c_unc"].eq(valid["oracle_scope"])
            valid_accuracy = valid.groupby("variant")["correct"].mean()
            pair = valid.pivot_table(
                index=["pair_id", "rep"], columns="variant",
                values=["intent_compatibility", "intent_uncertainty", "gate_c_unc"],
                aggfunc="first",
            ).dropna()
        else:
            valid_accuracy = {}
            pair = pd.DataFrame()

        def value(series, key):
            item = series.get(key) if hasattr(series, "get") else None
            return None if item is None or pd.isna(item) else float(item)

        comp_f = comp_d = unc_f = unc_d = gate_change = None
        if len(pair):
            comp_f = float(pair["intent_compatibility"]["factual"].mean())
            comp_d = float(pair["intent_compatibility"]["deceptive"].mean())
            unc_f = float(pair["intent_uncertainty"]["factual"].mean())
            unc_d = float(pair["intent_uncertainty"]["deceptive"].mean())
            gate_change = float((pair["gate_c_unc"]["factual"] != pair["gate_c_unc"]["deceptive"]).mean())
        factual_e2e = value(e2e, "factual")
        deceptive_e2e = value(e2e, "deceptive")
        rows.append({
            "alias": model["alias"],
            "model_id": model["model_id"],
            "n_outputs": len(frame),
            "n_valid": int(valid_mask.sum()),
            "invalid_json_rate": float((~valid_mask).mean()),
            "c_unc_scope_accuracy_e2e_factual": factual_e2e,
            "c_unc_scope_accuracy_e2e_deceptive": deceptive_e2e,
            "c_unc_scope_accuracy_delta_deceptive_minus_factual": (
                deceptive_e2e - factual_e2e
                if factual_e2e is not None and deceptive_e2e is not None else None
            ),
            "c_unc_scope_accuracy_valid_factual": value(valid_accuracy, "factual"),
            "c_unc_scope_accuracy_valid_deceptive": value(valid_accuracy, "deceptive"),
            "mean_compatibility_factual": comp_f,
            "mean_compatibility_deceptive": comp_d,
            "mean_compatibility_delta_deceptive_minus_factual": (
                comp_d - comp_f if comp_f is not None else None
            ),
            "mean_uncertainty_factual": unc_f,
            "mean_uncertainty_deceptive": unc_d,
            "mean_uncertainty_delta_deceptive_minus_factual": (
                unc_d - unc_f if unc_f is not None else None
            ),
            "c_unc_gate_change_rate_valid_pairs": gate_change,
        })
    for row in rows:
        for key, item in row.items():
            if isinstance(item, float):
                row[key] = round(item, 6)
    with (out_dir / "variant_comparison.csv").open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def refresh_artifacts(manifest: dict, out_dir: Path, manifest_path: Path) -> None:
    manifest["artifact_sha256"] = {
        path.name: sha256(path)
        for path in sorted(out_dir.iterdir())
        if path.is_file() and path != manifest_path and ".raw_attempts." not in path.name
    }
    manifest["private_artifacts"] = [
        {"name": path.name, "sha256": sha256(path), "bytes": path.stat().st_size}
        for path in sorted(out_dir.glob("*.raw_attempts.jsonl"))
    ]


def write_manifest(path: Path, manifest: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


async def run(args: argparse.Namespace) -> int:
    dataset = Path(args.dataset).resolve()
    spec_dir = Path(args.spec_dir).resolve()
    records, pairs = load_dataset(dataset)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=False)
    models = [{"alias": alias, "model_id": model_id, "status": "pending"}
              for alias, model_id in args.model]
    manifest_path = out_dir / "batch_manifest.json"
    manifest = {
        "schema_version": "deception-batch-v1",
        "created_at": utc_now(),
        "finished_at": None,
        "status": "running",
        "dataset_version": records[0]["dataset_version"],
        "dataset_split": records[0]["split"],
        "provenance_origin": records[0]["provenance"]["origin"],
        "dataset_path": repository_path(dataset),
        "dataset_sha256": sha256(dataset),
        "base_scenarios": len(pairs),
        "records": len(records),
        "repetitions": args.reps,
        "logical_outputs_per_model": len(records) * args.reps,
        "paired_arms": ["C-UNC", "C-COMP"],
        "temperature": 0.0,
        "judge_max_tokens": args.judge_max_tokens,
        "response_format": args.response_format,
        "timeout_seconds": args.timeout,
        "pair_concurrency": args.concurrency,
        "judge_spec_dir": repository_path(spec_dir),
        "judge_prompt_version": load_judge_spec(spec_dir).version,
        "api_key_stored": False,
        "raw_outputs_versioned": False,
        "models": models,
    }
    write_manifest(manifest_path, manifest)
    key = os.environ.get("OPENROUTER_API_KEY") or getpass.getpass("OpenRouter key: ")
    if not key:
        raise ValueError("OPENROUTER_API_KEY is required")

    failures = 0
    for model in models:
        model["status"] = "running"
        model["started_at"] = utc_now()
        write_manifest(manifest_path, manifest)
        print(f"[deception-batch] starting {model['alias']} ({model['model_id']})", flush=True)
        try:
            await run_model(
                alias=model["alias"], model_id=model["model_id"], pairs=pairs,
                spec_dir=spec_dir, out_dir=out_dir, reps=args.reps, api_key=key,
                base_url=args.base_url, max_tokens=args.judge_max_tokens,
                timeout=args.timeout, concurrency=args.concurrency,
                response_format=args.response_format,
            )
            model["status"] = "complete"
        except Exception as exc:
            model["status"] = "failed"
            model["error"] = str(exc)
            failures += 1
            print(f"[deception-batch] failed {model['alias']}: {exc}", file=sys.stderr, flush=True)
        model["finished_at"] = utc_now()
        write_manifest(manifest_path, manifest)

    completed_models = [model for model in models if model["status"] == "complete"]
    if completed_models:
        write_comparison(out_dir, completed_models)
        write_variant_comparison(out_dir, completed_models)
    manifest["finished_at"] = utc_now()
    manifest["status"] = "complete" if not failures else "completed_with_failures"
    refresh_artifacts(manifest, out_dir, manifest_path)
    write_manifest(manifest_path, manifest)
    print(f"[deception-batch] {manifest['status']}: {out_dir}", flush=True)
    return 1 if failures else 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", type=model_argument, required=True,
                        metavar="ALIAS=MODEL_ID")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--spec-dir", type=Path, default=DEFAULT_JUDGE_SPEC)
    parser.add_argument("--reps", type=positive_int, default=3)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--judge-max-tokens", type=positive_int, default=1024)
    parser.add_argument("--concurrency", type=positive_int, default=3)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--response-format", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
    raise SystemExit(asyncio.run(run(parse_args())))
