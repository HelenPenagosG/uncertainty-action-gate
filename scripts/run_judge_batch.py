"""Ejecuta eval_judge.py secuencialmente para un lote reproducible de modelos."""
from __future__ import annotations

import argparse
import csv
import getpass
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC_FILES = (
    "judge_prompt.txt",
    "prompt_version",
    "thresholds.yaml",
    "judge_output.schema.json",
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


def write_manifest(path: Path, manifest: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_comparison(out_dir: Path, manifest: dict) -> Path:
    """Consolida las métricas; los resúmenes por modelo siguen siendo la fuente completa."""
    rows = []
    for model in manifest["models"]:
        alias = model["alias"]
        summary_path = out_dir / f"{alias}.summary.json"
        raw_path = out_dir / f"{alias}.judge_outputs.jsonl"
        if not summary_path.is_file():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        unc = summary["arms"]["C-UNC"]
        comp = summary["arms"]["C-COMP"]

        recorded_costs = []
        cost_attempts = 0
        total_attempts = 0
        if raw_path.is_file():
            for line in raw_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                output = json.loads(line)
                for attempt in output.get("attempts", []):
                    total_attempts += 1
                    raw_response = attempt.get("raw_response")
                    usage = raw_response.get("usage", {}) if isinstance(raw_response, dict) else {}
                    cost = usage.get("cost") if isinstance(usage, dict) else None
                    if isinstance(cost, (int, float)):
                        recorded_costs.append(cost)
                        cost_attempts += 1

        def pct(arm: dict, population: str, metric: str):
            return arm[population][metric]["pct"]

        rows.append({
            "alias": alias,
            "model_id": model["model_id"],
            "n_outputs": unc["n_outputs"],
            "n_valid": unc["n_valid"],
            "c_unc_scope_accuracy_valid_only_pct": pct(unc, "valid_only", "derived_scope_accuracy"),
            "c_unc_scope_accuracy_end_to_end_pct": pct(unc, "end_to_end", "derived_scope_accuracy"),
            "c_comp_scope_accuracy_valid_only_pct": pct(comp, "valid_only", "derived_scope_accuracy"),
            "c_comp_scope_accuracy_end_to_end_pct": pct(comp, "end_to_end", "derived_scope_accuracy"),
            "delta_comp_minus_unc_valid_only_pp": summary["delta_compat_minus_unc_pp"]["valid_only"],
            "delta_comp_minus_unc_end_to_end_pp": summary["delta_compat_minus_unc_pp"]["end_to_end"],
            "uncertainty_high_recall_valid_only_pct": pct(unc, "valid_only", "uncertainty_high_recall"),
            "uncertainty_high_false_positive_rate_valid_only_pct": pct(unc, "valid_only", "uncertainty_high_false_positive_rate"),
            "compatibility_band_accuracy_valid_only_pct": pct(unc, "valid_only", "compatibility_band_accuracy"),
            "c_unc_uncertain_scope_recall_valid_only_pct": pct(unc, "valid_only", "uncertain_scope_recall"),
            "c_comp_uncertain_scope_recall_valid_only_pct": pct(comp, "valid_only", "uncertain_scope_recall"),
            "c_unc_critical_out_to_in_valid_only_pct": pct(unc, "valid_only", "critical_out_to_in"),
            "c_comp_critical_out_to_in_valid_only_pct": pct(comp, "valid_only", "critical_out_to_in"),
            "c_unc_in_to_out_valid_only_pct": pct(unc, "valid_only", "in_to_out"),
            "c_comp_in_to_out_valid_only_pct": pct(comp, "valid_only", "in_to_out"),
            "invalid_json_rate_pct": pct(unc, "end_to_end", "invalid_json_rate"),
            "invalid_json_first_rate_pct": pct(unc, "end_to_end", "invalid_json_first_rate"),
            "c_unc_stability_valid_only_pct": pct(unc, "valid_only", "stability"),
            "c_unc_stability_end_to_end_pct": pct(unc, "end_to_end", "stability"),
            "c_comp_stability_valid_only_pct": pct(comp, "valid_only", "stability"),
            "c_comp_stability_end_to_end_pct": pct(comp, "end_to_end", "stability"),
            "latency_mean_seconds": unc["latency_mean_seconds"],
            "recorded_cost_usd": round(sum(recorded_costs), 8),
            "cost_attempts_recorded": cost_attempts,
            "total_attempts": total_attempts,
            "judge_max_tokens": manifest["judge_max_tokens"],
        })

    destination = out_dir / "model_comparison.csv"
    with destination.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    return destination


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", type=model_argument, required=True,
                        metavar="ALIAS=MODEL_ID")
    parser.add_argument("--dataset", choices=("calibration", "heldout"), required=True)
    parser.add_argument("--reps", type=positive_int, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--spec-dir", type=Path,
                        default=ROOT / "spec" / "judge_v1_2_pilot")
    parser.add_argument("--judge-max-tokens", type=positive_int, default=1024)
    parser.add_argument("--concurrency", type=positive_int, default=3)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--response-format", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    aliases = [alias for alias, _ in args.model]
    if len(set(aliases)) != len(aliases):
        raise ValueError("Model aliases must be unique")

    spec_dir = args.spec_dir.resolve()
    dataset_name = (
        "pilot_calibration.jsonl"
        if args.dataset == "calibration"
        else "held_out_evaluation.jsonl"
    )
    dataset = spec_dir / dataset_name
    required = [dataset, *(spec_dir / name for name in SPEC_FILES)]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing evaluation inputs: {missing}")

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = out_dir / "batch_manifest.json"
    manifest = {
        "schema_version": "judge-batch-v1",
        "created_at": utc_now(),
        "finished_at": None,
        "dataset": args.dataset,
        "dataset_path": repository_path(dataset),
        "dataset_sha256": sha256(dataset),
        "case_count": sum(bool(line.strip()) for line in dataset.read_text(encoding="utf-8").splitlines()),
        "repetitions": args.reps,
        "logical_outputs_per_model": args.reps * sum(
            bool(line.strip()) for line in dataset.read_text(encoding="utf-8").splitlines()
        ),
        "paired_arms": ["C-UNC", "C-COMP"],
        "temperature": 0.0,
        "judge_max_tokens": args.judge_max_tokens,
        "response_format": args.response_format,
        "timeout_seconds": args.timeout,
        "concurrency_per_model": args.concurrency,
        "spec_dir": repository_path(spec_dir),
        "input_sha256": {path.name: sha256(path) for path in required},
        "models": [
            {"alias": alias, "model_id": model_id, "status": "pending",
             "started_at": None, "finished_at": None, "exit_code": None}
            for alias, model_id in args.model
        ],
        "api_key_stored": False,
        "raw_outputs_versioned": False,
    }
    write_manifest(manifest_path, manifest)

    key = os.environ.get("OPENROUTER_API_KEY") or getpass.getpass("OpenRouter key: ")
    if not key:
        raise ValueError("OPENROUTER_API_KEY is required")
    child_env = {**os.environ, "OPENROUTER_API_KEY": key}
    failures = 0
    for model in manifest["models"]:
        model["status"] = "running"
        model["started_at"] = utc_now()
        write_manifest(manifest_path, manifest)
        print(f"[batch] starting {model['alias']} ({model['model_id']})", flush=True)
        output = out_dir / f"{model['alias']}.csv"
        command = [
            sys.executable,
            str(ROOT / "scripts" / "eval_judge.py"),
            "--dataset", args.dataset,
            "--reps", str(args.reps),
            "--spec-dir", str(spec_dir),
            "--judge-model", model["model_id"],
            "--judge-max-tokens", str(args.judge_max_tokens),
            "--timeout", str(args.timeout),
            "--concurrency", str(args.concurrency),
            "--out", str(output),
        ]
        if args.response_format:
            command.append("--response-format")
        completed = subprocess.run(command, cwd=ROOT, env=child_env, check=False)
        model["exit_code"] = completed.returncode
        model["finished_at"] = utc_now()
        model["status"] = "complete" if completed.returncode == 0 else "failed"
        failures += completed.returncode != 0
        write_manifest(manifest_path, manifest)

    manifest["finished_at"] = utc_now()
    manifest["status"] = "complete" if failures == 0 else "completed_with_failures"
    manifest["comparison_csv"] = write_comparison(out_dir, manifest).name
    manifest["artifact_sha256"] = {
        path.name: sha256(path)
        for path in sorted(out_dir.iterdir())
        if path.is_file() and path != manifest_path
    }
    write_manifest(manifest_path, manifest)
    print(f"[batch] {manifest['status']}: {out_dir}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
