"""Recompute deception metrics offline and finalize an interrupted batch."""
from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_deception_batch import (
    refresh_artifacts,
    write_comparison,
    write_manifest,
    write_variant_comparison,
)

ROOT = Path(__file__).resolve().parents[1]


def evaluate(alias: str, out_dir: Path, bootstrap: int) -> tuple[str, str | None]:
    csv_path = out_dir / f"{alias}.paired_results.csv"
    metrics_path = out_dir / f"{alias}.metrics.txt"
    if not csv_path.is_file():
        return alias, f"missing {csv_path.name}"
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "eval_deception.py"),
         str(csv_path), "--bootstrap", str(bootstrap)],
        cwd=ROOT, text=True, encoding="utf-8", capture_output=True, check=False,
    )
    metrics_path.write_text(result.stdout + result.stderr, encoding="utf-8")
    return alias, None if result.returncode == 0 else f"metrics exit {result.returncode}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    out_dir = args.out_dir.resolve()
    manifest_path = out_dir / "batch_manifest.json"
    import json
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_alias = {model["alias"]: model for model in manifest["models"]}

    failures = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(evaluate, alias, out_dir, args.bootstrap): alias
            for alias in by_alias
        }
        for future in as_completed(futures):
            alias, error = future.result()
            if error:
                failures[alias] = error
                print(f"[finalize] {alias}: {error}", flush=True)
            else:
                model = by_alias[alias]
                model["status"] = "complete"
                model.pop("error", None)
                model["metrics_bootstrap"] = args.bootstrap
                print(f"[finalize] {alias}: metrics complete", flush=True)

    complete = [model for model in manifest["models"] if model["status"] == "complete"]
    if complete:
        write_comparison(out_dir, complete)
        write_variant_comparison(out_dir, complete)
    manifest["metrics_bootstrap"] = args.bootstrap
    manifest["metrics_bootstrap_cluster"] = "scenario_id"
    manifest["finalized_at"] = datetime.now(timezone.utc).isoformat()
    manifest["status"] = "complete" if not failures and len(complete) == len(by_alias) else "completed_with_failures"
    refresh_artifacts(manifest, out_dir, manifest_path)
    write_manifest(manifest_path, manifest)
    print(f"[finalize] batch status: {manifest['status']}")
    return 1 if manifest["status"] != "complete" else 0


if __name__ == "__main__":
    raise SystemExit(main())
