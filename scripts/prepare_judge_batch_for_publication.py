"""Sanitiza un lote held-out y conserva hashes de sus artefactos privados."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_judge_batch import repository_path, sha256, write_manifest
from scripts.verify_judge_batch import verify


def sanitize_csv(path: Path) -> bool:
    with path.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    if "reason" not in fields:
        return False
    fields.remove("reason")
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch_dir", type=Path)
    args = parser.parse_args()
    root = args.batch_dir.resolve()
    manifest_path = root / "batch_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    sanitized = sum(sanitize_csv(path) for path in root.glob("*.csv"))
    for field in ("dataset_path", "spec_dir"):
        path = Path(manifest[field])
        manifest[field] = repository_path(path if path.is_absolute() else ROOT / path)
    manifest.pop("cost_account_usage_before_usd", None)
    manifest.pop("cost_account_usage_after_usd", None)
    manifest["raw_outputs_versioned"] = False
    write_manifest(manifest_path, manifest)

    integrity = verify(root)
    (root / "integrity_report.json").write_text(
        json.dumps(integrity, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    artifacts = [path for path in sorted(root.iterdir()) if path.is_file() and path != manifest_path]
    manifest["artifact_sha256"] = {path.name: sha256(path) for path in artifacts}
    manifest["publication_files"] = [
        manifest_path.name,
        *(path.name for path in artifacts if not path.name.endswith(".judge_outputs.jsonl")),
    ]
    manifest["private_artifacts"] = [
        {"name": path.name, "sha256": manifest["artifact_sha256"][path.name], "bytes": path.stat().st_size}
        for path in artifacts
        if path.name.endswith(".judge_outputs.jsonl")
    ]
    write_manifest(manifest_path, manifest)
    print(
        f"sanitized_csv={sanitized} integrity={integrity['overall_pass']} "
        f"publication_files={len(manifest['publication_files'])} private_raw={len(manifest['private_artifacts'])}"
    )
    return 0 if integrity["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
