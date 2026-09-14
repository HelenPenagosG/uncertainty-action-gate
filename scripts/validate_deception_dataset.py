#!/usr/bin/env python3
import argparse, json, sys
from collections import defaultdict
from pathlib import Path

JUDGE_KEYS = [
    "operator_authorization","authorized_resources","explicit_restrictions",
    "current_context","recent_action_history","proposed_action","action_justification"
]
FORBIDDEN_NAMES = {
    "oracle_scope","actual_scope","expected_compatibility","expected_uncertainty",
    "behavior","primary_behavior","behavior_labels","adjudication","probative_note",
    "variant","pair_id","template_family","private_labels","provenance","actor_evidence"
}

def load_jsonl(path):
    rows=[]
    for i,line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(),1):
        if not line.strip(): continue
        try: rows.append(json.loads(line))
        except Exception as e: raise SystemExit(f"{path}:{i}: invalid JSON: {e}")
    return rows

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("jsonl", nargs="+")
    ap.add_argument("--schema", required=False)
    args=ap.parse_args()

    schema=None
    if args.schema:
        schema=json.loads(Path(args.schema).read_text(encoding="utf-8"))
        try:
            import jsonschema
        except ImportError:
            raise SystemExit("jsonschema is required when --schema is supplied")

    rows=[]; errors=[]
    for path in args.jsonl:
        batch=load_jsonl(path); rows.extend(batch)
        for r in batch:
            cid=r.get("case_id")
            if schema is not None:
                try: jsonschema.validate(r,schema)
                except Exception as e:
                    errors.append(f"{cid}: schema validation failed: {getattr(e,'message',e)}")

            ji=r.get("judge_input",{})
            if set(ji)!=set(JUDGE_KEYS):
                errors.append(f"{cid}: judge_input keys differ from required seven fields")

            serialized=json.dumps(ji,sort_keys=True)
            for forbidden in FORBIDDEN_NAMES:
                if f'"{forbidden}"' in serialized:
                    errors.append(f"{cid}: leaked private field name into judge_input: {forbidden}")

            pl=r.get("private_labels",{})
            beh=pl.get("behavior",{})
            if pl.get("primary_behavior")!=beh.get("primary"):
                errors.append(f"{cid}: primary_behavior is not synchronized with behavior.primary")
            if pl.get("behavior_labels")!=beh.get("labels"):
                errors.append(f"{cid}: behavior_labels is not synchronized with behavior.labels")

            evidence_ids={e.get("id") for e in r.get("actor_evidence",[])}
            refs=set(beh.get("evidence_refs",[]))
            if refs-evidence_ids:
                errors.append(f"{cid}: behavior evidence refs not found: {sorted(refs-evidence_ids)}")

            if r.get("provenance",{}).get("origin") not in {"prepared","actor_generated","explicitly_induced"}:
                errors.append(f"{cid}: invalid provenance.origin")

    case_ids=[r.get("case_id") for r in rows]
    if len(case_ids)!=len(set(case_ids)):
        errors.append("duplicate case_id values found")

    by_pair=defaultdict(list)
    for r in rows: by_pair[r.get("pair_id")].append(r)

    for pid,group in by_pair.items():
        if not pid: continue
        if len(group)!=2:
            errors.append(f"{pid}: expected exactly 2 prepared variants, found {len(group)}")
            continue
        variants={r.get("variant") for r in group}
        if variants!={"factual","deceptive"}:
            errors.append(f"{pid}: expected factual/deceptive variants, found {variants}")
        if len({r.get("scenario_id") for r in group})!=1:
            errors.append(f"{pid}: pair spans multiple scenario_ids")
        if len({r.get("split") for r in group})!=1:
            errors.append(f"{pid}: pair spans multiple splits")
        if len({r.get("provenance",{}).get("origin") for r in group})!=1:
            errors.append(f"{pid}: pair mixes provenance origins")

        base=group[0]
        for other in group[1:]:
            for key in JUDGE_KEYS:
                if key=="action_justification": continue
                if base["judge_input"].get(key)!=other["judge_input"].get(key):
                    errors.append(f"{pid}: paired field changed: {key}")
            if base.get("actor_evidence")!=other.get("actor_evidence"):
                errors.append(f"{pid}: actor_evidence changed across variants")
            for key in ["actual_scope","oracle_scope","expected_compatibility","expected_uncertainty"]:
                if base["private_labels"].get(key)!=other["private_labels"].get(key):
                    errors.append(f"{pid}: private pair invariant changed: {key}")

    scenario_splits=defaultdict(set); family_splits=defaultdict(set)
    for r in rows:
        scenario_splits[r.get("scenario_id")].add(r.get("split"))
        family_splits[r.get("template_family")].add(r.get("split"))
    for sid,splits in scenario_splits.items():
        if len(splits)>1: errors.append(f"{sid}: scenario shared across splits: {sorted(splits)}")
    for fam,splits in family_splits.items():
        if fam and len(splits)>1: errors.append(f"{fam}: template family shared across splits: {sorted(splits)}")

    if errors:
        print("VALIDATION FAILED")
        for e in errors: print("-",e)
        sys.exit(1)
    print(f"VALIDATION OK: {len(rows)} records, {len(by_pair)} pairs")

if __name__=="__main__":
    main()
