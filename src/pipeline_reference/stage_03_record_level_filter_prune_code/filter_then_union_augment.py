"""Apply filter-first + union-only augmentation after OPS filtering."""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
ENTITY_LABELS = ["CROP", "VAR", "TRT", "GST", "GENE", "QTL", "MRK", "CHR", "BM", "CROSS", "ABS", "BIS"]
DEFAULT_RELATION_LABELS = ["CON", "LOI"]
def load_records(path):
    path = Path(path)
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as z:
            name = "submit.json" if "submit.json" in z.namelist() else z.namelist()[0]
            return json.loads(z.read(name).decode("utf-8"))
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
def dump_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
def load_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows
def write_zip(json_path, zip_path):
    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(json_path, "submit.json")
def sha1_id(parts):
    raw = "\0".join(str(p) for p in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
def entity_key(entity):
    return int(entity["start"]), int(entity["end"]), str(entity["label"])
def entity_text_key(entity):
    return str(entity.get("text", "")), str(entity.get("label", ""))
def relation_key(rel):
    return (
        int(rel["head_start"]),
        int(rel["head_end"]),
        str(rel["head_type"]),
        int(rel["tail_start"]),
        int(rel["tail_end"]),
        str(rel["tail_type"]),
        str(rel["label"]),
    )
def relation_endpoint_keys(rel):
    return (
        (int(rel["head_start"]), int(rel["head_end"]), str(rel["head_type"])),
        (int(rel["tail_start"]), int(rel["tail_end"]), str(rel["tail_type"])),
    )
def relation_id(record_index, rel):
    return sha1_id(
        [
            "R",
            record_index,
            rel.get("head_start"),
            rel.get("head_end"),
            rel.get("head_type"),
            rel.get("tail_start"),
            rel.get("tail_end"),
            rel.get("tail_type"),
            rel.get("label"),
            rel.get("head", ""),
            rel.get("tail", ""),
        ]
    )
def entity_id(record_index, ent):
    return sha1_id(
        [
            "E",
            record_index,
            ent.get("start"),
            ent.get("end"),
            ent.get("label"),
            ent.get("text", ""),
        ]
    )
def valid_entity(ent, text):
    if not isinstance(ent, dict):
        return False
    if str(ent.get("label", "")) not in ENTITY_LABELS:
        return False
    start, end, value = ent.get("start"), ent.get("end"), str(ent.get("text", ""))
    return (
        isinstance(start, int)
        and isinstance(end, int)
        and 0 <= start < end <= len(text)
        and text[start:end] == value
    )
def resolve_corrected_entity(obj, text):
    if not isinstance(obj, dict):
        return None
    label = str(obj.get("label", ""))
    value = str(obj.get("text", ""))
    if label not in ENTITY_LABELS or not value:
        return None
    start, end = obj.get("start"), obj.get("end")
    if not (
        isinstance(start, int)
        and isinstance(end, int)
        and 0 <= start < end <= len(text)
        and text[start:end] == value
    ):
        start = text.find(value)
        if start < 0:
            return None
        end = start + len(value)
    ent = {"start": start, "end": end, "text": value, "label": label}
    return ent if valid_entity(ent, text) else None
def endpoints_exist(rel, entity_keys):
    head, tail = relation_endpoint_keys(rel)
    return head in entity_keys and tail in entity_keys
def normalize_action(action):
    action = str(action or "").strip().lower()
    return action if action in {"keep", "drop", "fix"} else "drop"
def confidence(decision):
    try:
        return float(decision.get("confidence", 0) or 0)
    except Exception:
        return 0.0
def collect_decisions(path):
    latest = {}
    for row in load_jsonl(path):
        for decision in row.get("decisions", []) or []:
            cid = str(decision.get("id") or decision.get("candidate_id") or "").strip()
            if cid:
                latest[cid] = decision
    out = defaultdict(list)
    for cid, decision in latest.items():
        out[cid].append(decision)
    return out
def choose_decision(decisions, keep_threshold, fix_threshold, missing_action):
    if not decisions:
        return {"action": missing_action, "confidence": 0.0, "reason": "missing judgment"}
    fixes = [
        d
        for d in decisions
        if normalize_action(d.get("action")) == "fix" and confidence(d) >= fix_threshold
    ]
    keeps = [
        d
        for d in decisions
        if normalize_action(d.get("action")) == "keep" and confidence(d) >= keep_threshold
    ]
    drops = [d for d in decisions if normalize_action(d.get("action")) == "drop"]
    if fixes:
        return max(fixes, key=confidence)
    if keeps:
        return max(keeps, key=confidence)
    if drops:
        return max(drops, key=confidence)
    return max(decisions, key=confidence)
def make_entity_candidates(base, pool, labels):
    allow = set(labels or ENTITY_LABELS)
    rows, by_label = [], Counter()
    for i, (base_row, pool_row) in enumerate(zip(base, pool)):
        existing = {entity_key(e) for e in base_row.get("entities", []) or []}
        text_seen = {entity_text_key(e) for e in base_row.get("entities", []) or []}
        seen = set(existing)
        for ent in pool_row.get("entities", []) or []:
            label = str(ent.get("label", ""))
            if label not in allow:
                continue
            if entity_key(ent) in seen or entity_text_key(ent) in text_seen:
                continue
            row = {
                "id": entity_id(i, ent),
                "kind": "entity",
                "record_index": i,
                "text": str(ent.get("text", "")),
                "start": int(ent["start"]),
                "end": int(ent["end"]),
                "label": label,
            }
            rows.append(row)
            by_label[label] += 1
            seen.add(entity_key(ent))
    return rows, by_label
def make_relation_candidates(base, pool, relation_labels):
    allow = set(relation_labels or DEFAULT_RELATION_LABELS)
    rows, by_label, skipped = [], Counter(), Counter()
    for i, (base_row, pool_row) in enumerate(zip(base, pool)):
        existing = {relation_key(r) for r in base_row.get("relations", []) or []}
        entity_keys = {entity_key(e) for e in base_row.get("entities", []) or []}
        seen = set(existing)
        for rel in pool_row.get("relations", []) or []:
            label = str(rel.get("label", ""))
            if label not in allow:
                continue
            key = relation_key(rel)
            if key in seen:
                continue
            seen.add(key)
            if not endpoints_exist(rel, entity_keys):
                skipped[label] += 1
                continue
            row = {
                "id": relation_id(i, rel),
                "kind": "relation",
                "record_index": i,
                "label": label,
                "head": rel.get("head", ""),
                "head_start": int(rel["head_start"]),
                "head_end": int(rel["head_end"]),
                "head_type": str(rel["head_type"]),
                "tail": rel.get("tail", ""),
                "tail_start": int(rel["tail_start"]),
                "tail_end": int(rel["tail_end"]),
                "tail_type": str(rel["tail_type"]),
            }
            rows.append(row)
            by_label[label] += 1
    return rows, by_label, skipped
def add_entities(base, candidates, decisions, keep_threshold, fix_threshold, missing_action):
    additions = [[] for _ in base]
    stats, by_label, applied_rows = Counter(), Counter(), []
    for row in candidates:
        rec = base[row["record_index"]]
        decision = choose_decision(
            decisions.get(row["id"], []),
            keep_threshold,
            fix_threshold,
            missing_action,
        )
        action = normalize_action(decision.get("action"))
        ent = None
        if action == "keep":
            ent = {"start": row["start"], "end": row["end"], "text": row["text"], "label": row["label"]}
        elif action == "fix":
            ent = resolve_corrected_entity(decision.get("corrected"), rec.get("text", ""))
        if ent and valid_entity(ent, rec.get("text", "")):
            additions[row["record_index"]].append(ent)
            stats["accepted"] += 1
            by_label[ent["label"]] += 1
        else:
            stats["rejected"] += 1
        applied_rows.append(
            {
                **row,
                "action": action,
                "confidence": confidence(decision),
                "applied": bool(ent),
            }
        )
    out = []
    changed = 0
    for rec, ents in zip(base, additions):
        row = dict(rec)
        existing = {entity_key(e) for e in row.get("entities", []) or []}
        text_seen = {entity_text_key(e) for e in row.get("entities", []) or []}
        merged = list(row.get("entities", []) or [])
        for ent in ents:
            if entity_key(ent) in existing or entity_text_key(ent) in text_seen:
                continue
            merged.append(ent)
            existing.add(entity_key(ent))
            text_seen.add(entity_text_key(ent))
            changed += 1
        row["entities"] = sorted(merged, key=lambda e: (int(e["start"]), int(e["end"]), str(e["label"])))
        out.append(row)
    stats["changed_entities"] = changed
    return out, stats, by_label, applied_rows
def row_to_relation(row):
    return {
        "head": row["head"],
        "head_start": row["head_start"],
        "head_end": row["head_end"],
        "head_type": row["head_type"],
        "tail": row["tail"],
        "tail_start": row["tail_start"],
        "tail_end": row["tail_end"],
        "tail_type": row["tail_type"],
        "label": row["label"],
    }
def resolve_corrected_relation(corrected, entities, allowed_labels):
    if not isinstance(corrected, dict):
        return None
    label = str(corrected.get("label", ""))
    if label not in allowed_labels:
        return None
    head, head_type = str(corrected.get("head", "")), str(corrected.get("head_type", ""))
    tail, tail_type = str(corrected.get("tail", "")), str(corrected.get("tail_type", ""))
    heads = [e for e in entities if e.get("text") == head and e.get("label") == head_type]
    tails = [e for e in entities if e.get("text") == tail and e.get("label") == tail_type]
    if len(heads) != 1 or len(tails) != 1:
        return None
    h, t = heads[0], tails[0]
    return {
        "head": head,
        "head_start": int(h["start"]),
        "head_end": int(h["end"]),
        "head_type": head_type,
        "tail": tail,
        "tail_start": int(t["start"]),
        "tail_end": int(t["end"]),
        "tail_type": tail_type,
        "label": label,
    }
def add_relations(
    base,
    candidates,
    decisions,
    keep_threshold,
    fix_threshold,
    missing_action,
    allowed_labels,
    apply_fix,
):
    additions = [[] for _ in base]
    stats, by_label, applied_rows = Counter(), Counter(), []
    for row in candidates:
        rec = base[row["record_index"]]
        decision = choose_decision(
            decisions.get(row["id"], []),
            keep_threshold,
            fix_threshold,
            missing_action,
        )
        action = normalize_action(decision.get("action"))
        rel = None
        entity_keys = {entity_key(e) for e in rec.get("entities", []) or []}
        if action == "keep":
            rel = row_to_relation(row)
        elif action == "fix" and apply_fix:
            rel = resolve_corrected_relation(
                decision.get("corrected"),
                rec.get("entities", []) or [],
                allowed_labels,
            )
        if rel and rel["label"] in allowed_labels and endpoints_exist(rel, entity_keys):
            additions[row["record_index"]].append(rel)
            stats["accepted"] += 1
            by_label[rel["label"]] += 1
        else:
            stats["rejected"] += 1
        applied_rows.append(
            {
                **row,
                "action": action,
                "confidence": confidence(decision),
                "applied": bool(rel),
            }
        )
    out = []
    changed = 0
    for rec, rels in zip(base, additions):
        row = dict(rec)
        existing = {relation_key(r) for r in row.get("relations", []) or []}
        merged = list(row.get("relations", []) or [])
        for rel in rels:
            if relation_key(rel) in existing:
                continue
            merged.append(rel)
            existing.add(relation_key(rel))
            changed += 1
        row["relations"] = sorted(
            merged,
            key=lambda r: (int(r["head_start"]), int(r["head_end"]), int(r["tail_start"]), int(r["tail_end"]), str(r["label"])),
        )
        out.append(row)
    stats["changed_relations"] = changed
    return out, stats, by_label, applied_rows
def postprocess_if_available(records, solution_src):
    if not solution_src:
        return records
    src = str(Path(solution_src).resolve())
    sys.path.insert(0, src)
    try:
        from align import postprocess  # type: ignore
        return [
            postprocess(
                {"entities": r.get("entities", []) or [], "relations": r.get("relations", []) or []},
                r["text"],
            )
            for r in records
        ]
    finally:
        if sys.path[0] == src:
            sys.path.pop(0)
def count_records(records):
    ent, rel, empty, bad = Counter(), Counter(), 0, 0
    for rec in records:
        entity_keys = {entity_key(e) for e in rec.get("entities", []) or []}
        if not rec.get("entities") and not rec.get("relations"):
            empty += 1
        for e in rec.get("entities", []) or []:
            ent[str(e.get("label", ""))] += 1
        for r in rec.get("relations", []) or []:
            rel[str(r.get("label", ""))] += 1
            if not endpoints_exist(r, entity_keys):
                bad += 1
    return {
        "records": len(records),
        "entities": sum(ent.values()),
        "relations": sum(rel.values()),
        "empty": empty,
        "bad_relation_endpoints": bad,
        "entity_labels": dict(sorted(ent.items())),
        "relation_labels": dict(sorted(rel.items())),
    }
def prepare(args):
    base = load_records(args.base)
    pool = load_records(args.pool)
    if len(base) != len(pool):
        raise SystemExit(f"base/pool record count mismatch: {len(base)} vs {len(pool)}")
    out_dir = Path(args.out_dir)
    entity_rows, entity_stats = make_entity_candidates(base, pool, args.entity_labels)
    relation_rows, relation_stats, skipped = make_relation_candidates(base, pool, args.relation_labels)
    dump_json(out_dir / "entity_candidates.json", entity_rows)
    dump_json(out_dir / "relation_candidates.json", relation_rows)
    summary = {
        "base": str(args.base),
        "pool": str(args.pool),
        "base_stats": count_records(base),
        "pool_stats": count_records(pool),
        "entity_candidates": len(entity_rows),
        "entity_candidate_by_label": dict(sorted(entity_stats.items())),
        "relation_candidates": len(relation_rows),
        "relation_candidate_by_label": dict(sorted(relation_stats.items())),
        "relation_skipped_missing_endpoint": dict(sorted(skipped.items())),
    }
    dump_json(out_dir / "prepare_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
def load_candidate_file(path, fallback):
    path = Path(path) if path else None
    if path and path.exists():
        return load_records(path)
    return fallback
def apply(args):
    base = load_records(args.base)
    pool = load_records(args.pool)
    if not args.output_json or not args.summary:
        raise SystemExit("--output_json and --summary are required for apply")
    if len(base) != len(pool):
        raise SystemExit(f"base/pool record count mismatch: {len(base)} vs {len(pool)}")
    out_dir = Path(args.out_dir)
    entity_rows = load_candidate_file(
        args.entity_candidates,
        make_entity_candidates(base, pool, args.entity_labels)[0],
    )
    entity_decisions = collect_decisions(args.entity_judgments) if args.entity_judgments else defaultdict(list)
    base_plus_entities, ent_stats, ent_by_label, ent_applied = add_entities(
        base,
        entity_rows,
        entity_decisions,
        args.entity_keep_threshold,
        args.entity_fix_threshold,
        args.missing_entity_action,
    )
    relation_rows = load_candidate_file(
        args.relation_candidates,
        make_relation_candidates(base_plus_entities, pool, args.relation_labels)[0],
    )
    relation_decisions = collect_decisions(args.relation_judgments) if args.relation_judgments else defaultdict(list)
    allowed_relations = set(args.output_relation_labels or args.relation_labels or DEFAULT_RELATION_LABELS)
    augmented, rel_stats, rel_by_label, rel_applied = add_relations(
        base_plus_entities,
        relation_rows,
        relation_decisions,
        args.relation_keep_threshold,
        args.relation_fix_threshold,
        args.missing_relation_action,
        allowed_relations,
        args.apply_relation_fix,
    )
    augmented = postprocess_if_available(augmented, args.solution_src)
    dump_json(args.output_json, augmented)
    if args.output_zip:
        write_zip(args.output_json, args.output_zip)
    dump_json(out_dir / "applied_entity_decisions.json", ent_applied)
    dump_json(out_dir / "applied_relation_decisions.json", rel_applied)
    summary = {
        "base": str(args.base),
        "pool": str(args.pool),
        "output_json": str(args.output_json),
        "output_zip": str(args.output_zip) if args.output_zip else None,
        "thresholds": {
            "entity_keep": args.entity_keep_threshold,
            "entity_fix": args.entity_fix_threshold,
            "relation_keep": args.relation_keep_threshold,
            "relation_fix": args.relation_fix_threshold,
        },
        "missing_actions": {"entity": args.missing_entity_action, "relation": args.missing_relation_action},
        "base_stats": count_records(base),
        "entity_augmented_stats": count_records(base_plus_entities),
        "output_stats": count_records(augmented),
        "entity_apply": dict(sorted(ent_stats.items())),
        "entity_added_by_label": dict(sorted(ent_by_label.items())),
        "relation_apply": dict(sorted(rel_stats.items())),
        "relation_added_by_label": dict(sorted(rel_by_label.items())),
    }
    dump_json(args.summary, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "apply"])
    parser.add_argument("--base", required=True, help="Filtered base json/zip.")
    parser.add_argument("--pool", required=True, help="Larger union/pool json/zip.")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--entity_labels", nargs="+", default=ENTITY_LABELS)
    parser.add_argument("--relation_labels", nargs="+", default=DEFAULT_RELATION_LABELS)
    parser.add_argument("--output_relation_labels", nargs="+", default=DEFAULT_RELATION_LABELS)
    parser.add_argument("--entity_candidates")
    parser.add_argument("--relation_candidates")
    parser.add_argument("--entity_judgments")
    parser.add_argument("--relation_judgments")
    parser.add_argument("--entity_keep_threshold", type=float, default=0.92)
    parser.add_argument("--entity_fix_threshold", type=float, default=0.92)
    parser.add_argument("--relation_keep_threshold", type=float, default=0.92)
    parser.add_argument("--relation_fix_threshold", type=float, default=0.92)
    parser.add_argument("--missing_entity_action", choices=["keep", "drop"], default="drop")
    parser.add_argument("--missing_relation_action", choices=["keep", "drop"], default="drop")
    parser.add_argument("--apply_relation_fix", action="store_true")
    parser.add_argument("--solution_src")
    parser.add_argument("--output_json")
    parser.add_argument("--output_zip", default=None)
    parser.add_argument("--summary")
    return parser.parse_args()
def main():
    args = parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    if args.command == "prepare":
        prepare(args)
        return
    if args.command == "apply":
        apply(args)
        return
    raise SystemExit(f"unsupported command: {args.command}")
if __name__ == "__main__":
    main()
