import argparse
import json
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def normalize_action(action):
    action = str(action or "").strip().lower()
    return action if action in {"keep", "drop", "fix"} else "keep"


def confidence(decision):
    try:
        return float(decision.get("confidence", 0) or 0)
    except Exception:
        return 0.0


def valid_entity(obj):
    return isinstance(obj, dict) and obj.get("text") and obj.get("label")


def collect_decisions(rows):
    by_record = defaultdict(lambda: defaultdict(list))
    for row in rows:
        idx = row.get("record_index")
        if idx is None:
            continue
        for decision in row.get("decisions", []) or []:
            cid = str(decision.get("id", "")).strip()
            if cid:
                by_record[int(idx)][cid].append(decision)
    return by_record


def aggregate_entity(decisions, policy):
    if not decisions:
        return {"action": "keep"}
    actions = [normalize_action(d.get("action")) for d in decisions]
    drops = [d for d in decisions if normalize_action(d.get("action")) == "drop"]
    fixes = [
        d
        for d in decisions
        if normalize_action(d.get("action")) == "fix" and valid_entity(d.get("corrected"))
    ]
    if policy == "entity_any_drop":
        if drops:
            return max(drops, key=confidence)
    elif policy == "entity_drop095":
        strong = [d for d in drops if confidence(d) >= 0.95]
        if strong:
            return max(strong, key=confidence)
    elif policy == "entity_majority_drop":
        counts = Counter(actions)
        if counts["drop"] > max(counts["keep"], counts["fix"]):
            return max(drops, key=confidence)
    else:
        raise ValueError(f"unknown entity policy: {policy}")
    if fixes:
        return max(fixes, key=confidence)
    return {"action": "keep"}


def relation_should_drop(decisions, threshold):
    return any(
        normalize_action(decision.get("action")) == "drop"
        and confidence(decision) >= threshold
        for decision in decisions
    )


def apply_policy(record, decisions, postprocess, entity_policy, rel_drop_threshold):
    entities = []
    relations = []
    stats = {"entity_drop": 0, "entity_fix": 0, "relation_drop": 0, "relation_fix": 0}

    for i, entity in enumerate(record.get("entities", []) or []):
        decision = aggregate_entity(decisions.get(f"E{i}", []), entity_policy)
        action = normalize_action(decision.get("action"))
        if action == "drop":
            stats["entity_drop"] += 1
            continue
        if action == "fix" and valid_entity(decision.get("corrected")):
            corrected = dict(decision["corrected"])
            entities.append({
                "text": corrected.get("text", ""),
                "label": corrected.get("label", ""),
                "start": corrected.get("start"),
                "end": corrected.get("end"),
            })
            stats["entity_fix"] += 1
            continue
        entities.append(entity)

    for i, relation in enumerate(record.get("relations", []) or []):
        if relation_should_drop(decisions.get(f"R{i}", []), rel_drop_threshold):
            stats["relation_drop"] += 1
            continue
        relations.append(relation)

    return postprocess({"entities": entities, "relations": relations}, record["text"]), stats


def count_items(records):
    return {
        "records": len(records),
        "entities": sum(len(r.get("entities", []) or []) for r in records),
        "relations": sum(len(r.get("relations", []) or []) for r in records),
        "empty": sum(1 for r in records if not r.get("entities") and not r.get("relations")),
    }


def write_zip(json_path, zip_path):
    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(json_path, "submit.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True)
    parser.add_argument("--judgments", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_zip", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--entity_policy", default="entity_any_drop")
    parser.add_argument("--rel_drop_threshold", type=float, default=0.95)
    parser.add_argument("--solution_src", required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(Path(args.solution_src).resolve()))
    from align import postprocess  # noqa: E402

    pred = load_json(args.pred)
    decisions_by_record = collect_decisions(load_jsonl(args.judgments))
    out = list(pred)
    stats = Counter()
    for idx, decisions in decisions_by_record.items():
        if 0 <= idx < len(out):
            out[idx], change = apply_policy(
                out[idx],
                decisions,
                postprocess,
                entity_policy=args.entity_policy,
                rel_drop_threshold=args.rel_drop_threshold,
            )
            stats.update(change)

    dump_json(args.output_json, out)
    write_zip(args.output_json, args.output_zip)
    summary = {
        "input": count_items(pred),
        "output": count_items(out),
        "judged_records": len(decisions_by_record),
        "entity_policy": args.entity_policy,
        "rel_drop_threshold": args.rel_drop_threshold,
        "applied": dict(stats),
        "output_json": args.output_json,
        "output_zip": args.output_zip,
    }
    dump_json(args.summary, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
