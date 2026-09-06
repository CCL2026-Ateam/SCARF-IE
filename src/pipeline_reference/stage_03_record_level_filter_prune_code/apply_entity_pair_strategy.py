import argparse
import json
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_PRESETS = {
    "report_anydrop_rel095": {
        "entity_policy": "any_drop",
        "entity_drop_threshold": 0.0,
        "apply_entity_fix": True,
        "entity_fix_threshold": 0.0,
        "relation_policy": "drop_threshold",
        "relation_drop_threshold": 0.95,
        "apply_relation_fix": False,
        "protect_relation_endpoints": False,
        "noempty_fallback": True,
    },
    "safe_anydrop_rel098": {
        "entity_policy": "any_drop",
        "entity_drop_threshold": 0.0,
        "apply_entity_fix": True,
        "entity_fix_threshold": 0.0,
        "relation_policy": "drop_threshold",
        "relation_drop_threshold": 0.98,
        "apply_relation_fix": False,
        "protect_relation_endpoints": False,
        "noempty_fallback": True,
    },
    "endpoint_safe_rel098": {
        "entity_policy": "any_drop",
        "entity_drop_threshold": 0.0,
        "apply_entity_fix": True,
        "entity_fix_threshold": 0.0,
        "relation_policy": "drop_threshold",
        "relation_drop_threshold": 0.98,
        "apply_relation_fix": False,
        "protect_relation_endpoints": True,
        "noempty_fallback": True,
    },
    "entity_drop095_rel098": {
        "entity_policy": "drop_threshold",
        "entity_drop_threshold": 0.95,
        "apply_entity_fix": True,
        "entity_fix_threshold": 0.0,
        "relation_policy": "drop_threshold",
        "relation_drop_threshold": 0.98,
        "apply_relation_fix": False,
        "protect_relation_endpoints": False,
        "noempty_fallback": True,
    },
    "audit_report_drop095": {
        "entity_policy": "drop_threshold",
        "entity_drop_threshold": 0.95,
        "apply_entity_fix": True,
        "entity_fix_threshold": 0.0,
        "relation_policy": "drop_threshold",
        "relation_drop_threshold": 0.95,
        "apply_relation_fix": False,
        "protect_relation_endpoints": False,
        "noempty_fallback": True,
    },
    "keep_relations_entity_drop095": {
        "entity_policy": "drop_threshold",
        "entity_drop_threshold": 0.95,
        "apply_entity_fix": True,
        "entity_fix_threshold": 0.0,
        "relation_policy": "keep_all",
        "relation_drop_threshold": 1.01,
        "apply_relation_fix": False,
        "protect_relation_endpoints": True,
        "noempty_fallback": True,
    },
}


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


def valid_relation(obj):
    return (
        isinstance(obj, dict)
        and obj.get("head")
        and obj.get("head_type")
        and obj.get("tail")
        and obj.get("tail_type")
        and obj.get("label")
    )


def entity_key(entity):
    return entity.get("text"), entity.get("label")


def relation_endpoint_keys(relation):
    return {
        (relation.get("head"), relation.get("head_type")),
        (relation.get("tail"), relation.get("tail_type")),
    }


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


def strongest(decisions, action=None, threshold=None, require_valid=None):
    out = []
    for decision in decisions:
        if action is not None and normalize_action(decision.get("action")) != action:
            continue
        if threshold is not None and confidence(decision) < threshold:
            continue
        if require_valid == "entity" and not valid_entity(decision.get("corrected")):
            continue
        if require_valid == "relation" and not valid_relation(decision.get("corrected")):
            continue
        out.append(decision)
    if not out:
        return None
    return max(out, key=confidence)


def aggregate_entity(decisions, strategy):
    if not decisions:
        return {"action": "keep"}

    policy = strategy["entity_policy"]
    drops = [d for d in decisions if normalize_action(d.get("action")) == "drop"]
    fixes = [
        d
        for d in decisions
        if normalize_action(d.get("action")) == "fix" and valid_entity(d.get("corrected"))
    ]

    if policy == "keep_all":
        pass
    elif policy == "any_drop":
        if drops:
            return max(drops, key=confidence)
    elif policy == "drop_threshold":
        decision = strongest(drops, threshold=strategy["entity_drop_threshold"])
        if decision:
            return decision
    elif policy == "majority_drop":
        counts = Counter(normalize_action(d.get("action")) for d in decisions)
        if counts["drop"] > max(counts["keep"], counts["fix"]) and drops:
            return max(drops, key=confidence)
    elif policy == "majority_or_threshold":
        counts = Counter(normalize_action(d.get("action")) for d in decisions)
        decision = strongest(drops, threshold=strategy["entity_drop_threshold"])
        if decision or (counts["drop"] > max(counts["keep"], counts["fix"]) and drops):
            return decision or max(drops, key=confidence)
    else:
        raise ValueError(f"unknown entity_policy: {policy}")

    if strategy["apply_entity_fix"]:
        decision = strongest(
            fixes,
            threshold=strategy["entity_fix_threshold"],
            require_valid="entity",
        )
        if decision:
            return decision
    return {"action": "keep"}


def aggregate_relation(decisions, strategy):
    if not decisions:
        return {"action": "keep"}
    if strategy["relation_policy"] == "keep_all":
        return {"action": "keep"}
    if strategy["relation_policy"] == "drop_threshold":
        decision = strongest(
            decisions,
            action="drop",
            threshold=strategy["relation_drop_threshold"],
        )
        if decision:
            return decision
    elif strategy["relation_policy"] != "drop_threshold":
        raise ValueError(f"unknown relation_policy: {strategy['relation_policy']}")

    if strategy["apply_relation_fix"]:
        decision = strongest(
            decisions,
            action="fix",
            threshold=strategy["relation_fix_threshold"],
            require_valid="relation",
        )
        if decision:
            return decision
    return {"action": "keep"}


def collect_kept_relation_endpoint_keys(record, decisions, strategy):
    endpoints = set()
    for i, relation in enumerate(record.get("relations", []) or []):
        decision = aggregate_relation(decisions.get(f"R{i}", []), strategy)
        if normalize_action(decision.get("action")) == "drop":
            continue
        if normalize_action(decision.get("action")) == "fix" and valid_relation(decision.get("corrected")):
            endpoints.update(relation_endpoint_keys(decision["corrected"]))
        else:
            endpoints.update(relation_endpoint_keys(relation))
    return endpoints


def apply_strategy_to_record(record, original_record, decisions, postprocess, strategy):
    endpoint_keys = (
        collect_kept_relation_endpoint_keys(record, decisions, strategy)
        if strategy["protect_relation_endpoints"]
        else set()
    )
    entities = []
    relations = []
    stats = Counter()

    for i, entity in enumerate(record.get("entities", []) or []):
        decision = aggregate_entity(decisions.get(f"E{i}", []), strategy)
        action = normalize_action(decision.get("action"))
        protected = entity_key(entity) in endpoint_keys
        if protected and action in {"drop", "fix"}:
            stats["entity_protected"] += 1
            entities.append(entity)
            continue
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
        decision = aggregate_relation(decisions.get(f"R{i}", []), strategy)
        action = normalize_action(decision.get("action"))
        if action == "drop":
            stats["relation_drop"] += 1
            continue
        if action == "fix" and valid_relation(decision.get("corrected")):
            corrected = dict(decision["corrected"])
            relations.append({
                "head": corrected.get("head", ""),
                "head_type": corrected.get("head_type", ""),
                "tail": corrected.get("tail", ""),
                "tail_type": corrected.get("tail_type", ""),
                "label": corrected.get("label", ""),
            })
            stats["relation_fix"] += 1
            continue
        relations.append(relation)

    out = postprocess({"entities": entities, "relations": relations}, record["text"])
    if strategy["noempty_fallback"] and not out.get("entities") and not out.get("relations"):
        stats["noempty_fallback"] += 1
        return original_record, stats
    return out, stats


def count_items(records):
    ent = Counter()
    rel = Counter()
    bad_endpoints = 0
    for record in records:
        entity_keys = {(e.get("text"), e.get("label")) for e in record.get("entities", []) or []}
        for entity in record.get("entities", []) or []:
            ent[entity.get("label")] += 1
        for relation in record.get("relations", []) or []:
            rel[relation.get("label")] += 1
            if (
                (relation.get("head"), relation.get("head_type")) not in entity_keys
                or (relation.get("tail"), relation.get("tail_type")) not in entity_keys
            ):
                bad_endpoints += 1
    return {
        "records": len(records),
        "entities": sum(ent.values()),
        "relations": sum(rel.values()),
        "empty": sum(1 for r in records if not r.get("entities") and not r.get("relations")),
        "bad_relation_endpoints": bad_endpoints,
        "entity_labels": dict(sorted(ent.items())),
        "relation_labels": dict(sorted(rel.items())),
    }


def delta_counts(after, before, key):
    labels = set(before[key]) | set(after[key])
    return {label: after[key].get(label, 0) - before[key].get(label, 0) for label in sorted(labels)}


def write_zip(json_path, zip_path):
    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(json_path, "submit.json")


def load_strategy(args):
    if args.strategy_json:
        strategy = load_json(args.strategy_json)
    else:
        strategy = dict(DEFAULT_PRESETS[args.preset])

    overrides = {
        "entity_policy": args.entity_policy,
        "entity_drop_threshold": args.entity_drop_threshold,
        "apply_entity_fix": args.apply_entity_fix,
        "entity_fix_threshold": args.entity_fix_threshold,
        "relation_policy": args.relation_policy,
        "relation_drop_threshold": args.relation_drop_threshold,
        "apply_relation_fix": args.apply_relation_fix,
        "relation_fix_threshold": args.relation_fix_threshold,
        "protect_relation_endpoints": args.protect_relation_endpoints,
        "noempty_fallback": args.noempty_fallback,
    }
    for key, value in overrides.items():
        if value is not None:
            strategy[key] = value

    defaults = {
        "entity_policy": "any_drop",
        "entity_drop_threshold": 0.0,
        "apply_entity_fix": True,
        "entity_fix_threshold": 0.0,
        "relation_policy": "drop_threshold",
        "relation_drop_threshold": 0.98,
        "apply_relation_fix": False,
        "relation_fix_threshold": 0.95,
        "protect_relation_endpoints": False,
        "noempty_fallback": True,
    }
    for key, value in defaults.items():
        strategy.setdefault(key, value)
    return strategy


def parse_bool(text):
    if text is None:
        return None
    value = str(text).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid bool value: {text}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True)
    parser.add_argument("--judgments", required=True)
    parser.add_argument("--solution_src", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_zip", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--preset", choices=sorted(DEFAULT_PRESETS), default="safe_anydrop_rel098")
    parser.add_argument("--strategy_json")
    parser.add_argument("--entity_policy", choices=["keep_all", "any_drop", "drop_threshold", "majority_drop", "majority_or_threshold"])
    parser.add_argument("--entity_drop_threshold", type=float)
    parser.add_argument("--apply_entity_fix", type=parse_bool)
    parser.add_argument("--entity_fix_threshold", type=float)
    parser.add_argument("--relation_policy", choices=["keep_all", "drop_threshold"])
    parser.add_argument("--relation_drop_threshold", type=float)
    parser.add_argument("--apply_relation_fix", type=parse_bool)
    parser.add_argument("--relation_fix_threshold", type=float)
    parser.add_argument("--protect_relation_endpoints", type=parse_bool)
    parser.add_argument("--noempty_fallback", type=parse_bool)
    parser.add_argument("--write_strategy_json")
    args = parser.parse_args()

    strategy = load_strategy(args)
    if args.write_strategy_json:
        dump_json(args.write_strategy_json, strategy)

    sys.path.insert(0, str(Path(args.solution_src).resolve()))
    from align import postprocess  # noqa: E402

    pred = load_json(args.pred)
    decisions_by_record = collect_decisions(load_jsonl(args.judgments))
    out = list(pred)
    stats = Counter()
    changed_records = 0
    for idx, decisions in decisions_by_record.items():
        if not (0 <= idx < len(out)):
            continue
        before = json.dumps(out[idx], ensure_ascii=False, sort_keys=True)
        out[idx], change = apply_strategy_to_record(
            out[idx],
            pred[idx],
            decisions,
            postprocess,
            strategy,
        )
        after = json.dumps(out[idx], ensure_ascii=False, sort_keys=True)
        if before != after:
            changed_records += 1
        stats.update(change)

    dump_json(args.output_json, out)
    write_zip(args.output_json, args.output_zip)

    before_stats = count_items(pred)
    after_stats = count_items(out)
    summary = {
        "strategy": strategy,
        "input": before_stats,
        "output": after_stats,
        "delta": {
            "entities": after_stats["entities"] - before_stats["entities"],
            "relations": after_stats["relations"] - before_stats["relations"],
            "empty": after_stats["empty"] - before_stats["empty"],
            "bad_relation_endpoints": after_stats["bad_relation_endpoints"] - before_stats["bad_relation_endpoints"],
            "entity_labels": delta_counts(after_stats, before_stats, "entity_labels"),
            "relation_labels": delta_counts(after_stats, before_stats, "relation_labels"),
        },
        "judged_records": len(decisions_by_record),
        "changed_records": changed_records,
        "applied": dict(sorted(stats.items())),
        "output_json": args.output_json,
        "output_zip": args.output_zip,
    }
    dump_json(args.summary, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
