import argparse
import json
import sys
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


def source_slug(path):
    return Path(path).stem


def normalize_action(action):
    action = str(action or "").strip().lower()
    return action if action in {"keep", "drop", "fix"} else "keep"


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


def build_decision_map(judgment_rows):
    by_record = {}
    stats = {"decisions": 0, "keep": 0, "drop": 0, "fix": 0, "unknown": 0}
    for row in judgment_rows:
        idx = row.get("record_index")
        if idx is None:
            continue
        rec = by_record.setdefault(int(idx), {})
        for d in row.get("decisions", []):
            cid = str(d.get("id", "")).strip()
            if not cid:
                continue
            action = normalize_action(d.get("action"))
            stats["decisions"] += 1
            stats[action] = stats.get(action, 0) + 1
            rec[cid] = d
    return by_record, stats


def apply_one(record, decisions, postprocess):
    raw_entities = []
    raw_relations = []
    changes = {"entity_drop": 0, "entity_fix": 0, "relation_drop": 0, "relation_fix": 0}

    for i, ent in enumerate(record.get("entities", []) or []):
        cid = f"E{i}"
        d = decisions.get(cid)
        action = normalize_action(d.get("action") if d else "keep")
        if action == "drop":
            changes["entity_drop"] += 1
            continue
        if action == "fix" and valid_entity(d.get("corrected")):
            corr = dict(d["corrected"])
            raw_entities.append({
                "text": corr.get("text", ""),
                "label": corr.get("label", ""),
                "start": corr.get("start"),
                "end": corr.get("end"),
            })
            changes["entity_fix"] += 1
            continue
        raw_entities.append(ent)

    for i, rel in enumerate(record.get("relations", []) or []):
        cid = f"R{i}"
        d = decisions.get(cid)
        action = normalize_action(d.get("action") if d else "keep")
        if action == "drop":
            changes["relation_drop"] += 1
            continue
        if action == "fix" and valid_relation(d.get("corrected")):
            raw_relations.append(dict(d["corrected"]))
            changes["relation_fix"] += 1
            continue
        raw_relations.append(rel)

    return postprocess({"entities": raw_entities, "relations": raw_relations}, record["text"]), changes


def metric_flat(m):
    return {
        "total_score": m["total_score"],
        "score_ner": m["score_ner"],
        "score_re": m["score_re"],
        "entity_precision": m["entity"]["precision"],
        "entity_recall": m["entity"]["recall"],
        "entity_f1": m["entity"]["f1"],
        "relation_precision": m["relation"]["precision"],
        "relation_recall": m["relation"]["recall"],
        "relation_f1": m["relation"]["f1"],
        "entity_tp": m["entity"]["tp"],
        "entity_fp": m["entity"]["fp"],
        "entity_fn": m["entity"]["fn"],
        "relation_tp": m["relation"]["tp"],
        "relation_fp": m["relation"]["fp"],
        "relation_fn": m["relation"]["fn"],
    }


def diff(after, before):
    return {k: after[k] - before[k] for k in after if isinstance(after[k], (int, float))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--judgments", required=True)
    ap.add_argument("--output_pred", required=True)
    ap.add_argument("--output_report", required=True)
    ap.add_argument("--solution_src", default="src")
    ap.add_argument("--evaluated_records", choices=["judged", "all"], default="judged")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(args.solution_src).resolve()))
    from align import postprocess  # noqa: E402
    from common import evaluate  # noqa: E402

    gold = load_json(args.gold)
    pred = load_json(args.pred)
    judgment_rows = load_jsonl(args.judgments)
    decisions_by_record, decision_stats = build_decision_map(judgment_rows)

    out = list(pred)
    change_stats = {"entity_drop": 0, "entity_fix": 0, "relation_drop": 0, "relation_fix": 0}
    for idx, decisions in decisions_by_record.items():
        if idx < 0 or idx >= len(pred):
            continue
        out[idx], changes = apply_one(pred[idx], decisions, postprocess)
        for k, v in changes.items():
            change_stats[k] += v

    if args.evaluated_records == "judged":
        idxs = sorted(i for i in decisions_by_record if 0 <= i < len(gold))
        gold_eval = [gold[i] for i in idxs]
        pred_before = [pred[i] for i in idxs]
        pred_after = [out[i] for i in idxs]
    else:
        idxs = list(range(len(gold)))
        gold_eval = gold
        pred_before = pred
        pred_after = out

    before = metric_flat(evaluate(gold_eval, pred_before))
    after = metric_flat(evaluate(gold_eval, pred_after))
    report = {
        "pred": args.pred,
        "judgments": args.judgments,
        "evaluated_records_mode": args.evaluated_records,
        "evaluated_record_count": len(idxs),
        "evaluated_record_indices": idxs,
        "decision_stats": decision_stats,
        "applied_change_stats": change_stats,
        "before": before,
        "after": after,
        "delta": diff(after, before),
    }
    dump_json(args.output_pred, out)
    dump_json(args.output_report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
