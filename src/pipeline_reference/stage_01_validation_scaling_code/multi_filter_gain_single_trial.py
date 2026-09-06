"""Replay multi-filter combinations on one saved Test-B trial.

This is an offline experiment: it reuses saved route predictions and saved
record-level judgments from outputs/record_level_filter_5x5. No model calls are
made.
"""
from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path


PROJECT = Path(__file__).resolve().parent
PAIR_ROOT = PROJECT / "outputs" / "record_level_filter_5x5" / "pairs"
OUT_ROOT = PROJECT / "outputs"
SOLUTION = Path("K:/\u6d69\u7136/CCL/solution/solution")
SOLUTION_SRC = SOLUTION / "src"
FILTER_WORKSPACE = Path(
    "C:/Users/Zzl410410/Documents/Codex/2026-06-26/"
    "019ef541-55e7-7af1-8f34-0927a521fe82"
)
APPLY_STRATEGY = FILTER_WORKSPACE / "work" / "apply_entity_pair_strategy.py"

ROUTES = [
    "majority",
    "best",
    "union",
    "strict3",
    "highscore_plus_light_drop_blocked_loi",
    "highscore_plus_light_add_has_aff1",
]

DEFAULT_STRATEGY = {
    "entity_policy": "drop_threshold",
    "entity_drop_threshold": 0.0,
    "apply_entity_fix": True,
    "entity_fix_threshold": 0.0,
    "relation_policy": "drop_threshold",
    "relation_drop_threshold": 0.91,
    "apply_relation_fix": False,
    "relation_fix_threshold": 0.95,
    "protect_relation_endpoints": False,
    "noempty_fallback": True,
}


sys.path.insert(0, str(SOLUTION_SRC))
from align import postprocess  # noqa: E402
from common import dump_json, evaluate, load_json  # noqa: E402


def load_apply_module():
    spec = importlib.util.spec_from_file_location("apply_entity_pair_strategy", APPLY_STRATEGY)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {APPLY_STRATEGY}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


APPLY = load_apply_module()


def judgments_path(pair_dir):
    return pair_dir / "judgments" / "prune_unseen_pred.opus_item_judgments.jsonl"


def item_key(kind, item):
    if kind == "entity":
        return (
            "E",
            str(item.get("text", "")),
            str(item.get("label", "")),
        )
    if kind == "relation":
        return (
            "R",
            str(item.get("head", "")),
            str(item.get("head_type", "")),
            str(item.get("tail", "")),
            str(item.get("tail_type", "")),
            str(item.get("label", "")),
        )
    raise ValueError(f"unknown kind: {kind}")


def metric_row(gold, pred):
    m = evaluate(gold, pred)
    return {
        "total": m["total_score"],
        "ner": m["score_ner"],
        "re": m["score_re"],
        "entity_tp": m["entity"]["tp"],
        "entity_fp": m["entity"]["fp"],
        "entity_fn": m["entity"]["fn"],
        "relation_tp": m["relation"]["tp"],
        "relation_fp": m["relation"]["fp"],
        "relation_fn": m["relation"]["fn"],
        "entities": sum(len(r.get("entities", []) or []) for r in pred),
        "relations": sum(len(r.get("relations", []) or []) for r in pred),
    }


def delta(after, before):
    return {
        "total": after["total"] - before["total"],
        "ner": after["ner"] - before["ner"],
        "re": after["re"] - before["re"],
        "entity_tp": after["entity_tp"] - before["entity_tp"],
        "entity_fp": after["entity_fp"] - before["entity_fp"],
        "entity_fn": after["entity_fn"] - before["entity_fn"],
        "relation_tp": after["relation_tp"] - before["relation_tp"],
        "relation_fp": after["relation_fp"] - before["relation_fp"],
        "relation_fn": after["relation_fn"] - before["relation_fn"],
        "entities": after["entities"] - before["entities"],
        "relations": after["relations"] - before["relations"],
    }


def decisions_by_content(pred, rows):
    by_record = defaultdict(lambda: defaultdict(list))
    for row in rows:
        idx = row.get("record_index")
        if idx is None or not (0 <= int(idx) < len(pred)):
            continue
        record = pred[int(idx)]
        for decision in row.get("decisions", []) or []:
            cid = str(decision.get("id", "")).strip()
            if len(cid) < 2:
                continue
            kind = "entity" if cid[0] == "E" else "relation" if cid[0] == "R" else None
            if kind is None:
                continue
            try:
                item_index = int(cid[1:])
            except ValueError:
                continue
            items = record.get("entities" if kind == "entity" else "relations", []) or []
            if not (0 <= item_index < len(items)):
                continue
            copied = dict(decision)
            copied["source_id"] = cid
            copied["source_kind"] = kind
            by_record[int(idx)][item_key(kind, items[item_index])].append(copied)
    return by_record


def load_trial_route(trial, route):
    pair_dir = PAIR_ROOT / f"trial_{trial:02d}__{route}"
    pred = load_json(pair_dir / "prune_unseen_pred.json")
    rows = APPLY.load_jsonl(judgments_path(pair_dir))
    decisions = decisions_by_content(pred, rows)
    return {
        "route": route,
        "pair_dir": pair_dir,
        "pred": pred,
        "decisions_by_content": decisions,
    }


def remap_content_decisions(target_pred, content_decisions):
    by_record = defaultdict(lambda: defaultdict(list))
    for idx, record in enumerate(target_pred):
        source_record_decisions = content_decisions.get(idx, {})
        for i, entity in enumerate(record.get("entities", []) or []):
            by_record[idx][f"E{i}"].extend(source_record_decisions.get(item_key("entity", entity), []))
        for i, relation in enumerate(record.get("relations", []) or []):
            by_record[idx][f"R{i}"].extend(source_record_decisions.get(item_key("relation", relation), []))
    return by_record


def merge_content_decisions(routes):
    merged = defaultdict(lambda: defaultdict(list))
    for route in routes:
        for idx, record_decisions in route["decisions_by_content"].items():
            for key, decisions in record_decisions.items():
                for decision in decisions:
                    copied = dict(decision)
                    copied["source_route"] = route["route"]
                    merged[idx][key].append(copied)
    return merged


def apply_content_filter(pred, content_decisions, strategy):
    out = deepcopy(pred)
    remapped = remap_content_decisions(out, content_decisions)
    stats = Counter()
    changed = 0
    for idx, decisions in remapped.items():
        before = json.dumps(out[idx], ensure_ascii=False, sort_keys=True)
        out[idx], change = APPLY.apply_strategy_to_record(
            out[idx],
            pred[idx],
            decisions,
            postprocess,
            strategy,
        )
        after = json.dumps(out[idx], ensure_ascii=False, sort_keys=True)
        if before != after:
            changed += 1
        stats.update(change)
    return out, stats, changed


def apply_content_cascade(pred, route_decisions, order, strategy):
    out = deepcopy(pred)
    stats = Counter()
    changed_total = 0
    step_metrics = []
    for route in order:
        before = json.dumps(out, ensure_ascii=False, sort_keys=True)
        out, step_stats, step_changed = apply_content_filter(out, route_decisions[route], strategy)
        after = json.dumps(out, ensure_ascii=False, sort_keys=True)
        stats.update({f"{route}:{key}": value for key, value in step_stats.items()})
        changed_total += step_changed
        step_metrics.append({
            "route": route,
            "changed_records": step_changed,
            "changed_prediction": before != after,
            "applied": dict(sorted(step_stats.items())),
        })
    return out, stats, changed_total, step_metrics


def compact(name, pred, gold, base_metric, stats=None, changed=0, extra=None):
    metric = metric_row(gold, pred)
    row = {
        "name": name,
        "metric": metric,
        "delta_vs_base": delta(metric, base_metric),
        "changed_records": changed,
        "applied": dict(sorted((stats or Counter()).items())),
    }
    if extra:
        row.update(extra)
    return row


def render_report(path, trial, target_route, strategy, rows, permutation_rows):
    best_total = max(rows, key=lambda r: r["metric"]["total"])
    lines = [
        f"# Multi-Filter Replay Trial {trial:02d}",
        "",
        "Offline replay using saved record-level judgments only; no model calls were made.",
        "",
        f"Target prediction route: `{target_route}`.",
        f"Strategy: entity drop >= {strategy['entity_drop_threshold']:.2f}, relation drop >= {strategy['relation_drop_threshold']:.2f}, entity fixes enabled.",
        "",
        "## Main Results",
        "",
        "| rank | experiment | total | NER | RE | delta total | entities | relations | changed records | entity TP/FP/FN | relation TP/FP/FN |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for i, row in enumerate(sorted(rows, key=lambda r: r["metric"]["total"], reverse=True), 1):
        metric = row["metric"]
        d = row["delta_vs_base"]
        lines.append(
            f"| {i} | `{row['name']}` | {metric['total']:.6f} | {metric['ner']:.6f} | {metric['re']:.6f} | "
            f"{d['total']:+.6f} | {metric['entities']} | {metric['relations']} | {row['changed_records']} | "
            f"{metric['entity_tp']}/{metric['entity_fp']}/{metric['entity_fn']} | "
            f"{metric['relation_tp']}/{metric['relation_fp']}/{metric['relation_fn']} |"
        )
    lines += [
        "",
        "## Best Cascade Orders",
        "",
        "| rank | order | total | NER | RE | delta total | changed records | entities | relations |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for i, row in enumerate(permutation_rows[:20], 1):
        metric = row["metric"]
        d = row["delta_vs_base"]
        order = " -> ".join(row["order"])
        lines.append(
            f"| {i} | `{order}` | {metric['total']:.6f} | {metric['ner']:.6f} | {metric['re']:.6f} | "
            f"{d['total']:+.6f} | {row['changed_records']} | {metric['entities']} | {metric['relations']} |"
        )
    lines += [
        "",
        "## Takeaway",
        "",
        f"Best main experiment: `{best_total['name']}` with total {best_total['metric']['total']:.6f} "
        f"({best_total['delta_vs_base']['total']:+.6f} vs target base).",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial", type=int, default=0)
    parser.add_argument("--target_route", default="strict3", choices=ROUTES)
    parser.add_argument("--entity_threshold", type=float, default=DEFAULT_STRATEGY["entity_drop_threshold"])
    parser.add_argument("--relation_threshold", type=float, default=DEFAULT_STRATEGY["relation_drop_threshold"])
    parser.add_argument("--max_permutations", type=int, default=720)
    args = parser.parse_args()

    strategy = dict(DEFAULT_STRATEGY)
    strategy["entity_drop_threshold"] = args.entity_threshold
    strategy["relation_drop_threshold"] = args.relation_threshold

    gold = load_json(OUT_ROOT / "s100_t10" / f"trial_{args.trial:02d}" / "gold.json")
    route_data = {route: load_trial_route(args.trial, route) for route in ROUTES}
    target_pred = route_data[args.target_route]["pred"]
    base_metric = metric_row(gold, target_pred)

    route_content = {route: data["decisions_by_content"] for route, data in route_data.items()}
    rows = [compact("target_base_pruned", target_pred, gold, base_metric)]

    for route in ROUTES:
        filtered, stats, changed = apply_content_filter(target_pred, route_content[route], strategy)
        rows.append(compact(f"single_filter__{route}", filtered, gold, base_metric, stats, changed))

    own_once, own_stats, own_changed = apply_content_filter(
        target_pred,
        route_content[args.target_route],
        strategy,
    )
    own_twice, twice_stats, twice_changed = apply_content_filter(
        own_once,
        route_content[args.target_route],
        strategy,
    )
    rows.append(compact("repeat_same_filter_twice", own_twice, gold, base_metric, twice_stats, twice_changed))

    merged_all = merge_content_decisions(route_data.values())
    union_filtered, union_stats, union_changed = apply_content_filter(target_pred, merged_all, strategy)
    rows.append(compact("merged_all_filters_once", union_filtered, gold, base_metric, union_stats, union_changed))

    default_order = [
        args.target_route,
        "best",
        "highscore_plus_light_drop_blocked_loi",
        "highscore_plus_light_add_has_aff1",
        "majority",
        "union",
    ]
    default_order = [route for route in default_order if route in ROUTES]
    cascaded, cascade_stats, cascade_changed, steps = apply_content_cascade(
        target_pred,
        route_content,
        default_order,
        strategy,
    )
    rows.append(compact(
        "cascade_default_order",
        cascaded,
        gold,
        base_metric,
        cascade_stats,
        cascade_changed,
        {"order": default_order, "steps": steps},
    ))

    permutation_rows = []
    for order in itertools.islice(itertools.permutations(ROUTES), args.max_permutations):
        pred, stats, changed, steps = apply_content_cascade(target_pred, route_content, order, strategy)
        permutation_rows.append(compact(
            "cascade_permutation",
            pred,
            gold,
            base_metric,
            stats,
            changed,
            {"order": list(order), "steps": steps},
        ))
    permutation_rows.sort(key=lambda r: r["metric"]["total"], reverse=True)

    out_dir = OUT_ROOT / f"record_level_multi_filter_trial_{args.trial:02d}__{args.target_route}"
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "trial": args.trial,
        "target_route": args.target_route,
        "strategy": strategy,
        "routes": ROUTES,
        "main_results": sorted(rows, key=lambda r: r["metric"]["total"], reverse=True),
        "cascade_permutations": permutation_rows,
    }
    dump_json(out_dir / "multi_filter_results.json", result)
    render_report(
        out_dir / "MULTI_FILTER_REPLAY.md",
        args.trial,
        args.target_route,
        strategy,
        rows,
        permutation_rows,
    )
    print(json.dumps({
        "trial": args.trial,
        "target_route": args.target_route,
        "best_main": result["main_results"][0],
        "best_cascade": permutation_rows[0],
        "report": str(out_dir / "MULTI_FILTER_REPLAY.md"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
