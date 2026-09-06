"""Sweep record-level filter thresholds using existing online judgments.

This does not call any online model API. It replays the saved record-level
judgments from record_level_filter_5x5 and evaluates different entity/relation
drop thresholds on the same 25 trial-route pairs.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
from collections import Counter
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[2]
PAIR_ROOT = Path(__file__).resolve().parent / "outputs" / "record_level_filter_5x5" / "pairs"
OUT_DIR = Path(__file__).resolve().parent / "outputs" / "record_level_filter_5x5_threshold_sweep"
SOLUTION = Path("K:/\u6d69\u7136/CCL/solution/solution")
SOLUTION_SRC = SOLUTION / "src"
FILTER_WORKSPACE = Path(
    "C:/Users/Zzl410410/Documents/Codex/2026-06-26/"
    "019ef541-55e7-7af1-8f34-0927a521fe82"
)
APPLY_STRATEGY = FILTER_WORKSPACE / "work" / "apply_entity_pair_strategy.py"

TRIALS = [0, 1, 4, 6, 7]
ROUTES = [
    "majority",
    "best",
    "union",
    "strict3",
    "highscore_plus_light_drop_blocked_loi",
    "highscore_plus_light_add_has_aff1",
]


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


def count_bad_judgments(path):
    bad = 0
    rows = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows += 1
            obj = json.loads(line)
            if obj.get("parse_error") or not obj.get("decisions"):
                bad += 1
    return rows, bad


def apply_strategy(pred, decisions_by_record, strategy):
    out = list(pred)
    stats = Counter()
    changed = 0
    for idx, decisions in decisions_by_record.items():
        if not (0 <= idx < len(out)):
            continue
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


def load_pair(trial, route):
    pair_dir = PAIR_ROOT / f"trial_{trial:02d}__{route}"
    gold = load_json(PAIR_ROOT.parents[1] / "s100_t10" / f"trial_{trial:02d}" / "gold.json")
    pred = load_json(pair_dir / "prune_unseen_pred.json")
    rows, bad = count_bad_judgments(judgments_path(pair_dir))
    if bad:
        raise RuntimeError(f"{pair_dir.name} has bad judgments: {bad}/{rows}")
    decisions = APPLY.collect_decisions(APPLY.load_jsonl(judgments_path(pair_dir)))
    return {
        "trial": trial,
        "route": route,
        "gold": gold,
        "pred": pred,
        "decisions": decisions,
        "base_metric": metric_row(gold, pred),
    }


def summarize(rows):
    return {
        "n": len(rows),
        "mean_total": statistics.mean(r["total"] for r in rows),
        "mean_ner": statistics.mean(r["ner"] for r in rows),
        "mean_re": statistics.mean(r["re"] for r in rows),
        "entity_tp": sum(r["entity_tp"] for r in rows),
        "entity_fp": sum(r["entity_fp"] for r in rows),
        "entity_fn": sum(r["entity_fn"] for r in rows),
        "relation_tp": sum(r["relation_tp"] for r in rows),
        "relation_fp": sum(r["relation_fp"] for r in rows),
        "relation_fn": sum(r["relation_fn"] for r in rows),
        "entities": sum(r["entities"] for r in rows),
        "relations": sum(r["relations"] for r in rows),
    }


def parse_float_list(text, default):
    if text:
        return [float(x) for x in text.split(",") if x.strip()]
    return default


def render_report(results, baseline, path):
    top = sorted(results, key=lambda r: r["overall"]["mean_total"], reverse=True)
    lines = [
        "# Record-Level Filter Threshold Sweep",
        "",
        "All scores replay existing online judgments; no model calls are made.",
        "",
        "## Overall Top 20",
        "",
        "| rank | ent_thr | rel_thr | total | NER | RE | delta vs 0.75/0.95 | entity TP/FP/FN | relation TP/FP/FN |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    base_total = baseline["overall"]["mean_total"]
    for i, row in enumerate(top[:20], 1):
        o = row["overall"]
        lines.append(
            f"| {i} | {row['entity_threshold']:.2f} | {row['relation_threshold']:.2f} | "
            f"{o['mean_total']:.6f} | {o['mean_ner']:.6f} | {o['mean_re']:.6f} | "
            f"{o['mean_total'] - base_total:+.6f} | "
            f"{o['entity_tp']}/{o['entity_fp']}/{o['entity_fn']} | "
            f"{o['relation_tp']}/{o['relation_fp']}/{o['relation_fn']} |"
        )
    lines += ["", "## Best Per Route", ""]
    lines.append("| route | ent_thr | rel_thr | total | NER | RE | entity TP/FP/FN | relation TP/FP/FN |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    route_names = sorted(top[0]["by_route"])
    for route in route_names:
        best = max(results, key=lambda r: r["by_route"][route]["mean_total"])
        o = best["by_route"][route]
        lines.append(
            f"| `{route}` | {best['entity_threshold']:.2f} | {best['relation_threshold']:.2f} | "
            f"{o['mean_total']:.6f} | {o['mean_ner']:.6f} | {o['mean_re']:.6f} | "
            f"{o['entity_tp']}/{o['entity_fp']}/{o['entity_fn']} | "
            f"{o['relation_tp']}/{o['relation_fp']}/{o['relation_fn']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity_thresholds", default=None)
    parser.add_argument("--relation_thresholds", default=None)
    args = parser.parse_args()

    entity_thresholds = parse_float_list(args.entity_thresholds, [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95])
    relation_thresholds = parse_float_list(args.relation_thresholds, [0.85, 0.88, 0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pairs = [load_pair(trial, route) for trial in TRIALS for route in ROUTES]

    results = []
    baseline = None
    for ent_thr in entity_thresholds:
        for rel_thr in relation_thresholds:
            strategy = {
                "entity_policy": "drop_threshold",
                "entity_drop_threshold": ent_thr,
                "apply_entity_fix": True,
                "entity_fix_threshold": 0.0,
                "relation_policy": "drop_threshold",
                "relation_drop_threshold": rel_thr,
                "apply_relation_fix": False,
                "relation_fix_threshold": 0.95,
                "protect_relation_endpoints": False,
                "noempty_fallback": True,
            }
            rows = []
            by_route = {route: [] for route in ROUTES}
            applied = Counter()
            changed_records = 0
            for pair in pairs:
                pred, stats, changed = apply_strategy(pair["pred"], pair["decisions"], strategy)
                metric = metric_row(pair["gold"], pred)
                rows.append(metric)
                by_route[pair["route"]].append(metric)
                applied.update(stats)
                changed_records += changed
            item = {
                "entity_threshold": ent_thr,
                "relation_threshold": rel_thr,
                "overall": summarize(rows),
                "by_route": {route: summarize(vals) for route, vals in by_route.items()},
                "applied": dict(sorted(applied.items())),
                "changed_records": changed_records,
            }
            results.append(item)
            if abs(ent_thr - 0.75) < 1e-9 and abs(rel_thr - 0.95) < 1e-9:
                baseline = item

    if baseline is None:
        baseline = max(results, key=lambda r: r["overall"]["mean_total"])
    ranked = sorted(results, key=lambda r: r["overall"]["mean_total"], reverse=True)
    dump_json(OUT_DIR / "threshold_sweep_results.json", {"baseline": baseline, "ranked": ranked})
    render_report(ranked, baseline, OUT_DIR / "THRESHOLD_SWEEP.md")

    sol_out = SOLUTION / "outputs"
    if sol_out.exists():
        dump_json(sol_out / "record_level_filter_threshold_sweep_results.json", {"baseline": baseline, "ranked": ranked})
        (sol_out / "record_level_filter_THRESHOLD_SWEEP.md").write_text(
            (OUT_DIR / "THRESHOLD_SWEEP.md").read_text(encoding="utf-8"),
            encoding="utf-8",
        )

    best = ranked[0]
    print(json.dumps({
        "best": {
            "entity_threshold": best["entity_threshold"],
            "relation_threshold": best["relation_threshold"],
            "overall": best["overall"],
        },
        "baseline_075_095": baseline["overall"],
        "report": str(OUT_DIR / "THRESHOLD_SWEEP.md"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
