"""Probe: entity OPS first, then all-entity + all-relation augmentation.

This reuses existing union-only AI judgments and writes a one-trial prediction
that can be passed through a final OPS layer with relation filtering.
"""
from __future__ import annotations

import argparse
import json
import zipfile
from collections import Counter
from pathlib import Path

import order_swap_entityonly_experiment as order
import relation_aug_core as rel_aug


PROJECT = Path(__file__).resolve().parent
OUT_DIR = PROJECT / "outputs" / "entity_ops_then_all_relation_layer"


def dump_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_zip(json_path):
    json_path = Path(json_path)
    zip_path = json_path.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(json_path, "submit.json")
    return zip_path


def add_entities_span_only(base, additions):
    out = []
    changed = 0
    for record, ents in zip(base, additions):
        row = dict(record)
        existing = {order.ent_aug.entity_key(e) for e in row.get("entities", []) or []}
        merged = list(row.get("entities", []) or [])
        for entity in ents:
            key = order.ent_aug.entity_key(entity)
            if key in existing:
                continue
            merged.append(entity)
            existing.add(key)
            changed += 1
        row["entities"] = sorted(merged, key=lambda e: (int(e["start"]), int(e["end"]), str(e["label"])))
        out.append(row)
    return out, changed


def augment_base_with_entity_dedupe(trial, base, entity_rows, entity_decisions, relation_decisions, entity_dedupe):
    if entity_dedupe == "text_label":
        return order.augment_base(trial, base, entity_rows, entity_decisions, relation_decisions)

    original_add_entities = order.ent_aug.add_entities
    order.ent_aug.add_entities = add_entities_span_only
    try:
        return order.augment_base(trial, base, entity_rows, entity_decisions, relation_decisions)
    finally:
        order.ent_aug.add_entities = original_add_entities


def run_trial(trial, entity_threshold, entity_rows, entity_decisions, relation_decisions, out_dir, entity_dedupe):
    out_dir.mkdir(parents=True, exist_ok=True)
    order.RELATION_OUTPUT_LABELS = set(rel_aug.RELATION_LABELS)

    gold = order.load_json(order.OUT_ROOT / "s100_t10" / f"trial_{trial:02d}" / "gold.json")
    prune = order.load_json(order.pair_dir(trial, "strict3") / "prune_unseen_pred.json")

    filtered_base, filter_stats, filter_changed = order.apply_entity_only_ops(
        trial,
        prune,
        entity_threshold,
    )
    augmented, augment_stats = augment_base_with_entity_dedupe(
        trial,
        filtered_base,
        entity_rows,
        entity_decisions,
        relation_decisions,
        entity_dedupe,
    )

    base_path = out_dir / f"trial_{trial:02d}_entity_ops_then_entity_all_relation_all_augmented.json"
    dump_json(base_path, augmented)
    zip_path = write_zip(base_path)

    summary = {
        "trial": trial,
        "settings": {
            "base": "strict3 prune_unseen",
            "first_ops_entity_threshold": entity_threshold,
            "first_ops_relation_threshold": 1.01,
            "entity_labels": "all",
            "relation_output_labels": "all",
            "entity_dedupe": entity_dedupe,
        },
        "paths": {
            "prediction": str(base_path),
            "zip": str(zip_path),
        },
        "entity_candidates": sum(1 for row in entity_rows if row["trial"] == trial),
        "entity_candidate_by_label": dict(
            sorted(Counter(row["label"] for row in entity_rows if row["trial"] == trial).items())
        ),
        "entity_filter_base": {
            "applied": filter_stats,
            "changed_records": filter_changed,
        },
        "augment_stats": augment_stats,
        "metrics": {
            "strict3_prune_unseen": order.metric(gold, prune),
            "entity_ops_base": order.metric(gold, filtered_base),
            "entity_ops_then_all_relation_augment": order.metric(gold, augmented),
        },
    }
    dump_json(out_dir / f"trial_{trial:02d}_prepare_summary.json", summary)
    return summary


def run_trials(trials, entity_threshold, out_dir, entity_dedupe):
    order.RELATION_OUTPUT_LABELS = set(rel_aug.RELATION_LABELS)
    entity_rows, _, _ = order.ent_aug.make_candidates(trials, None)
    entity_decisions = order.ent_aug.collect_decisions()
    relation_decisions = rel_aug.collect_decisions()
    summaries = [
        run_trial(trial, entity_threshold, entity_rows, entity_decisions, relation_decisions, out_dir, entity_dedupe)
        for trial in trials
    ]
    aggregate_summary = {
        "trials": trials,
        "settings": {
            "base": "strict3 prune_unseen",
            "first_ops_entity_threshold": entity_threshold,
            "first_ops_relation_threshold": 1.01,
            "entity_labels": "all",
            "relation_output_labels": "all",
            "entity_dedupe": entity_dedupe,
        },
        "stages": {
            stage: order.aggregate([summary["metrics"][stage] for summary in summaries])
            for stage in [
                "strict3_prune_unseen",
                "entity_ops_base",
                "entity_ops_then_all_relation_augment",
            ]
        },
        "trial_summaries": summaries,
    }
    dump_json(out_dir / "prepare_summary.json", aggregate_summary)
    return aggregate_summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial", type=int, default=0)
    parser.add_argument("--trials", type=int, nargs="+", default=None)
    parser.add_argument("--entity_threshold", type=float, default=0.75)
    parser.add_argument("--out_dir", default=str(OUT_DIR))
    parser.add_argument(
        "--entity_dedupe",
        choices=["text_label", "span_only"],
        default="text_label",
        help="Entity merge rule for this first augmentation layer.",
    )
    args = parser.parse_args()
    trials = args.trials or [args.trial]
    summary = run_trials(trials, args.entity_threshold, Path(args.out_dir), args.entity_dedupe)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
