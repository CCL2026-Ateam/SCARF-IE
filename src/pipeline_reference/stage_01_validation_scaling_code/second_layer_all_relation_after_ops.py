"""Add a second all-entity + all-relation augmentation layer after final OPS."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import entity_ops_then_all_relation_layer_experiment as first_layer
import order_swap_entityonly_experiment as order
import relation_aug_core as rel_aug


PROJECT = Path(__file__).resolve().parent
OUT_ROOT = PROJECT / "outputs" / "entity_ops_then_all_relation_layer"
OUT_DIR = OUT_ROOT / "second_entity_relation_all_after_final_ops"


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def dump_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def metric_from_ops_aggregate(row):
    return {
        "mean_total": row["mean_total"],
        "mean_ner": row["mean_ner"],
        "mean_re": row["mean_re"],
        "entity_tp": row["entity_tp"],
        "entity_fp": row["entity_fp"],
        "entity_fn": row["entity_fn"],
        "relation_tp": row["relation_tp"],
        "relation_fp": row["relation_fp"],
        "relation_fn": row["relation_fn"],
        "entities": row["entities"],
        "relations": row["relations"],
    }


def aggregate_delta(after, before):
    return {
        "total": after["mean_total"] - before["mean_total"],
        "ner": after["mean_ner"] - before["mean_ner"],
        "re": after["mean_re"] - before["mean_re"],
        "entity_tp": after["entity_tp"] - before["entity_tp"],
        "entity_fp": after["entity_fp"] - before["entity_fp"],
        "entity_fn": after["entity_fn"] - before["entity_fn"],
        "relation_tp": after["relation_tp"] - before["relation_tp"],
        "relation_fp": after["relation_fp"] - before["relation_fp"],
        "relation_fn": after["relation_fn"] - before["relation_fn"],
        "entities": after["entities"] - before["entities"],
        "relations": after["relations"] - before["relations"],
    }


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


def render_report(path, summary):
    def stage_line(label, m):
        return (
            f"| {label} | {m['mean_total']:.6f} | {m['mean_ner']:.6f} | {m['mean_re']:.6f} | "
            f"{m['entity_tp']}/{m['entity_fp']}/{m['entity_fn']} | "
            f"{m['relation_tp']}/{m['relation_fp']}/{m['relation_fn']} | "
            f"{m['entities']} | {m['relations']} |"
        )

    lines = [
        "# Second All-Relation Augment After Final OPS",
        "",
        f"Trials: {', '.join(str(t) for t in summary['trials'])}.",
        "",
        "Pipeline: strict3 prune_unseen -> entity OPS only -> entity all + relation all augment -> final OPS(entity=0.75, relation=0.95) -> entity all + relation all augment again.",
        f"Second-layer entity dedupe: {summary['strategy']['second_layer_entity_dedupe']}.",
        "",
        "| stage | total | NER | RE | entity TP/FP/FN | relation TP/FP/FN | entities | relations |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, label in [
        ("strict3_prune_unseen", "strict3 prune_unseen"),
        ("entity_ops_base", "entity OPS base"),
        ("first_all_relation_augment", "first entity all + relation all augment"),
        ("final_ops_rel095", "final OPS entity=0.75 relation=0.95"),
        ("second_all_relation_augment", "second entity all + relation all augment"),
    ]:
        lines.append(stage_line(label, summary["stages"][key]))
    lines += [
        "",
        "## Deltas",
        "",
        "| comparison | total | NER | RE | entity TP | entity FP | entity FN | relation TP | relation FP | relation FN |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, delta in [
        ("final OPS minus first augment", summary["deltas"]["final_ops_minus_first_augment"]),
        ("second augment minus final OPS", summary["deltas"]["second_augment_minus_final_ops"]),
        ("second augment minus first augment", summary["deltas"]["second_augment_minus_first_augment"]),
    ]:
        lines.append(
            f"| {label} | {delta['total']:+.6f} | {delta['ner']:+.6f} | {delta['re']:+.6f} | "
            f"{delta['entity_tp']:+d} | {delta['entity_fp']:+d} | {delta['entity_fn']:+d} | "
            f"{delta['relation_tp']:+d} | {delta['relation_fp']:+d} | {delta['relation_fn']:+d} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ops_summary",
        default=str(OUT_ROOT / "final_ops_rel095_5fold" / "ops_filter_sharded_summary.json"),
    )
    parser.add_argument("--variant", default="val_entity_ops_then_entity_all_relation_all")
    parser.add_argument("--trials", type=int, nargs="+", default=[0, 1, 4, 6, 7])
    parser.add_argument(
        "--prepare_summary",
        default=str(OUT_ROOT / "prepare_summary.json"),
    )
    parser.add_argument(
        "--out_dir",
        default=str(OUT_DIR),
    )
    parser.add_argument(
        "--user_out",
        default="C:/Users/Zzl410410/Documents/Codex/2026-06-28/fnag/outputs/entity_ops_then_all_relation_layer_5fold",
    )
    parser.add_argument(
        "--entity_dedupe",
        choices=["text_label", "span_only"],
        default="text_label",
        help="Entity merge rule for the second augment layer only.",
    )
    args = parser.parse_args()

    order.RELATION_OUTPUT_LABELS = set(rel_aug.RELATION_LABELS)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ops_summary = load_json(args.ops_summary)
    variant = ops_summary["variants"][args.variant]
    ops_rows = {row["trial"]: row for row in variant["trials"]}
    entity_rows, _, _ = order.ent_aug.make_candidates(args.trials, None)
    entity_decisions = order.ent_aug.collect_decisions()
    relation_decisions = rel_aug.collect_decisions()

    per_trial = []
    metrics = []
    for trial in args.trials:
        ops_row = ops_rows[trial]
        base = load_json(ops_row["output_json"])
        augmented, stats = augment_base_with_entity_dedupe(
            trial,
            base,
            entity_rows,
            entity_decisions,
            relation_decisions,
            args.entity_dedupe,
        )
        out_json = out_dir / f"trial_{trial:02d}_second_entity_all_relation_all_augmented.json"
        dump_json(out_json, augmented)
        first_layer.write_zip(out_json)
        gold = order.load_json(order.OUT_ROOT / "s100_t10" / f"trial_{trial:02d}" / "gold.json")
        m = order.metric(gold, augmented)
        metrics.append(m)
        per_trial.append(
            {
                "trial": trial,
                "input_after_final_ops": ops_row["output_json"],
                "output_json": str(out_json),
                "augment_stats": stats,
                "before_second_augment": ops_row["after"],
                "after_second_augment": m,
            }
        )

    prepare_summary = load_json(args.prepare_summary)
    stages = {
        "strict3_prune_unseen": prepare_summary["stages"]["strict3_prune_unseen"],
        "entity_ops_base": prepare_summary["stages"]["entity_ops_base"],
        "first_all_relation_augment": prepare_summary["stages"]["entity_ops_then_all_relation_augment"],
        "final_ops_rel095": metric_from_ops_aggregate(variant["after"]),
        "second_all_relation_augment": order.aggregate(metrics),
    }
    summary = {
        "trials": args.trials,
        "strategy": {
            "base": "strict3 prune_unseen",
            "first_ops": "entity=0.75 relation=1.01",
            "first_augment": "entity all + relation all",
            "final_ops": "entity=0.75 relation=0.95",
            "second_augment": "entity all + relation all",
            "second_layer_entity_dedupe": args.entity_dedupe,
        },
        "stages": stages,
        "deltas": {
            "final_ops_minus_first_augment": aggregate_delta(stages["final_ops_rel095"], stages["first_all_relation_augment"]),
            "second_augment_minus_final_ops": aggregate_delta(stages["second_all_relation_augment"], stages["final_ops_rel095"]),
            "second_augment_minus_first_augment": aggregate_delta(stages["second_all_relation_augment"], stages["first_all_relation_augment"]),
        },
        "per_trial": per_trial,
    }
    dump_json(out_dir / "second_layer_summary.json", summary)
    render_report(out_dir / "SECOND_LAYER_SUMMARY.md", summary)

    user_out = Path(args.user_out)
    user_out.mkdir(parents=True, exist_ok=True)
    for src in [
        Path(args.prepare_summary),
        Path(args.ops_summary),
        out_dir / "second_layer_summary.json",
        out_dir / "SECOND_LAYER_SUMMARY.md",
    ]:
        if src.exists():
            shutil.copy2(src, user_out / src.name)
    print(json.dumps({"summary": str(out_dir / "second_layer_summary.json"), "report": str(out_dir / "SECOND_LAYER_SUMMARY.md")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
