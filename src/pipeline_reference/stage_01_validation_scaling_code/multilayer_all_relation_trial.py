"""Iteratively add all-entity + all-relation augment layers for one trial."""
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
OUT_DIR = OUT_ROOT / "multilayer_all_relation_trial"


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def dump_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def delta_metric(after, before):
    return {
        "total": after["total"] - before["total"],
        "ner": after["ner"] - before["ner"],
        "re": after["re"] - before["re"],
        "entity_tp": after["entity"]["tp"] - before["entity"]["tp"],
        "entity_fp": after["entity"]["fp"] - before["entity"]["fp"],
        "entity_fn": after["entity"]["fn"] - before["entity"]["fn"],
        "relation_tp": after["relation"]["tp"] - before["relation"]["tp"],
        "relation_fp": after["relation"]["fp"] - before["relation"]["fp"],
        "relation_fn": after["relation"]["fn"] - before["relation"]["fn"],
        "entities": after["entities"] - before["entities"],
        "relations": after["relations"] - before["relations"],
    }


def render_report(path, summary):
    lines = [
        "# Multilayer All-Relation Trial Probe",
        "",
        f"Trial: {summary['trial']}.",
        "",
        "Layer 0 is the final OPS output after: strict3 prune_unseen -> entity OPS -> entity all + relation all augment -> final OPS(entity=0.75, relation=0.95).",
        "",
        "| layer | total | NER | RE | entity TP/FP/FN | relation TP/FP/FN | entities | relations | changed entities | changed relations |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["layers"]:
        m = row["metric"]
        stats = row.get("augment_stats") or {}
        entity_apply = stats.get("entity_apply") or {}
        relation_apply = stats.get("relation_apply") or {}
        lines.append(
            f"| {row['layer']} | {m['total']:.6f} | {m['ner']:.6f} | {m['re']:.6f} | "
            f"{m['entity']['tp']}/{m['entity']['fp']}/{m['entity']['fn']} | "
            f"{m['relation']['tp']}/{m['relation']['fp']}/{m['relation']['fn']} | "
            f"{m['entities']} | {m['relations']} | "
            f"{entity_apply.get('changed_entities', 0)} | {relation_apply.get('changed_relations', 0)} |"
        )
    lines += [
        "",
        "## Deltas From Previous Layer",
        "",
        "| layer | total | NER | RE | entity TP | entity FP | relation TP | relation FP | entities | relations |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["layers"][1:]:
        d = row["delta_from_previous"]
        lines.append(
            f"| {row['layer']} | {d['total']:+.6f} | {d['ner']:+.6f} | {d['re']:+.6f} | "
            f"{d['entity_tp']:+d} | {d['entity_fp']:+d} | "
            f"{d['relation_tp']:+d} | {d['relation_fp']:+d} | "
            f"{d['entities']:+d} | {d['relations']:+d} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial", type=int, default=7)
    parser.add_argument("--layers", type=int, default=5)
    parser.add_argument(
        "--ops_summary",
        default=str(OUT_ROOT / "final_ops_rel095_5fold" / "ops_filter_sharded_summary.json"),
    )
    parser.add_argument("--variant", default="val_entity_ops_then_entity_all_relation_all")
    args = parser.parse_args()

    order.RELATION_OUTPUT_LABELS = set(rel_aug.RELATION_LABELS)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ops_summary = load_json(args.ops_summary)
    rows = {
        row["trial"]: row
        for row in ops_summary["variants"][args.variant]["trials"]
    }
    if args.trial not in rows:
        raise SystemExit(f"trial {args.trial} not found in {args.ops_summary}")

    entity_rows, _, _ = order.ent_aug.make_candidates([args.trial], None)
    entity_decisions = order.ent_aug.collect_decisions()
    relation_decisions = rel_aug.collect_decisions()
    gold = order.load_json(order.OUT_ROOT / "s100_t10" / f"trial_{args.trial:02d}" / "gold.json")

    current = load_json(rows[args.trial]["output_json"])
    layers = []
    previous_metric = None

    for layer in range(args.layers + 1):
        out_json = OUT_DIR / f"trial_{args.trial:02d}_layer_{layer:02d}_entity_all_relation_all.json"
        dump_json(out_json, current)
        first_layer.write_zip(out_json)
        metric = order.metric(gold, current)
        row = {
            "layer": layer,
            "path": str(out_json),
            "metric": metric,
        }
        if previous_metric is not None:
            row["delta_from_previous"] = delta_metric(metric, previous_metric)
        layers.append(row)
        previous_metric = metric

        if layer >= args.layers:
            break
        augmented, stats = order.augment_base(
            args.trial,
            current,
            entity_rows,
            entity_decisions,
            relation_decisions,
        )
        current = augmented
        row["next_augment_stats"] = stats
        # Store stats on the following layer too for easier table rendering.
        if layer + 1 <= args.layers:
            pending_stats = stats
        else:
            pending_stats = None
        if pending_stats is not None:
            # The next loop creates the layer row, so keep a side table.
            pass

    # Attach augment stats to the layer they produced.
    current = load_json(rows[args.trial]["output_json"])
    for i in range(1, len(layers)):
        augmented, stats = order.augment_base(
            args.trial,
            current,
            entity_rows,
            entity_decisions,
            relation_decisions,
        )
        layers[i]["augment_stats"] = stats
        current = augmented
    layers[0]["augment_stats"] = {}

    summary = {
        "trial": args.trial,
        "layers_requested": args.layers,
        "base_after_final_ops": rows[args.trial]["output_json"],
        "layers": layers,
    }
    dump_json(OUT_DIR / f"trial_{args.trial:02d}_multilayer_summary.json", summary)
    render_report(OUT_DIR / f"trial_{args.trial:02d}_MULTILAYER_SUMMARY.md", summary)

    user_out = Path("C:/Users/Zzl410410/Documents/Codex/2026-06-28/fnag/outputs/multilayer_all_relation_trial")
    user_out.mkdir(parents=True, exist_ok=True)
    for src in [
        OUT_DIR / f"trial_{args.trial:02d}_multilayer_summary.json",
        OUT_DIR / f"trial_{args.trial:02d}_MULTILAYER_SUMMARY.md",
    ]:
        shutil.copy2(src, user_out / src.name)
    print(json.dumps({"summary": str(user_out / f"trial_{args.trial:02d}_multilayer_summary.json"), "report": str(user_out / f"trial_{args.trial:02d}_MULTILAYER_SUMMARY.md")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
