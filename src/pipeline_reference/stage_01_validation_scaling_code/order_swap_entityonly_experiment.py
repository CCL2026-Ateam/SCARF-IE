"""Compare entity-only OPS filter order with union-only augmentation.

Pipelines:
1. strict3 prune_unseen -> entity all + relation CON/LOI augment -> entity-only OPS
2. strict3 prune_unseen -> entity-only OPS -> entity all + relation CON/LOI augment
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

import relation_aug_core as rel_aug
import union_only_entity_ai_augmenter as ent_aug


PROJECT = Path(__file__).resolve().parent
OUT_ROOT = PROJECT / "outputs"
PAIR_ROOT = OUT_ROOT / "record_level_filter_5x5" / "pairs"
EXP_OUT = OUT_ROOT / "order_swap_entityonly_filter"
APPLY_PATH = Path(
    "C:/Users/Zzl410410/Documents/Codex/2026-06-26/"
    "019ef541-55e7-7af1-8f34-0927a521fe82/work/apply_entity_pair_strategy.py"
)
TRIALS = [0, 1, 4, 6, 7]
RELATION_OUTPUT_LABELS = {"CON", "LOI"}


def solution_src():
    candidates = sorted(Path("K:/").glob("*/CCL/solution/solution/src/common.py"))
    if not candidates:
        raise RuntimeError("Cannot find solution common.py under K:/")
    return candidates[0].parent


SOLUTION_SRC = solution_src()
sys.path.insert(0, str(SOLUTION_SRC))
from align import postprocess  # noqa: E402
from common import evaluate  # noqa: E402


def load_apply_module():
    spec = importlib.util.spec_from_file_location("apply_entity_pair_strategy", APPLY_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {APPLY_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


APPLY = load_apply_module()


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def dump_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_zip(json_path, zip_path):
    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(json_path, "submit.json")


def pair_dir(trial, route):
    return PAIR_ROOT / f"trial_{trial:02d}__{route}"


def judgment_path(trial):
    return pair_dir(trial, "strict3") / "judgments" / "prune_unseen_pred.opus_item_judgments.jsonl"


def metric(gold, pred):
    m = evaluate(gold, pred)
    return {
        "total": m["total_score"],
        "ner": m["score_ner"],
        "re": m["score_re"],
        "entity": m["entity"],
        "relation": m["relation"],
        "entities": sum(len(r.get("entities") or []) for r in pred),
        "relations": sum(len(r.get("relations") or []) for r in pred),
    }


def aggregate(metrics):
    return {
        "mean_total": sum(m["total"] for m in metrics) / len(metrics),
        "mean_ner": sum(m["ner"] for m in metrics) / len(metrics),
        "mean_re": sum(m["re"] for m in metrics) / len(metrics),
        "entity_tp": sum(m["entity"]["tp"] for m in metrics),
        "entity_fp": sum(m["entity"]["fp"] for m in metrics),
        "entity_fn": sum(m["entity"]["fn"] for m in metrics),
        "relation_tp": sum(m["relation"]["tp"] for m in metrics),
        "relation_fp": sum(m["relation"]["fp"] for m in metrics),
        "relation_fn": sum(m["relation"]["fn"] for m in metrics),
        "entities": sum(m["entities"] for m in metrics),
        "relations": sum(m["relations"] for m in metrics),
    }


def entity_only_strategy(entity_threshold=0.75):
    return {
        "entity_policy": "drop_threshold",
        "entity_drop_threshold": entity_threshold,
        "apply_entity_fix": True,
        "entity_fix_threshold": 0.0,
        "relation_policy": "drop_threshold",
        "relation_drop_threshold": 1.01,
        "apply_relation_fix": False,
        "relation_fix_threshold": 0.95,
        "protect_relation_endpoints": False,
        "noempty_fallback": True,
    }


def apply_entity_only_ops(trial, pred, entity_threshold=0.75):
    decisions = APPLY.collect_decisions(load_jsonl(judgment_path(trial)))
    strategy = entity_only_strategy(entity_threshold)
    out = list(pred)
    stats = Counter()
    changed = 0
    for idx, record_decisions in decisions.items():
        if not (0 <= idx < len(out)):
            continue
        before = json.dumps(out[idx], ensure_ascii=False, sort_keys=True)
        out[idx], change = APPLY.apply_strategy_to_record(
            out[idx],
            pred[idx],
            record_decisions,
            postprocess,
            strategy,
        )
        after = json.dumps(out[idx], ensure_ascii=False, sort_keys=True)
        if before != after:
            changed += 1
        stats.update(change)
    return out, dict(sorted(stats.items())), changed


def entity_key(entity):
    return int(entity["start"]), int(entity["end"]), str(entity["label"])


def relation_key(relation):
    return (
        int(relation["head_start"]),
        int(relation["head_end"]),
        str(relation["head_type"]),
        int(relation["tail_start"]),
        int(relation["tail_end"]),
        str(relation["tail_type"]),
        str(relation["label"]),
    )


def endpoints_exist(relation, entity_keys):
    return (
        (int(relation["head_start"]), int(relation["head_end"]), str(relation["head_type"])) in entity_keys
        and (int(relation["tail_start"]), int(relation["tail_end"]), str(relation["tail_type"])) in entity_keys
    )


def augment_base(trial, base, entity_rows, entity_decisions, relation_decisions):
    additions = [[] for _ in base]
    entity_apply = Counter()
    entity_by_label = Counter()
    missing_entity_judgments = 0
    for row in entity_rows:
        if row["trial"] != trial:
            continue
        decisions = entity_decisions.get(row["candidate_id"], [])
        if not decisions:
            missing_entity_judgments += 1
            continue
        decision = ent_aug.choose_decision(decisions, 0.92, 0.92)
        action = ent_aug.normalize_action(decision.get("action"))
        entity = None
        if action == "keep":
            entity = {"start": row["start"], "end": row["end"], "text": row["text"], "label": row["label"]}
        elif action == "fix" and ent_aug.valid_corrected_entity(decision.get("corrected"), base[row["record_index"]]["text"]):
            entity = ent_aug.resolve_corrected_entity(decision["corrected"], base[row["record_index"]]["text"])
        if entity:
            additions[row["record_index"]].append(entity)
            entity_apply["accepted"] += 1
            entity_by_label[entity["label"]] += 1
        else:
            entity_apply["rejected"] += 1

    entity_augmented, changed_entities = ent_aug.add_entities(base, additions)
    entity_apply["changed_entities"] = changed_entities

    union = load_json(pair_dir(trial, "union") / "filtered_pred.json")
    relation_rows = []
    relation_candidates = Counter()
    skipped_missing_endpoint = Counter()
    for record_index, (union_row, base_row) in enumerate(zip(union, entity_augmented)):
        existing = {relation_key(r) for r in base_row.get("relations", []) or []}
        ents = {entity_key(e) for e in base_row.get("entities", []) or []}
        seen = set(existing)
        for relation in union_row.get("relations", []) or []:
            key = relation_key(relation)
            if key in seen:
                continue
            seen.add(key)
            label = str(relation.get("label", ""))
            if not endpoints_exist(relation, ents):
                skipped_missing_endpoint[label] += 1
                continue
            relation_rows.append(
                {
                    "candidate_id": rel_aug.relation_id(trial, record_index, relation),
                    "trial": trial,
                    "record_index": record_index,
                    "label": label,
                    "head": relation.get("head", ""),
                    "head_start": int(relation["head_start"]),
                    "head_end": int(relation["head_end"]),
                    "head_type": str(relation["head_type"]),
                    "tail": relation.get("tail", ""),
                    "tail_start": int(relation["tail_start"]),
                    "tail_end": int(relation["tail_end"]),
                    "tail_type": str(relation["tail_type"]),
                }
            )
            relation_candidates[label] += 1

    relation_additions = [[] for _ in entity_augmented]
    relation_apply = Counter()
    relation_by_label = Counter()
    missing_relation_judgments = 0
    for row in relation_rows:
        decisions = relation_decisions.get(row["candidate_id"], [])
        if not decisions:
            missing_relation_judgments += 1
        decision = rel_aug.choose_decision(decisions, 0.92, 0.92)
        action = ent_aug.normalize_action(decision.get("action"))
        relation = None
        if action == "keep":
            relation = {
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
        elif action == "fix":
            relation = rel_aug.resolve_corrected_relation(
                decision.get("corrected"),
                entity_augmented[row["record_index"]].get("entities", []) or [],
            )
        if relation and relation["label"] not in RELATION_OUTPUT_LABELS:
            relation = None
        ents = {entity_key(e) for e in entity_augmented[row["record_index"]].get("entities", []) or []}
        if relation and endpoints_exist(relation, ents):
            relation_additions[row["record_index"]].append(relation)
            relation_apply["accepted"] += 1
            relation_by_label[relation["label"]] += 1
        else:
            relation_apply["rejected"] += 1

    augmented, changed_relations = rel_aug.add_relations(entity_augmented, relation_additions)
    relation_apply["changed_relations"] = changed_relations
    return augmented, {
        "entity_apply": dict(sorted(entity_apply.items())),
        "entity_added_by_label": dict(sorted(entity_by_label.items())),
        "missing_entity_judgments": missing_entity_judgments,
        "relation_candidates": len(relation_rows),
        "relation_candidate_by_label": dict(sorted(relation_candidates.items())),
        "skipped_missing_endpoint": dict(sorted(skipped_missing_endpoint.items())),
        "relation_apply": dict(sorted(relation_apply.items())),
        "relation_added_by_label": dict(sorted(relation_by_label.items())),
        "missing_relation_judgments": missing_relation_judgments,
    }


def prepare(trials):
    EXP_OUT.mkdir(parents=True, exist_ok=True)
    entity_rows, _, entity_stats = ent_aug.make_candidates(trials, None)
    entity_decisions = ent_aug.collect_decisions()
    relation_decisions = rel_aug.collect_decisions()
    summaries = []

    for trial in trials:
        gold = load_json(OUT_ROOT / "s100_t10" / f"trial_{trial:02d}" / "gold.json")
        prune = load_json(pair_dir(trial, "strict3") / "prune_unseen_pred.json")

        filtered_base, filter_stats, filter_changed = apply_entity_only_ops(trial, prune)
        filtered_path = EXP_OUT / "entity_filter_base" / f"trial_{trial:02d}_prune_unseen_entity_only_ops_filtered.json"
        dump_json(filtered_path, filtered_base)
        write_zip(filtered_path, filtered_path.with_suffix(".zip"))

        aug_first, aug_first_stats = augment_base(trial, prune, entity_rows, entity_decisions, relation_decisions)
        aug_first_path = (
            EXP_OUT
            / "augment_then_entity_filter_input"
            / f"trial_{trial:02d}_prune_unseen_entity_all_relation_con_loi_augmented.json"
        )
        dump_json(aug_first_path, aug_first)
        write_zip(aug_first_path, aug_first_path.with_suffix(".zip"))

        filter_first, filter_first_stats = augment_base(trial, filtered_base, entity_rows, entity_decisions, relation_decisions)
        filter_first_path = (
            EXP_OUT
            / "entity_filter_then_augment"
            / f"trial_{trial:02d}_entity_filter_then_entity_all_relation_con_loi_augmented.json"
        )
        dump_json(filter_first_path, filter_first)
        write_zip(filter_first_path, filter_first_path.with_suffix(".zip"))

        summary = {
            "trial": trial,
            "paths": {
                "prune_unseen": str(pair_dir(trial, "strict3") / "prune_unseen_pred.json"),
                "entity_filter_base": str(filtered_path),
                "augment_then_filter_input": str(aug_first_path),
                "filter_then_augment": str(filter_first_path),
            },
            "entity_candidates": sum(1 for r in entity_rows if r["trial"] == trial),
            "entity_candidate_by_label": dict(sorted(entity_stats.items())),
            "entity_filter_base": {"applied": filter_stats, "changed_records": filter_changed},
            "augment_then_filter_input_stats": aug_first_stats,
            "filter_then_augment_stats": filter_first_stats,
            "metrics": {
                "prune_unseen": metric(gold, prune),
                "entity_filter_base": metric(gold, filtered_base),
                "augment_then_filter_input": metric(gold, aug_first),
                "filter_then_augment": metric(gold, filter_first),
            },
        }
        summaries.append(summary)
        dump_json(EXP_OUT / f"prepare_summary_trial_{trial:02d}.json", summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)

    aggregate_summary = {
        "trials": trials,
        "stages": {
            stage: aggregate([s["metrics"][stage] for s in summaries])
            for stage in ["prune_unseen", "entity_filter_base", "augment_then_filter_input", "filter_then_augment"]
        },
        "trial_summaries": summaries,
    }
    dump_json(EXP_OUT / "prepare_summary.json", aggregate_summary)
    return aggregate_summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, nargs="+", default=TRIALS)
    parser.add_argument("--prepare", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.prepare:
        args.prepare = True
    if args.prepare:
        summary = prepare(args.trials)
        print(json.dumps({"summary": str(EXP_OUT / "prepare_summary.json"), "stages": summary["stages"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
