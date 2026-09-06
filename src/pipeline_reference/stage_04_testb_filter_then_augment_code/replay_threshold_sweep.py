"""Replay existing entity/relation AI judgments with stricter thresholds."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import run_filter_then_union_testb as cfg

sys.path.insert(0, str(cfg.VALIDATION))
import filter_then_union_augment as ftu  # noqa: E402


OUT = cfg.ROOT / "outputs/test_b_filter_then_augment_threshold_sweep"
STRICT_OUT = cfg.ROOT / "outputs/test_b_filter_then_augment_threshold_sweep_strict"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def harden_decisions(decisions: dict[str, list[dict]], keep_threshold: float, fix_threshold: float) -> dict[str, list[dict]]:
    hardened: dict[str, list[dict]] = {}
    for cid, rows in decisions.items():
        new_rows = []
        for row in rows:
            action = ftu.normalize_action(row.get("action"))
            try:
                confidence = float(row.get("confidence", 0) or 0)
            except Exception:
                confidence = 0.0
            if (action == "keep" and confidence < keep_threshold) or (action == "fix" and confidence < fix_threshold):
                changed = dict(row)
                changed["original_action"] = action
                changed["action"] = "drop"
                changed["reason"] = f"below hard threshold {keep_threshold:.2f}/{fix_threshold:.2f}: " + str(row.get("reason", ""))
                new_rows.append(changed)
            else:
                new_rows.append(row)
        hardened[cid] = new_rows
    return hardened


def replay_variant(
    name: str,
    entity_threshold: float,
    relation_threshold: float,
    base: list[dict],
    pool: list[dict],
    base_stats: dict,
    entity_rows: list[dict],
    entity_decisions: dict[str, list[dict]],
    relation_decisions: dict[str, list[dict]],
    out_root: Path,
    hard_thresholds: bool,
) -> dict:
    var_dir = out_root / name
    var_dir.mkdir(parents=True, exist_ok=True)

    ent_decisions = (
        harden_decisions(entity_decisions, entity_threshold, entity_threshold)
        if hard_thresholds
        else entity_decisions
    )
    base_plus_entities, ent_stats, ent_by_label, ent_applied = ftu.add_entities(
        base,
        entity_rows,
        ent_decisions,
        entity_threshold,
        entity_threshold,
        "drop",
    )
    relation_rows, rel_candidates_by_label, skipped = ftu.make_relation_candidates(
        base_plus_entities,
        pool,
        cfg.RELATION_LABELS,
    )
    rel_decisions = (
        harden_decisions(relation_decisions, relation_threshold, relation_threshold)
        if hard_thresholds
        else relation_decisions
    )
    augmented, rel_stats, rel_by_label, rel_applied = ftu.add_relations(
        base_plus_entities,
        relation_rows,
        rel_decisions,
        relation_threshold,
        relation_threshold,
        "drop",
        set(cfg.RELATION_LABELS),
        False,
    )
    augmented = ftu.postprocess_if_available(augmented, cfg.SOLUTION_SRC)

    out_json = var_dir / f"{name}.json"
    out_zip = var_dir / f"{name}_submit.zip"
    dump_json(out_json, augmented)
    ftu.write_zip(out_json, out_zip)

    summary = {
        "name": name,
        "hard_thresholds": hard_thresholds,
        "entity_threshold": entity_threshold,
        "relation_threshold": relation_threshold,
        "base_stats": base_stats,
        "entity_augmented_stats": ftu.count_records(base_plus_entities),
        "output_stats": ftu.count_records(augmented),
        "entity_apply": dict(sorted(ent_stats.items())),
        "entity_added_by_label": dict(sorted(ent_by_label.items())),
        "relation_candidates": len(relation_rows),
        "relation_candidate_by_label": dict(sorted(rel_candidates_by_label.items())),
        "relation_skipped_missing_endpoint": dict(sorted(skipped.items())),
        "relation_apply": dict(sorted(rel_stats.items())),
        "relation_added_by_label": dict(sorted(rel_by_label.items())),
        "output_json": str(out_json),
        "output_zip": str(out_zip),
    }
    dump_json(var_dir / f"{name}_summary.json", summary)
    out_stats = summary["output_stats"]
    return {
        "name": name,
        "hard_thresholds": hard_thresholds,
        "entity_threshold": entity_threshold,
        "relation_threshold": relation_threshold,
        "entities": out_stats["entities"],
        "delta_entities": out_stats["entities"] - base_stats["entities"],
        "relations": out_stats["relations"],
        "delta_relations": out_stats["relations"] - base_stats["relations"],
        "entity_changed_pre_post": ent_stats.get("changed_entities", 0),
        "relation_changed": rel_stats.get("changed_relations", 0),
        "empty": out_stats["empty"],
        "bad_relation_endpoints": out_stats["bad_relation_endpoints"],
        "zip": str(out_zip),
    }


def main() -> None:
    base = ftu.load_records(cfg.BASE)
    pool = ftu.load_records(cfg.POOL)
    entity_rows = load_json(cfg.WORK_OUT / "entity_candidates.json")
    entity_decisions = ftu.collect_decisions(cfg.WORK_OUT / "entity_ai_judgments.jsonl")
    relation_decisions = ftu.collect_decisions(cfg.WORK_OUT / "relation_ai_judgments.jsonl")
    base_stats = ftu.count_records(base)

    variants = [
        ("e092_r092_baseline_replay", 0.92, 0.92),
        ("e093_r093", 0.93, 0.93),
        ("e094_r094", 0.94, 0.94),
        ("e095_r095", 0.95, 0.95),
        ("e094_r092_entity_stricter", 0.94, 0.92),
        ("e092_r094_relation_stricter", 0.92, 0.94),
        ("e093_r092_entity_light_stricter", 0.93, 0.92),
        ("e092_r093_relation_light_stricter", 0.92, 0.93),
    ]

    compact = []
    for name, entity_threshold, relation_threshold in variants:
        compact.append(
            replay_variant(
                name,
                entity_threshold,
                relation_threshold,
                base,
                pool,
                base_stats,
                entity_rows,
                entity_decisions,
                relation_decisions,
                OUT,
                False,
            )
        )

    dump_json(OUT / "threshold_sweep_compare.json", compact)

    strict_variants = [
        ("hard_e092_r092", 0.92, 0.92),
        ("hard_e093_r093", 0.93, 0.93),
        ("hard_e094_r094", 0.94, 0.94),
        ("hard_e095_r095", 0.95, 0.95),
        ("hard_e096_r096", 0.96, 0.96),
        ("hard_e094_r092_entity_stricter", 0.94, 0.92),
        ("hard_e092_r094_relation_stricter", 0.92, 0.94),
        ("hard_e093_r092_entity_light_stricter", 0.93, 0.92),
        ("hard_e092_r093_relation_light_stricter", 0.92, 0.93),
    ]
    strict_compact = []
    for name, entity_threshold, relation_threshold in strict_variants:
        strict_compact.append(
            replay_variant(
                name,
                entity_threshold,
                relation_threshold,
                base,
                pool,
                base_stats,
                entity_rows,
                entity_decisions,
                relation_decisions,
                STRICT_OUT,
                True,
            )
        )

    dump_json(STRICT_OUT / "threshold_sweep_compare.json", strict_compact)
    print(json.dumps({"soft": compact, "hard": strict_compact}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
