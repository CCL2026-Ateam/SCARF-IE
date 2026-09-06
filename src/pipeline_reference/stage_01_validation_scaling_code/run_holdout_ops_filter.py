"""Run a second-stage Opus/Gemini item filter on holdout augmented outputs."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from env_loader import api_key_env_for_model, default_env_paths, load_env_defaults


PROJECT = Path(__file__).resolve().parent
load_env_defaults(*default_env_paths(PROJECT))
OUT_ROOT = PROJECT / "outputs"
S100_T10 = OUT_ROOT / "s100_t10"
OPS_OUT = Path(os.getenv("OPS_FILTER_OUT_DIR") or (OUT_ROOT / "holdout_08_09_ops_filter"))
SOLUTION = Path(os.getenv("SOLUTION_ROOT", "K:/\u6d69\u7136/CCL/solution/solution"))
SOLUTION_SRC = SOLUTION / "src"
JUDGE_SCRIPT = PROJECT.parent / "test_b_original1000_4member" / "src" / "judge_json_items_opus.py"
FILTER_WORKSPACE = Path(
    "C:/Users/Zzl410410/Documents/Codex/2026-06-26/"
    "019ef541-55e7-7af1-8f34-0927a521fe82"
)
APPLY_STRATEGY = FILTER_WORKSPACE / "work" / "apply_entity_pair_strategy.py"


VARIANT_PATHS = {
    "all_entities_all_relations": {
        8: OUT_ROOT / "holdout_08_09_relation_ai_all" / "trial_08_strict3_entity_ai_all_relation_ai_augmented.json",
        9: OUT_ROOT / "holdout_08_09_relation_ai_all" / "trial_09_strict3_entity_ai_all_relation_ai_augmented.json",
    },
    "all_entities_partial_relations": {
        8: OUT_ROOT / "holdout_08_09_relation_ai_partial" / "trial_08_strict3_entity_ai_all_relation_ai_augmented.json",
        9: OUT_ROOT / "holdout_08_09_relation_ai_partial" / "trial_09_strict3_entity_ai_all_relation_ai_augmented.json",
    },
    "prune_unseen_all_entities_partial_relations": {
        8: OUT_ROOT / "prune_unseen_augment_then_ops_trial08" / "trial_08_strict3_prune_unseen_entity_ai_all_relation_con_loi_augmented.json",
        9: OUT_ROOT / "prune_unseen_augment_then_ops_trial08" / "trial_09_strict3_prune_unseen_entity_ai_all_relation_con_loi_augmented.json",
    },
    "val_prune_unseen_augment_all_entities_partial_relations": {
        0: OUT_ROOT / "order_swap_entityonly_filter" / "augment_then_entity_filter_input" / "trial_00_prune_unseen_entity_all_relation_con_loi_augmented.json",
        1: OUT_ROOT / "order_swap_entityonly_filter" / "augment_then_entity_filter_input" / "trial_01_prune_unseen_entity_all_relation_con_loi_augmented.json",
        4: OUT_ROOT / "order_swap_entityonly_filter" / "augment_then_entity_filter_input" / "trial_04_prune_unseen_entity_all_relation_con_loi_augmented.json",
        6: OUT_ROOT / "order_swap_entityonly_filter" / "augment_then_entity_filter_input" / "trial_06_prune_unseen_entity_all_relation_con_loi_augmented.json",
        7: OUT_ROOT / "order_swap_entityonly_filter" / "augment_then_entity_filter_input" / "trial_07_prune_unseen_entity_all_relation_con_loi_augmented.json",
    },
    "trial00_entity_ops_then_entity_all_relation_all": {
        0: OUT_ROOT / "entity_ops_then_all_relation_layer" / "trial_00_entity_ops_then_entity_all_relation_all_augmented.json",
    },
    "val_entity_ops_then_entity_all_relation_all": {
        0: OUT_ROOT / "entity_ops_then_all_relation_layer" / "trial_00_entity_ops_then_entity_all_relation_all_augmented.json",
        1: OUT_ROOT / "entity_ops_then_all_relation_layer" / "trial_01_entity_ops_then_entity_all_relation_all_augmented.json",
        4: OUT_ROOT / "entity_ops_then_all_relation_layer" / "trial_04_entity_ops_then_entity_all_relation_all_augmented.json",
        6: OUT_ROOT / "entity_ops_then_all_relation_layer" / "trial_06_entity_ops_then_entity_all_relation_all_augmented.json",
        7: OUT_ROOT / "entity_ops_then_all_relation_layer" / "trial_07_entity_ops_then_entity_all_relation_all_augmented.json",
    },
    "holdout_entity_ops_then_entity_all_relation_all": {
        8: OUT_ROOT / "entity_ops_then_all_relation_layer" / "trial_08_entity_ops_then_entity_all_relation_all_augmented.json",
        9: OUT_ROOT / "entity_ops_then_all_relation_layer" / "trial_09_entity_ops_then_entity_all_relation_all_augmented.json",
    },
    "first_layer_span_only_entity_ops_then_entity_all_relation_all_sample": {
        0: OUT_ROOT / "entity_ops_then_all_relation_layer" / "first_layer_span_only_sample_00_07" / "trial_00_entity_ops_then_entity_all_relation_all_augmented.json",
        7: OUT_ROOT / "entity_ops_then_all_relation_layer" / "first_layer_span_only_sample_00_07" / "trial_07_entity_ops_then_entity_all_relation_all_augmented.json",
        8: OUT_ROOT / "entity_ops_then_all_relation_layer" / "first_layer_span_only_sample_08" / "trial_08_entity_ops_then_entity_all_relation_all_augmented.json",
    },
    "partial_entities_partial_relations": {
        8: OUT_ROOT / "holdout_08_09_entity_partial_relation_partial_v2" / "trial_08_strict3_entity_ai_partial_relation_ai_partial_augmented.json",
        9: OUT_ROOT / "holdout_08_09_entity_partial_relation_partial_v2" / "trial_09_strict3_entity_ai_partial_relation_ai_partial_augmented.json",
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
    path = Path(path)
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def expected_judgment_rows(records):
    return sum(1 for r in records if r.get("entities") or r.get("relations"))


def count_judgment_rows(path):
    seen = set()
    for row in load_jsonl(path):
        seen.add((row.get("record_index"), row.get("batch_index")))
    return len(seen)


def file_slug(path):
    import re

    return re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(path).stem)


def judgment_path(out_dir, pred_path):
    return Path(out_dir) / "judgments" / f"{file_slug(pred_path)}.opus_item_judgments.jsonl"


def trial_out_dir(variant, trial):
    return OPS_OUT / variant / f"trial_{trial:02d}"


def metric_row(gold, pred):
    sys.path.insert(0, str(SOLUTION_SRC))
    from common import evaluate  # noqa: E402

    m = evaluate(gold, pred)
    return {
        "total": m["total_score"],
        "ner": m["score_ner"],
        "re": m["score_re"],
        "entity": m["entity"],
        "relation": m["relation"],
        "entities": sum(len(r.get("entities", []) or []) for r in pred),
        "relations": sum(len(r.get("relations", []) or []) for r in pred),
    }


def summarize_judgments(path):
    actions = Counter()
    kinds = Counter()
    parse_errors = 0
    decisions = 0
    for row in load_jsonl(path):
        if row.get("parse_error"):
            parse_errors += 1
        for decision in row.get("decisions", []) or []:
            decisions += 1
            kinds[str(decision.get("kind", "unknown"))] += 1
            actions[str(decision.get("action", "unknown")).lower()] += 1
    return {
        "rows": count_judgment_rows(path),
        "decisions": decisions,
        "parse_errors": parse_errors,
        "by_kind": dict(sorted(kinds.items())),
        "by_action": dict(sorted(actions.items())),
    }


def run_judge(variant, trial, pred_path, args):
    out_dir = trial_out_dir(variant, trial)
    jpath = judgment_path(out_dir, pred_path)
    records = load_json(pred_path)
    expected = expected_judgment_rows(records)
    if count_judgment_rows(jpath) >= expected:
        return {
            "variant": variant,
            "trial": trial,
            "status": "skipped_complete",
            "expected_rows": expected,
            "judgment_rows": count_judgment_rows(jpath),
            "judgments": str(jpath),
        }

    env = os.environ.copy()
    cmd = [
        sys.executable,
        str(JUDGE_SCRIPT),
        "--inputs",
        str(pred_path),
        "--output_dir",
        str(out_dir / "judgments"),
        "--model",
        args.model,
        "--base_url",
        args.base_url,
        "--api_key_env",
        args.api_key_env,
        "--timeout",
        str(args.timeout),
        "--retries",
        str(args.retries),
        "--max_tokens",
        str(args.max_tokens),
        "--batch_items",
        str(args.batch_items),
    ]
    log = out_dir / "online_judge.stdout.log"
    err = out_dir / "online_judge.stderr.log"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(log, "a", encoding="utf-8") as lf, open(err, "a", encoding="utf-8") as ef:
        subprocess.run(cmd, check=True, cwd=str(PROJECT.parent), env=env, stdout=lf, stderr=ef)
    return {
        "variant": variant,
        "trial": trial,
        "status": "ran",
        "expected_rows": expected,
        "judgment_rows": count_judgment_rows(jpath),
        "judgments": str(jpath),
        "stdout_log": str(log),
        "stderr_log": str(err),
    }


def apply_filter(variant, trial, pred_path, args):
    out_dir = trial_out_dir(variant, trial)
    jpath = judgment_path(out_dir, pred_path)
    output_json = out_dir / f"{variant}_trial_{trial:02d}_ops_filtered.json"
    output_zip = out_dir / f"{variant}_trial_{trial:02d}_ops_filtered.zip"
    summary_json = out_dir / f"{variant}_trial_{trial:02d}_ops_filter_summary.json"
    strategy_json = out_dir / f"{variant}_trial_{trial:02d}_ops_strategy.json"
    entity_policy = getattr(args, "entity_policy", "drop_threshold")
    apply_entity_fix = getattr(args, "apply_entity_fix", True)
    entity_fix_threshold = getattr(args, "entity_fix_threshold", 0.0)
    relation_policy = getattr(args, "relation_policy", "drop_threshold")
    apply_relation_fix = getattr(args, "apply_relation_fix", False)
    relation_fix_threshold = getattr(args, "relation_fix_threshold", 0.95)
    protect_relation_endpoints = getattr(args, "protect_relation_endpoints", False)
    noempty_fallback = getattr(args, "noempty_fallback", True)
    cmd = [
        sys.executable,
        str(APPLY_STRATEGY),
        "--pred",
        str(pred_path),
        "--judgments",
        str(jpath),
        "--solution_src",
        str(SOLUTION_SRC),
        "--output_json",
        str(output_json),
        "--output_zip",
        str(output_zip),
        "--summary",
        str(summary_json),
        "--preset",
        "safe_anydrop_rel098",
        "--entity_policy",
        entity_policy,
        "--entity_drop_threshold",
        str(args.entity_drop_threshold),
        "--apply_entity_fix",
        str(bool(apply_entity_fix)).lower(),
        "--entity_fix_threshold",
        str(entity_fix_threshold),
        "--relation_policy",
        relation_policy,
        "--relation_drop_threshold",
        str(args.relation_drop_threshold),
        "--apply_relation_fix",
        str(bool(apply_relation_fix)).lower(),
        "--relation_fix_threshold",
        str(relation_fix_threshold),
        "--protect_relation_endpoints",
        str(bool(protect_relation_endpoints)).lower(),
        "--noempty_fallback",
        str(bool(noempty_fallback)).lower(),
        "--write_strategy_json",
        str(strategy_json),
    ]
    subprocess.run(cmd, check=True, cwd=str(FILTER_WORKSPACE), capture_output=True, text=True)
    gold = load_json(S100_T10 / f"trial_{trial:02d}" / "gold.json")
    before = metric_row(gold, load_json(pred_path))
    after = metric_row(gold, load_json(output_json))
    return {
        "variant": variant,
        "trial": trial,
        "input_json": str(pred_path),
        "judgments": str(jpath),
        "judgment_summary": summarize_judgments(jpath),
        "output_json": str(output_json),
        "output_zip": str(output_zip),
        "filter_summary": str(summary_json),
        "strategy": str(strategy_json),
        "before": before,
        "after": after,
        "delta": delta_metric(after, before),
    }


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


def aggregate(rows, key):
    selected = [row[key] for row in rows]
    return {
        "mean_total": sum(r["total"] for r in selected) / len(selected),
        "mean_ner": sum(r["ner"] for r in selected) / len(selected),
        "mean_re": sum(r["re"] for r in selected) / len(selected),
        "entity_tp": sum(r["entity"]["tp"] for r in selected),
        "entity_fp": sum(r["entity"]["fp"] for r in selected),
        "entity_fn": sum(r["entity"]["fn"] for r in selected),
        "relation_tp": sum(r["relation"]["tp"] for r in selected),
        "relation_fp": sum(r["relation"]["fp"] for r in selected),
        "relation_fn": sum(r["relation"]["fn"] for r in selected),
        "entities": sum(r["entities"] for r in selected),
        "relations": sum(r["relations"] for r in selected),
    }


def render_report(path, summary):
    lines = [
        "# Holdout 08/09 Ops Filter Summary",
        "",
        f"Strategy: entity drop >= {summary['strategy']['entity_drop_threshold']}, relation drop >= {summary['strategy']['relation_drop_threshold']}, entity fixes on, relation fixes off.",
        "",
        "| variant | stage | total | NER | RE | entity TP/FP/FN | relation TP/FP/FN | entities | relations |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant, row in summary["variants"].items():
        for stage in ["before", "after"]:
            m = row[stage]
            lines.append(
                f"| `{variant}` | {stage} | {m['mean_total']:.6f} | {m['mean_ner']:.6f} | {m['mean_re']:.6f} | "
                f"{m['entity_tp']}/{m['entity_fp']}/{m['entity_fn']} | "
                f"{m['relation_tp']}/{m['relation_fp']}/{m['relation_fn']} | "
                f"{m['entities']} | {m['relations']} |"
            )
    lines += [
        "",
        "## Deltas",
        "",
        "| variant | total | NER | RE | entity TP | entity FP | relation TP | relation FP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant, row in summary["variants"].items():
        d = row["delta_after_before"]
        lines.append(
            f"| `{variant}` | {d['total']:+.6f} | {d['ner']:+.6f} | {d['re']:+.6f} | "
            f"{d['entity_tp']:+d} | {d['entity_fp']:+d} | {d['relation_tp']:+d} | {d['relation_fp']:+d} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", default=["all_entities_all_relations", "all_entities_partial_relations"])
    parser.add_argument("--trials", type=int, nargs="+", default=[8, 9])
    parser.add_argument("--skip_online", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--model", default=os.getenv("GEMINI_MODEL", "gemini-3.1-pro-preview"))
    parser.add_argument("--base_url", default=os.getenv("GEMINI_BASE_URL", os.getenv("BASE_URL", "https://zjapi.com")))
    parser.add_argument("--api_key_env", default=None)
    parser.add_argument("--timeout", type=int, default=int(os.getenv("OPUS_TIMEOUT", "240")))
    parser.add_argument("--retries", type=int, default=int(os.getenv("OPUS_RETRIES", "3")))
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--batch_items", type=int, default=40)
    parser.add_argument("--entity_drop_threshold", type=float, default=0.75)
    parser.add_argument("--relation_drop_threshold", type=float, default=0.95)
    args = parser.parse_args()
    if not args.api_key_env:
        args.api_key_env = api_key_env_for_model(args.model)
    return args


def main():
    args = parse_args()
    jobs = []
    for variant in args.variants:
        if variant not in VARIANT_PATHS:
            raise SystemExit(f"unknown variant: {variant}")
        for trial in args.trials:
            pred_path = VARIANT_PATHS[variant][trial]
            if not pred_path.exists():
                raise SystemExit(f"missing prediction: {pred_path}")
            jobs.append((variant, trial, pred_path))

    judge_results = []
    if not args.skip_online:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(run_judge, variant, trial, pred_path, args): (variant, trial) for variant, trial, pred_path in jobs}
            for fut in as_completed(futures):
                result = fut.result()
                judge_results.append(result)
                print(json.dumps(result, ensure_ascii=False), flush=True)

    apply_rows = []
    for variant, trial, pred_path in jobs:
        row = apply_filter(variant, trial, pred_path, args)
        apply_rows.append(row)
        print(json.dumps({"variant": variant, "trial": trial, "after": row["after"], "delta": row["delta"]}, ensure_ascii=False), flush=True)

    variants = {}
    for variant in args.variants:
        rows = [row for row in apply_rows if row["variant"] == variant]
        before = aggregate(rows, "before")
        after = aggregate(rows, "after")
        variants[variant] = {
            "before": before,
            "after": after,
            "delta_after_before": aggregate_delta(after, before),
            "trials": rows,
        }

    summary = {
        "strategy": {
            "entity_drop_threshold": args.entity_drop_threshold,
            "relation_drop_threshold": args.relation_drop_threshold,
            "apply_entity_fix": True,
            "apply_relation_fix": False,
            "model": args.model,
            "base_url": args.base_url,
        },
        "judge_results": judge_results,
        "variants": variants,
    }
    OPS_OUT.mkdir(parents=True, exist_ok=True)
    dump_json(OPS_OUT / "ops_filter_summary.json", summary)
    render_report(OPS_OUT / "OPS_FILTER_SUMMARY.md", summary)
    print(json.dumps({"summary": str(OPS_OUT / "ops_filter_summary.json"), "report": str(OPS_OUT / "OPS_FILTER_SUMMARY.md")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
