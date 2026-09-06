"""Run one more OPS filter after the second augmentation layer.

This is a thin wrapper around the existing sharded OPS runner. It registers
second-layer prediction files as temporary variants, then reuses the normal
online judgment, merge, apply, and metric code.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import run_holdout_ops_filter as ops
import run_holdout_ops_filter_sharded as sharded
from env_loader import api_key_env_for_model


PROJECT = Path(__file__).resolve().parent
LAYER_ROOT = PROJECT / "outputs" / "entity_ops_then_all_relation_layer"
DEFAULT_USER_OUT = Path(
    "C:/Users/Zzl410410/Documents/Codex/2026-06-28/fnag/outputs/"
    "extra_ops_after_second_layer_span_only_sample_3trials"
)


def parse_bool(text):
    value = str(text).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid bool value: {text}")


VARIANT_PATHS = {
    "second_layer_span_only_sample": {
        0: LAYER_ROOT
        / "second_layer_span_only_sample_00_07"
        / "trial_00_second_entity_all_relation_all_augmented.json",
        7: LAYER_ROOT
        / "second_layer_span_only_sample_00_07"
        / "trial_07_second_entity_all_relation_all_augmented.json",
        8: LAYER_ROOT
        / "second_layer_span_only_sample_08"
        / "trial_08_second_entity_all_relation_all_augmented.json",
    },
    "extra_ops1_span_only_sample": {
        trial: LAYER_ROOT
        / "extra_ops_after_second_layer_span_only_sample_3trials"
        / "second_layer_span_only_sample"
        / f"trial_{trial:02d}"
        / f"second_layer_span_only_sample_trial_{trial:02d}_ops_filtered.json"
        for trial in [0, 7, 8]
    },
    "second_layer_text_label_5fold": {
        trial: LAYER_ROOT
        / "second_entity_relation_all_after_final_ops"
        / f"trial_{trial:02d}_second_entity_all_relation_all_augmented.json"
        for trial in [0, 1, 4, 6, 7]
    },
    "second_layer_text_label_holdout_08_09": {
        trial: LAYER_ROOT
        / "second_entity_relation_all_after_final_ops_holdout_08_09"
        / f"trial_{trial:02d}_second_entity_all_relation_all_augmented.json"
        for trial in [8, 9]
    },
}

DEFAULT_TRIALS = {
    "second_layer_span_only_sample": [0, 7, 8],
    "extra_ops1_span_only_sample": [0, 7, 8],
    "second_layer_text_label_5fold": [0, 1, 4, 6, 7],
    "second_layer_text_label_holdout_08_09": [8, 9],
}


def dump_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def stage_line(label, metric):
    return (
        f"| {label} | {metric['mean_total']:.6f} | {metric['mean_ner']:.6f} | "
        f"{metric['mean_re']:.6f} | "
        f"{metric['entity_tp']}/{metric['entity_fp']}/{metric['entity_fn']} | "
        f"{metric['relation_tp']}/{metric['relation_fp']}/{metric['relation_fn']} | "
        f"{metric['entities']} | {metric['relations']} |"
    )


def render_extra_report(path, summary, variant):
    row = summary["variants"][variant]
    delta = row["delta_after_before"]
    strategy = summary["strategy"]
    if strategy.get("entity_policy") == "keep_all" and not strategy.get("apply_entity_fix"):
        ops_desc = f"relation-only OPS(relation={strategy['relation_drop_threshold']:.2f})"
    else:
        ops_desc = (
            f"OPS(entity={strategy['entity_drop_threshold']:.2f}, "
            f"relation={strategy['relation_drop_threshold']:.2f})"
        )
    if variant.startswith("extra_ops1_"):
        pipeline_tail = (
            "-> second entity all + relation all augment -> extra OPS #1(entity=0.75, relation=0.95) "
            f"-> extra {ops_desc}."
        )
        before_label = "before extra OPS"
        after_label = "after extra OPS"
    else:
        pipeline_tail = (
            f"-> second entity all + relation all augment -> extra {ops_desc}."
        )
        before_label = "before extra OPS"
        after_label = "after extra OPS"
    lines = [
        "# Extra OPS Layer",
        "",
        f"Variant: `{variant}`.",
        f"Trials: {', '.join(str(t) for t in summary['trials'])}.",
        "",
        "Pipeline: strict3 prune_unseen -> entity OPS -> entity all + relation all augment "
        f"-> final OPS(entity=0.75, relation=0.95) {pipeline_tail}",
        "",
        "| stage | total | NER | RE | entity TP/FP/FN | relation TP/FP/FN | entities | relations |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        stage_line(before_label, row["before"]),
        stage_line(after_label, row["after"]),
        "",
        "## Delta",
        "",
        "| comparison | total | NER | RE | entity TP | entity FP | entity FN | relation TP | relation FP | relation FN | entities | relations |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| after - before | {delta['total']:+.6f} | {delta['ner']:+.6f} | "
            f"{delta['re']:+.6f} | {delta['entity_tp']:+d} | {delta['entity_fp']:+d} | "
            f"{delta['entity_fn']:+d} | {delta['relation_tp']:+d} | "
            f"{delta['relation_fp']:+d} | {delta['relation_fn']:+d} | "
            f"{delta['entities']:+d} | {delta['relations']:+d} |"
        ),
        "",
        "## Per Trial",
        "",
        "| trial | stage | total | NER | RE | entity TP/FP/FN | relation TP/FP/FN | entities | relations |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for trial_row in row["trials"]:
        for stage in ["before", "after"]:
            metric = trial_row[stage]
            lines.append(
                f"| {trial_row['trial']} | {stage} | {metric['total']:.6f} | "
                f"{metric['ner']:.6f} | {metric['re']:.6f} | "
                f"{metric['entity']['tp']}/{metric['entity']['fp']}/{metric['entity']['fn']} | "
                f"{metric['relation']['tp']}/{metric['relation']['fp']}/{metric['relation']['fn']} | "
                f"{metric['entities']} | {metric['relations']} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def variant_paths_from_summary(summary_path, source_variant):
    summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    if source_variant is None:
        variants = list(summary.get("variants", {}))
        if len(variants) != 1:
            raise SystemExit(
                "--source_variant is required when source summary has multiple variants: "
                + ", ".join(variants)
            )
        source_variant = variants[0]
    rows = summary["variants"][source_variant]["trials"]
    return {int(row["trial"]): Path(row["output_json"]) for row in rows}


def copy_user_outputs(summary, user_out):
    user_out = Path(user_out)
    user_out.mkdir(parents=True, exist_ok=True)
    for name in [
        "ops_filter_sharded_summary.json",
        "OPS_FILTER_SHARDED_SUMMARY.md",
        "EXTRA_OPS_AFTER_SECOND_LAYER_SUMMARY.md",
    ]:
        src = ops.OPS_OUT / name
        if src.exists():
            shutil.copy2(src, user_out / name)

    for variant, variant_row in summary.get("variants", {}).items():
        for trial_row in variant_row.get("trials", []):
            trial_dir = user_out / variant / f"trial_{trial_row['trial']:02d}"
            trial_dir.mkdir(parents=True, exist_ok=True)
            for key in ["output_json", "output_zip", "filter_summary", "strategy", "judgments"]:
                src = Path(trial_row[key])
                if src.exists():
                    shutil.copy2(src, trial_dir / src.name)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="second_layer_span_only_sample")
    parser.add_argument(
        "--source_summary",
        help="Use output_json files from an earlier ops_filter_sharded_summary.json as this layer's inputs.",
    )
    parser.add_argument("--source_variant")
    parser.add_argument("--run_variant_name")
    parser.add_argument("--trials", type=int, nargs="+", default=None)
    parser.add_argument("--skip_online", action="store_true")
    parser.add_argument("--allow_partial", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--shard_size", type=int, default=5)
    parser.add_argument("--shard_attempts", type=int, default=3)
    parser.add_argument("--retry_sleep", type=int, default=15)
    parser.add_argument("--model", default=ops.os.getenv("GEMINI_MODEL", "gemini-3.1-pro-preview"))
    parser.add_argument("--base_url", default=ops.os.getenv("GEMINI_BASE_URL", ops.os.getenv("BASE_URL", "https://zjapi.com")))
    parser.add_argument("--api_key_env", default=None)
    parser.add_argument("--timeout", type=int, default=int(ops.os.getenv("OPUS_TIMEOUT", "180")))
    parser.add_argument("--retries", type=int, default=int(ops.os.getenv("OPUS_RETRIES", "2")))
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--batch_items", type=int, default=40)
    parser.add_argument("--entity_policy", default="drop_threshold", choices=["keep_all", "any_drop", "drop_threshold", "majority_drop", "majority_or_threshold"])
    parser.add_argument("--entity_drop_threshold", type=float, default=0.75)
    parser.add_argument("--apply_entity_fix", type=parse_bool, default=True)
    parser.add_argument("--entity_fix_threshold", type=float, default=0.0)
    parser.add_argument("--relation_policy", default="drop_threshold", choices=["keep_all", "drop_threshold"])
    parser.add_argument("--relation_drop_threshold", type=float, default=0.95)
    parser.add_argument("--apply_relation_fix", type=parse_bool, default=False)
    parser.add_argument("--relation_fix_threshold", type=float, default=0.95)
    parser.add_argument("--protect_relation_endpoints", type=parse_bool, default=False)
    parser.add_argument("--noempty_fallback", type=parse_bool, default=True)
    parser.add_argument(
        "--out_dir",
        default=str(LAYER_ROOT / "extra_ops_after_second_layer_span_only_sample_3trials"),
    )
    parser.add_argument("--user_out", default=str(DEFAULT_USER_OUT))
    parser.add_argument("--copy_outputs", action="store_true")
    args = parser.parse_args()
    if args.trials is None and not args.source_summary:
        args.trials = DEFAULT_TRIALS[args.variant]
    if not args.api_key_env:
        args.api_key_env = api_key_env_for_model(args.model)
    return args


def main():
    args = parse_args()
    if args.source_summary:
        variant_paths = variant_paths_from_summary(args.source_summary, args.source_variant)
        variant = args.run_variant_name or f"{args.source_variant or 'source'}_next_ops"
        if args.trials is None:
            args.trials = sorted(variant_paths)
    else:
        if args.variant not in VARIANT_PATHS:
            raise SystemExit(f"unknown variant: {args.variant}")
        variant_paths = VARIANT_PATHS[args.variant]
        variant = args.variant
    missing = [str(variant_paths[trial]) for trial in args.trials if not variant_paths[trial].exists()]
    if missing:
        raise SystemExit("missing prediction(s):\n" + "\n".join(missing))

    ops.OPS_OUT = Path(args.out_dir)
    ops.VARIANT_PATHS[variant] = variant_paths

    runner_args = SimpleNamespace(
        variants=[variant],
        trials=args.trials,
        skip_online=args.skip_online,
        allow_partial=args.allow_partial,
        workers=args.workers,
        shard_size=args.shard_size,
        shard_attempts=args.shard_attempts,
        retry_sleep=args.retry_sleep,
        model=args.model,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        timeout=args.timeout,
        retries=args.retries,
        max_tokens=args.max_tokens,
        batch_items=args.batch_items,
        entity_policy=args.entity_policy,
        entity_drop_threshold=args.entity_drop_threshold,
        apply_entity_fix=args.apply_entity_fix,
        entity_fix_threshold=args.entity_fix_threshold,
        relation_policy=args.relation_policy,
        relation_drop_threshold=args.relation_drop_threshold,
        apply_relation_fix=args.apply_relation_fix,
        relation_fix_threshold=args.relation_fix_threshold,
        protect_relation_endpoints=args.protect_relation_endpoints,
        noempty_fallback=args.noempty_fallback,
        copy_outputs=False,
    )
    online_results = [] if args.skip_online else sharded.run_online(runner_args)
    summary = sharded.finalize(runner_args, online_results)
    summary["trials"] = args.trials
    dump_json(ops.OPS_OUT / "ops_filter_sharded_summary.json", summary)
    render_extra_report(ops.OPS_OUT / "EXTRA_OPS_AFTER_SECOND_LAYER_SUMMARY.md", summary, variant)
    if args.copy_outputs:
        copy_user_outputs(summary, args.user_out)
    print(
        json.dumps(
            {
                "summary": str(ops.OPS_OUT / "ops_filter_sharded_summary.json"),
                "report": str(ops.OPS_OUT / "EXTRA_OPS_AFTER_SECOND_LAYER_SUMMARY.md"),
                "variant": variant,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
