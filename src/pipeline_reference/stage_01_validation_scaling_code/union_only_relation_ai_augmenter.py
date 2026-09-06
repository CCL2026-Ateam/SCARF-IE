"""CLI for AI-audited union-only relation augmentation.

The base is strict3 plus AI-reviewed union-only entity additions. Relations are
taken from union-only candidates whose endpoints exist in that entity-augmented
base.
"""
from __future__ import annotations

import argparse
import json
import os

import relation_aug_core as core
from env_loader import api_key_env_for_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, nargs="+", default=core.DEFAULT_TRIALS)
    parser.add_argument("--entity_labels", nargs="+", default=None)
    parser.add_argument("--relation_labels", nargs="+", default=None)
    parser.add_argument("--output_relation_labels", nargs="+", default=None)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--online", action="store_true")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--model", default=os.getenv("GEMINI_MODEL", "gemini-3.1-pro-preview"))
    parser.add_argument("--base_url", default=os.getenv("GEMINI_BASE_URL", os.getenv("BASE_URL", "https://zjapi.com")))
    parser.add_argument("--api_key_env", default=None)
    parser.add_argument("--timeout", type=int, default=int(os.getenv("OPUS_TIMEOUT", "240")))
    parser.add_argument("--retries", type=int, default=int(os.getenv("OPUS_RETRIES", "3")))
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--batch_items", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--examples_per_label", type=int, default=3)
    parser.add_argument("--no_label_examples", action="store_true")
    parser.add_argument("--entity_keep_threshold", type=float, default=0.92)
    parser.add_argument("--entity_fix_threshold", type=float, default=0.92)
    parser.add_argument("--keep_threshold", type=float, default=0.92)
    parser.add_argument("--fix_threshold", type=float, default=0.92)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--keep_raw", action="store_true")
    args = parser.parse_args()
    if not args.api_key_env:
        args.api_key_env = api_key_env_for_model(args.model)
    return args


def main():
    args = parse_args()
    if not any([args.prepare, args.online, args.apply, args.summary]):
        args.prepare = True
    if args.prepare:
        rows, by_record, stats, skipped, _, entity_summary = core.write_candidate_files(
            args.trials,
            args.relation_labels,
            args.entity_keep_threshold,
            args.entity_fix_threshold,
            args.entity_labels,
        )
        print(
            json.dumps(
                {
                    "out_dir": str(core.OUT_DIR),
                    "candidates": len(rows),
                    "records_with_candidates": len(by_record),
                    "by_label": dict(sorted(stats.items())),
                    "skipped_missing_endpoint": dict(sorted(skipped.items())),
                    "entity_base": entity_summary,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    if args.online:
        core.run_online(args)
    if args.summary:
        core.summarize_judgments(args)
    if args.apply:
        core.apply_augments(args)


if __name__ == "__main__":
    main()
