"""Resumable sharded runner for holdout second-stage ops filtering."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import run_holdout_ops_filter as ops
from env_loader import api_key_env_for_model


def jsonl_rows(path):
    rows = []
    path = Path(path)
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def row_key(row):
    return row.get("record_index"), row.get("batch_index")


def shard_ranges(n_records, shard_size):
    for start in range(0, n_records, shard_size):
        yield start, min(n_records, start + shard_size)


def shard_dir(variant, trial, start, end):
    return ops.trial_out_dir(variant, trial) / "shards" / f"records_{start:03d}_{end:03d}"


def canonical_judgment(variant, trial, pred_path):
    return ops.judgment_path(ops.trial_out_dir(variant, trial), pred_path)


def shard_judgment(variant, trial, pred_path, start, end):
    return ops.judgment_path(shard_dir(variant, trial, start, end), pred_path)


def collect_existing_rows(variant, trial, pred_path):
    rows = {}
    canon = canonical_judgment(variant, trial, pred_path)
    for row in jsonl_rows(canon):
        rows[row_key(row)] = row
    pattern = f"{ops.file_slug(pred_path)}.opus_item_judgments.jsonl"
    for path in sorted((ops.trial_out_dir(variant, trial) / "shards").glob(f"records_*/judgments/{pattern}")):
        for row in jsonl_rows(path):
            rows[row_key(row)] = row
    return rows


def merge_rows(variant, trial, pred_path):
    rows = collect_existing_rows(variant, trial, pred_path)
    ordered = [rows[k] for k in sorted(rows, key=lambda x: (x[0] is None, x[0], x[1] is None, x[1]))]
    out = canonical_judgment(variant, trial, pred_path)
    write_jsonl(out, ordered)
    return out, len(ordered)


def seed_shard(variant, trial, pred_path, start, end):
    rows = collect_existing_rows(variant, trial, pred_path)
    seeded = [
        row
        for (record_index, _), row in rows.items()
        if isinstance(record_index, int) and start <= record_index < end
    ]
    out = shard_judgment(variant, trial, pred_path, start, end)
    write_jsonl(out, seeded)
    return out, len(seeded)


def run_shard(job, args):
    variant, trial, pred_path, start, end = job
    out_dir = shard_dir(variant, trial, start, end)
    jpath, seeded = seed_shard(variant, trial, pred_path, start, end)
    records = ops.load_json(pred_path)
    expected = ops.expected_judgment_rows(records[start:end])
    if ops.count_judgment_rows(jpath) >= expected:
        return {
            "variant": variant,
            "trial": trial,
            "range": [start, end],
            "status": "skipped_complete",
            "rows": ops.count_judgment_rows(jpath),
            "expected": expected,
        }

    cmd = [
        sys.executable,
        str(ops.JUDGE_SCRIPT),
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
        "--start",
        str(start),
        "--end",
        str(end),
    ]
    env = os.environ.copy()
    log = out_dir / "online_judge.stdout.log"
    err = out_dir / "online_judge.stderr.log"
    out_dir.mkdir(parents=True, exist_ok=True)
    last_error = None
    for attempt in range(1, args.shard_attempts + 1):
        with open(log, "a", encoding="utf-8") as lf, open(err, "a", encoding="utf-8") as ef:
            rc = subprocess.run(cmd, cwd=str(ops.PROJECT.parent), env=env, stdout=lf, stderr=ef).returncode
        rows = ops.count_judgment_rows(jpath)
        if rows >= expected:
            return {
                "variant": variant,
                "trial": trial,
                "range": [start, end],
                "status": "complete",
                "rows": rows,
                "expected": expected,
                "attempts": attempt,
                "seeded": seeded,
            }
        last_error = f"rc={rc} rows={rows}/{expected}"
        if attempt < args.shard_attempts:
            time.sleep(args.retry_sleep)
    return {
        "variant": variant,
        "trial": trial,
        "range": [start, end],
        "status": "incomplete",
        "rows": ops.count_judgment_rows(jpath),
        "expected": expected,
        "error": last_error,
        "stdout_log": str(log),
        "stderr_log": str(err),
    }


def build_jobs(args):
    jobs = []
    for variant in args.variants:
        if variant not in ops.VARIANT_PATHS:
            raise SystemExit(f"unknown variant: {variant}")
        for trial in args.trials:
            pred_path = ops.VARIANT_PATHS[variant][trial]
            records = ops.load_json(pred_path)
            for start, end in shard_ranges(len(records), args.shard_size):
                jobs.append((variant, trial, pred_path, start, end))
    return jobs


def run_online(args):
    jobs = build_jobs(args)
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(run_shard, job, args): job for job in jobs}
        for i, fut in enumerate(as_completed(futures), 1):
            result = fut.result()
            results.append(result)
            print(json.dumps({"done": i, "total": len(jobs), **result}, ensure_ascii=False), flush=True)
    return results


def finalize(args, online_results=None):
    apply_rows = []
    merge_status = []
    for variant in args.variants:
        for trial in args.trials:
            pred_path = ops.VARIANT_PATHS[variant][trial]
            jpath, rows = merge_rows(variant, trial, pred_path)
            expected = ops.expected_judgment_rows(ops.load_json(pred_path))
            merge_status.append(
                {
                    "variant": variant,
                    "trial": trial,
                    "judgments": str(jpath),
                    "rows": rows,
                    "expected": expected,
                    "complete": rows >= expected,
                }
            )
            if rows < expected and not args.allow_partial:
                continue
            apply_rows.append(ops.apply_filter(variant, trial, pred_path, args))

    variants = {}
    for variant in args.variants:
        rows = [row for row in apply_rows if row["variant"] == variant]
        if not rows:
            continue
        before = ops.aggregate(rows, "before")
        after = ops.aggregate(rows, "after")
        variants[variant] = {
            "before": before,
            "after": after,
            "delta_after_before": ops.aggregate_delta(after, before),
            "trials": rows,
        }

    summary = {
        "strategy": {
            "entity_policy": getattr(args, "entity_policy", "drop_threshold"),
            "entity_drop_threshold": args.entity_drop_threshold,
            "relation_policy": getattr(args, "relation_policy", "drop_threshold"),
            "relation_drop_threshold": args.relation_drop_threshold,
            "apply_entity_fix": getattr(args, "apply_entity_fix", True),
            "entity_fix_threshold": getattr(args, "entity_fix_threshold", 0.0),
            "apply_relation_fix": getattr(args, "apply_relation_fix", False),
            "relation_fix_threshold": getattr(args, "relation_fix_threshold", 0.95),
            "protect_relation_endpoints": getattr(args, "protect_relation_endpoints", False),
            "noempty_fallback": getattr(args, "noempty_fallback", True),
            "model": args.model,
            "base_url": args.base_url,
            "shard_size": args.shard_size,
            "workers": args.workers,
        },
        "merge_status": merge_status,
        "online_results": online_results or [],
        "variants": variants,
    }
    ops.OPS_OUT.mkdir(parents=True, exist_ok=True)
    ops.dump_json(ops.OPS_OUT / "ops_filter_sharded_summary.json", summary)
    if variants:
        ops.render_report(ops.OPS_OUT / "OPS_FILTER_SHARDED_SUMMARY.md", summary)
    return summary


def copy_outputs_for_user():
    dest = Path("C:/Users/Zzl410410/Documents/Codex/2026-06-28/fnag/outputs/holdout_08_09_ops_filter")
    dest.mkdir(parents=True, exist_ok=True)
    for name in ["ops_filter_sharded_summary.json", "OPS_FILTER_SHARDED_SUMMARY.md"]:
        src = ops.OPS_OUT / name
        if src.exists():
            shutil.copy2(src, dest / name)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", default=["all_entities_all_relations", "all_entities_partial_relations"])
    parser.add_argument("--trials", type=int, nargs="+", default=[8, 9])
    parser.add_argument("--skip_online", action="store_true")
    parser.add_argument("--allow_partial", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--shard_size", type=int, default=5)
    parser.add_argument("--shard_attempts", type=int, default=2)
    parser.add_argument("--retry_sleep", type=int, default=8)
    parser.add_argument("--model", default=os.getenv("GEMINI_MODEL", "gemini-3.1-pro-preview"))
    parser.add_argument("--base_url", default=os.getenv("GEMINI_BASE_URL", os.getenv("BASE_URL", "https://zjapi.com")))
    parser.add_argument("--api_key_env", default=None)
    parser.add_argument("--timeout", type=int, default=int(os.getenv("OPUS_TIMEOUT", "180")))
    parser.add_argument("--retries", type=int, default=int(os.getenv("OPUS_RETRIES", "2")))
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--batch_items", type=int, default=40)
    parser.add_argument("--entity_drop_threshold", type=float, default=0.75)
    parser.add_argument("--relation_drop_threshold", type=float, default=0.95)
    parser.add_argument("--copy_outputs", action="store_true")
    args = parser.parse_args()
    if not args.api_key_env:
        args.api_key_env = api_key_env_for_model(args.model)
    return args


def main():
    args = parse_args()
    online_results = [] if args.skip_online else run_online(args)
    summary = finalize(args, online_results)
    if args.copy_outputs:
        copy_outputs_for_user()
    print(json.dumps({"summary": str(ops.OPS_OUT / "ops_filter_sharded_summary.json"), "variants": list(summary["variants"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
