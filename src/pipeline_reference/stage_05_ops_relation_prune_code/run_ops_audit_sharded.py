"""Sharded OPS audit for a prediction JSON using the existing Opus item judge."""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import runpy
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent
VALIDATION = PIPELINE_ROOT / "stage_01_validation_scaling_code"
BASE_JUDGE = PIPELINE_ROOT / "stage_00b_test_b_original1000_4member" / "src" / "judge_json_items_opus.py"

sys.path.insert(0, str(VALIDATION))
import env_loader  # noqa: E402


def load_env() -> None:
    env_loader.load_env_defaults(*env_loader.default_env_paths(PIPELINE_ROOT))


def file_slug(path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem)


def load_records(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ranges(n_records: int, shards: int) -> list[tuple[int, int]]:
    width = (n_records + shards - 1) // shards
    return [(start, min(n_records, start + width)) for start in range(0, n_records, width)]


def shard_dir(out_dir: Path, shard_index: int, start: int, end: int) -> Path:
    return out_dir / f"shard_{shard_index:02d}_{start:03d}_{end:03d}"


def run_shard(args) -> None:
    load_env()
    out_dir = shard_dir(args.out_dir, args.shard_index, args.start, args.end)
    out_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = out_dir / "ops_judge.stdout.log"
    stderr_path = out_dir / "ops_judge.stderr.log"
    argv = [
        str(BASE_JUDGE),
        "--inputs",
        str(args.pred),
        "--output_dir",
        str(out_dir / "judgments"),
        "--model",
        args.model,
        "--base_url",
        args.base_url or os.getenv("OPUS_BASE_URL", "https://zjapi.com"),
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
        str(args.start),
        "--end",
        str(args.end),
    ]
    if args.keep_raw:
        argv.append("--keep_raw")
    with open(stdout_path, "a", encoding="utf-8") as out, open(stderr_path, "a", encoding="utf-8") as err:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            print("[RUN] " + " ".join(argv), flush=True)
            old_argv = sys.argv
            try:
                sys.argv = argv
                runpy.run_path(str(BASE_JUDGE), run_name="__main__")
            finally:
                sys.argv = old_argv


def status(args) -> None:
    records = load_records(args.pred)
    slug = file_slug(args.pred)
    total = 0
    rows = {}
    for i, (start, end) in enumerate(ranges(len(records), args.shards)):
        path = shard_dir(args.out_dir, i, start, end) / "judgments" / f"{slug}.opus_item_judgments.jsonl"
        have = 0
        parse_errors = 0
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    rows[(row.get("record_index"), row.get("batch_index"))] = row
                    have += 1
                    parse_errors += 1 if row.get("parse_error") else 0
        total += have
        print(json.dumps({
            "shard": i,
            "range": [start, end],
            "expected_records": end - start,
            "rows": have,
            "parse_errors": parse_errors,
            "path": str(path),
            "exists": path.exists(),
        }, ensure_ascii=False))
    print(json.dumps({
        "records": len(records),
        "unique_rows": len(rows),
        "raw_rows": total,
        "remaining_record_rows": len(records) - len({k[0] for k in rows if isinstance(k[0], int)}),
    }, ensure_ascii=False))


def merge(args) -> None:
    records = load_records(args.pred)
    slug = file_slug(args.pred)
    rows = {}
    parse_errors = 0
    for i, (start, end) in enumerate(ranges(len(records), args.shards)):
        path = shard_dir(args.out_dir, i, start, end) / "judgments" / f"{slug}.opus_item_judgments.jsonl"
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                rows[(row.get("record_index"), row.get("batch_index"))] = row
                parse_errors += 1 if row.get("parse_error") else 0
    args.merged.parent.mkdir(parents=True, exist_ok=True)
    with open(args.merged, "w", encoding="utf-8") as f:
        for _, row in sorted(rows.items(), key=lambda kv: (kv[0][0], kv[0][1])):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({
        "output": str(args.merged),
        "records": len(records),
        "unique_rows": len(rows),
        "parse_errors": parse_errors,
        "complete": len({k[0] for k in rows if isinstance(k[0], int)}) == len(records),
    }, ensure_ascii=False))


def parse_args():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--pred", type=Path, required=True)
    common.add_argument("--out_dir", type=Path, default=PIPELINE_ROOT / "work/test_b_filter_then_augment/all_relations_ops")
    common.add_argument("--shards", type=int, default=8)

    run = sub.add_parser("run-shard", parents=[common])
    run.add_argument("--shard_index", type=int, required=True)
    run.add_argument("--start", type=int, required=True)
    run.add_argument("--end", type=int, required=True)
    run.add_argument("--model", default=os.getenv("OPUS_MODEL", os.getenv("MODEL_MAIN", "gpt-5.4")))
    run.add_argument("--base_url", default=None)
    run.add_argument("--api_key_env", default="OPUS_API_KEY")
    run.add_argument("--timeout", type=int, default=int(os.getenv("OPUS_TIMEOUT", "240")))
    run.add_argument("--retries", type=int, default=int(os.getenv("OPUS_RETRIES", "5")))
    run.add_argument("--max_tokens", type=int, default=4096)
    run.add_argument("--batch_items", type=int, default=999)
    run.add_argument("--keep_raw", action="store_true")

    sub.add_parser("status", parents=[common])
    merge_parser = sub.add_parser("merge", parents=[common])
    merge_parser.add_argument("--merged", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "run-shard":
        run_shard(args)
    elif args.command == "status":
        status(args)
    elif args.command == "merge":
        merge(args)


if __name__ == "__main__":
    main()
