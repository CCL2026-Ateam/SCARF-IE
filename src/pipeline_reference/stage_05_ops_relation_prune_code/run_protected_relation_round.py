from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path


ROOT = Path("C:/Users/Zzl410410/Documents/Codex/2026-06-28/c-users-zzl410410-documents-codex-2026-3")
RUN_OPS = ROOT / "work/test_b_filter_then_augment/run_ops_audit_sharded.py"
ITER_OPS = ROOT / "work/test_b_filter_then_augment/run_iterative_relation_ops.py"
JUDGE = Path(
    "C:/Users/Zzl410410/Documents/Codex/2026-06-24/"
    "019ef53b-8c1b-7a92-ad59-c7c456388b6f/work/test_b_original1000_4member/src/judge_json_items_opus.py"
)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def ranges(n_records: int, shards: int) -> list[tuple[int, int]]:
    width = (n_records + shards - 1) // shards
    return [(start, min(n_records, start + width)) for start in range(0, n_records, width)]


def chunks_from_missing(missing: list[int], max_width: int) -> list[tuple[int, int]]:
    intervals = []
    if not missing:
        return intervals
    start = prev = missing[0]
    for idx in missing[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        intervals.append((start, prev + 1))
        start = prev = idx
    intervals.append((start, prev + 1))

    chunks = []
    for start, end in intervals:
        cur = start
        while cur < end:
            nxt = min(end, cur + max_width)
            chunks.append((cur, nxt))
            cur = nxt
    return chunks


def run_group(
    pred: Path,
    out_dir: Path,
    intervals: list[tuple[int, int]],
    batch_items: int,
    retries: int,
    timeout: int,
    max_parallel: int,
    tag: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    total = len(intervals)
    for offset in range(0, total, max_parallel):
        group = intervals[offset : offset + max_parallel]
        processes = []
        for local_i, (start, end) in enumerate(group):
            idx = offset + local_i
            stdout = out_dir / f"{tag}_{idx}_{start}_{end}.stdout.log"
            stderr = out_dir / f"{tag}_{idx}_{start}_{end}.stderr.log"
            cmd = [
                sys.executable,
                str(RUN_OPS),
                "run-shard",
                "--pred",
                str(pred),
                "--out_dir",
                str(out_dir),
                "--shards",
                str(max(1, total)),
                "--shard_index",
                str(idx),
                "--start",
                str(start),
                "--end",
                str(end),
                "--batch_items",
                str(batch_items),
                "--timeout",
                str(timeout),
                "--retries",
                str(retries),
                "--max_tokens",
                "4096",
            ]
            with open(stdout, "ab") as out, open(stderr, "ab") as err:
                processes.append(subprocess.Popen(cmd, stdout=out, stderr=err))
        while any(p.poll() is None for p in processes):
            running = sum(1 for p in processes if p.poll() is None)
            print(json.dumps({"event": "wait_group", "tag": tag, "running": running}, ensure_ascii=False), flush=True)
            time.sleep(30)


def collect_rows(work_dir: Path) -> tuple[dict[tuple[int, int], dict], list[int]]:
    rows: dict[tuple[int, int], dict] = {}
    parse_records = set()
    for fp in work_dir.glob("**/judgments/*.jsonl"):
        with open(fp, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                key = (row.get("record_index"), row.get("batch_index"))
                old = rows.get(key)
                if old is None or old.get("parse_error") or not row.get("parse_error"):
                    row = dict(row)
                    rows[key] = row
                if row.get("parse_error") and isinstance(row.get("record_index"), int):
                    parse_records.add(int(row["record_index"]))
    return rows, sorted(parse_records)


def coverage(rows: dict[tuple[int, int], dict], n_records: int) -> tuple[list[int], list[int]]:
    seen = {key[0] for key in rows if isinstance(key[0], int)}
    missing = [idx for idx in range(n_records) if idx not in seen]
    parse_records = sorted({key[0] for key, row in rows.items() if row.get("parse_error") and isinstance(key[0], int)})
    return missing, parse_records


def add_conservative_fallback(
    rows: dict[tuple[int, int], dict],
    pred: Path,
    records: list[dict],
    missing: list[int],
) -> list[dict]:
    judge = load_module(JUDGE, "judge_json_items_opus")
    fallback = []
    for idx in missing:
        cands = judge.build_candidates(records[idx])
        rows[(idx, 0)] = {
            "source_file": str(pred.resolve()),
            "record_index": idx,
            "batch_index": 0,
            "prompt_hash": "conservative_empty_after_retries",
            "model": "conservative_fallback_keep_all",
            "candidate_ids": [cand.get("id") for cand in cands],
            "decisions": [],
            "conservative_fallback": True,
            "fallback_reason": "repeated missing/empty judge response; keep all candidates for this record",
        }
        fallback.append(
            {
                "record_index": idx,
                "entities": len(records[idx].get("entities", []) or []),
                "relations": len(records[idx].get("relations", []) or []),
                "candidate_count": len(cands),
            }
        )
    return fallback


def write_merged(rows: dict[tuple[int, int], dict], merged: Path, pred: Path) -> None:
    merged.parent.mkdir(parents=True, exist_ok=True)
    with open(merged, "w", encoding="utf-8") as f:
        for _, row in sorted(rows.items(), key=lambda kv: (kv[0][0], kv[0][1])):
            row = dict(row)
            row["source_file"] = str(pred.resolve())
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def drop_inventory(rows: dict[tuple[int, int], dict], threshold: float = 0.95) -> dict:
    by_label = Counter()
    confidences = Counter()
    total = 0
    for row in rows.values():
        for decision in row.get("decisions", []) or []:
            action = str(decision.get("action", "")).strip().lower()
            if action != "drop":
                continue
            try:
                confidence = float(decision.get("confidence", 0) or 0)
            except Exception:
                confidence = 0.0
            if confidence < threshold:
                continue
            total += 1
            label = str(decision.get("label") or decision.get("type") or "")
            by_label[label] += 1
            confidences[confidence] += 1
    return {
        "drop_decisions_ge_threshold": total,
        "by_decision_label_field": dict(sorted(by_label.items())),
        "confidence_counts": {str(k): v for k, v in sorted(confidences.items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--work_dir", type=Path, required=True)
    parser.add_argument("--shards", type=int, default=5)
    parser.add_argument("--max_parallel", type=int, default=5)
    parser.add_argument("--retries", type=int, default=12)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--max_gap_attempts", type=int, default=5)
    parser.add_argument("--protect_relation_labels", nargs="*", default=["HAS"])
    args = parser.parse_args()

    records = load_json(args.pred)
    args.work_dir.mkdir(parents=True, exist_ok=True)

    initial_ranges = ranges(len(records), args.shards)
    print(json.dumps({"event": "initial_start", "ranges": initial_ranges}, ensure_ascii=False), flush=True)
    run_group(args.pred, args.work_dir / "initial", initial_ranges, 999, args.retries, args.timeout, args.max_parallel, "initial")

    fallback = []
    rows, _ = collect_rows(args.work_dir)
    missing, parse_records = coverage(rows, len(records))
    print(
        json.dumps(
            {"event": "coverage", "stage": "initial", "rows": len(rows), "missing": len(missing), "parse": parse_records[:20]},
            ensure_ascii=False,
        ),
        flush=True,
    )

    for attempt in range(1, args.max_gap_attempts + 1):
        rows, _ = collect_rows(args.work_dir)
        missing, parse_records = coverage(rows, len(records))
        retry_records = sorted(set(missing) | set(parse_records))
        if not retry_records:
            break
        width = max(1, 24 // attempt)
        intervals = chunks_from_missing(retry_records, width)
        print(
            json.dumps(
                {
                    "event": "gap_attempt_start",
                    "attempt": attempt,
                    "missing": len(missing),
                    "parse": len(parse_records),
                    "width": width,
                    "intervals": intervals[:20],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        run_group(
            args.pred,
            args.work_dir / f"gap_attempt_{attempt:02d}",
            intervals,
            999 if attempt < 4 else 4,
            args.retries + attempt,
            args.timeout,
            args.max_parallel,
            f"gap{attempt}",
        )

    rows, _ = collect_rows(args.work_dir)
    missing, parse_records = coverage(rows, len(records))
    if missing or parse_records:
        retry_records = sorted(set(missing) | set(parse_records))
        fallback = add_conservative_fallback(rows, args.pred, records, retry_records)
    missing, parse_records = coverage(rows, len(records))
    if missing or parse_records:
        raise RuntimeError(f"coverage still incomplete: missing={missing} parse={parse_records}")

    merged = args.work_dir / "merged" / f"{args.prefix}.opus_item_judgments_fixed.jsonl"
    write_merged(rows, merged, args.pred)

    iterops = load_module(ITER_OPS, "run_iterative_relation_ops")
    protected = {str(label).strip() for label in args.protect_relation_labels if str(label).strip()}
    output_json, summary_path = iterops.apply_relation_only(args.pred, merged, args.prefix, protected)
    summary = iterops.load_json(summary_path)
    summary["judgment_merge"] = {
        "merged_rows": len(rows),
        "merged_judgments": str(merged),
        "conservative_fallback_records": fallback,
        "drop_inventory": drop_inventory(rows),
    }
    iterops.dump_json(summary_path, summary)
    print(
        json.dumps(
            {
                "event": "done",
                "output_json": str(output_json),
                "summary": str(summary_path),
                "fallback_records": fallback,
                "applied": summary.get("applied", {}),
                "delta": summary.get("delta", {}),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
