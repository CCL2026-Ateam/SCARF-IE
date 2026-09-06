import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def empty_count(records):
    return sum(1 for rec in records if not rec.get("entities") and not rec.get("relations"))


def valid_chunk(path, expected_len, max_empty=None):
    path = Path(path)
    if not path.exists():
        return False
    try:
        records = load_json(path)
    except Exception:
        return False
    if len(records) != expected_len:
        return False
    if max_empty is not None and empty_count(records) > max_empty:
        return False
    return True


def run(cmd, cwd, timeout=None):
    print("[RUN]", " ".join(str(x) for x in cmd), flush=True)
    try:
        return subprocess.run(cmd, cwd=cwd, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        class TimeoutResult:
            returncode = -9
        print(f"[TIMEOUT] command exceeded {timeout}s", flush=True)
        return TimeoutResult()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solution_root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--suffix", required=True)
    parser.add_argument("--tag", default="testB_original1000")
    parser.add_argument("--chunk_size", type=int, default=4)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--conventions", action="store_true")
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--max_empty_per_chunk", type=int, default=None)
    parser.add_argument("--retries_per_chunk", type=int, default=2)
    parser.add_argument("--retry_sleep", type=int, default=90)
    parser.add_argument("--chunk_timeout", type=int, default=420)
    parser.add_argument("--write_empty_on_failure", action="store_true")
    args = parser.parse_args()

    root = Path(args.solution_root)
    testb = root / "dataset" / "test_B.json"
    chunks_dir = root / "outputs" / f"{args.tag}_{args.suffix}_chunks"
    full_out = root / "outputs" / f"{args.tag}_{args.suffix}.json"
    test_records = load_json(testb)
    end = len(test_records) if args.end is None else min(args.end, len(test_records))
    chunks_dir.mkdir(parents=True, exist_ok=True)

    failures = []
    for start in range(args.start, end, args.chunk_size):
        stop = min(end, start + args.chunk_size)
        chunk_path = chunks_dir / f"{args.suffix}_{start:04d}_{stop:04d}.json"
        expected_len = stop - start
        if valid_chunk(chunk_path, expected_len, args.max_empty_per_chunk):
            print(f"[SKIP] chunk {start}:{stop} exists", flush=True)
            continue
        cmd = [
            sys.executable,
            "src/run_slice.py",
            "--input",
            "dataset/test_B.json",
            "--output",
            str(chunk_path),
            "--model",
            args.model,
            "--kshot",
            "4",
            "--retriever",
            "lexical",
            "--start",
            str(start),
            "--end",
            str(stop),
            "--concurrency",
            str(args.concurrency),
            "--max_tokens",
            str(args.max_tokens),
        ]
        if args.conventions:
            cmd.append("--conventions")
        ok = False
        last_returncode = None
        for attempt in range(1, args.retries_per_chunk + 1):
            if chunk_path.exists():
                chunk_path.unlink()
            print(f"[TRY] chunk {start}:{stop} attempt {attempt}/{args.retries_per_chunk}", flush=True)
            proc = run(cmd, root, timeout=args.chunk_timeout)
            last_returncode = proc.returncode
            if valid_chunk(chunk_path, expected_len, args.max_empty_per_chunk):
                chunk_records = load_json(chunk_path)
                print(f"[OK] chunk {start}:{stop} empty={empty_count(chunk_records)}/{expected_len}", flush=True)
                ok = True
                break
            if chunk_path.exists():
                try:
                    chunk_records = load_json(chunk_path)
                    print(f"[BAD] chunk {start}:{stop} empty={empty_count(chunk_records)}/{len(chunk_records)}", flush=True)
                except Exception:
                    print(f"[BAD] chunk {start}:{stop} unreadable", flush=True)
            if attempt < args.retries_per_chunk:
                print(f"[WAIT] sleeping {args.retry_sleep}s before retry", flush=True)
                time.sleep(args.retry_sleep)
        if not ok:
            if args.write_empty_on_failure:
                empty = [{"text": test_records[i]["text"], "entities": [], "relations": []} for i in range(start, stop)]
                dump_json(chunk_path, empty)
                print(f"[FALLBACK] wrote empty chunk {start}:{stop}", flush=True)
            else:
                failures.append({"start": start, "end": stop, "returncode": last_returncode})
                print(f"[FAIL] chunk {start}:{stop}", flush=True)

    if failures:
        dump_json(chunks_dir / "failures.json", failures)
        raise SystemExit(f"{len(failures)} chunk(s) failed; see {chunks_dir / 'failures.json'}")

    if args.start == 0 and end == len(test_records):
        merged = []
        for start in range(args.start, end, args.chunk_size):
            stop = min(end, start + args.chunk_size)
            chunk_path = chunks_dir / f"{args.suffix}_{start:04d}_{stop:04d}.json"
            if not valid_chunk(chunk_path, stop - start, args.max_empty_per_chunk):
                raise SystemExit(f"cannot merge invalid chunk: {chunk_path}")
            merged.extend(load_json(chunk_path))
        dump_json(full_out, merged)
        print(f"[OK] merged {len(merged)} records -> {full_out}", flush=True)


if __name__ == "__main__":
    main()
