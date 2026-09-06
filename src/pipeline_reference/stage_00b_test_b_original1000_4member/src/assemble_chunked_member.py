import argparse
import json
import re
import time
from pathlib import Path


CHUNK_RE = re.compile(r"_(\d{4})_(\d{4})\.json$")


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def chunk_span(path):
    match = CHUNK_RE.search(path.name)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def collect_chunks(chunks_dir, suffix):
    chunks = []
    for path in sorted(Path(chunks_dir).glob(f"{suffix}_*.json")):
        span = chunk_span(path)
        if span is None:
            continue
        start, stop = span
        try:
            records = load_json(path)
        except Exception as exc:
            chunks.append({"path": str(path), "start": start, "stop": stop, "valid": False, "error": repr(exc)})
            continue
        chunks.append({
            "path": str(path),
            "start": start,
            "stop": stop,
            "valid": len(records) == stop - start,
            "records": records,
            "chunk_len": stop - start,
            "mtime": path.stat().st_mtime,
        })
    return chunks


def assemble(chunks_dir, suffix, expected):
    slots = [None] * expected
    source = [None] * expected
    invalid = []
    chunks = sorted(
        collect_chunks(chunks_dir, suffix),
        key=lambda item: (item["chunk_len"] if item.get("valid") else 10**9, -item.get("mtime", 0)),
    )
    for chunk in chunks:
        if not chunk.get("valid"):
            invalid.append({k: chunk[k] for k in ("path", "start", "stop", "error") if k in chunk})
            continue
        start = chunk["start"]
        for offset, record in enumerate(chunk["records"]):
            index = start + offset
            if 0 <= index < expected and slots[index] is None:
                slots[index] = record
                source[index] = chunk["path"]
    missing = [i for i, record in enumerate(slots) if record is None]
    empty = [i for i, record in enumerate(slots) if record is not None and not record.get("entities") and not record.get("relations")]
    return slots, {
        "expected": expected,
        "covered": expected - len(missing),
        "missing": missing,
        "empty": empty,
        "empty_count": len(empty),
        "invalid_chunks": invalid,
        "sources": source,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solution_root", required=True)
    parser.add_argument("--suffix", required=True)
    parser.add_argument("--tag", default="testB_original1000")
    parser.add_argument("--expected", type=int, default=600)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--poll_seconds", type=int, default=120)
    parser.add_argument("--timeout_hours", type=float, default=8)
    args = parser.parse_args()

    root = Path(args.solution_root)
    chunks_dir = root / "outputs" / f"{args.tag}_{args.suffix}_chunks"
    full_out = root / "outputs" / f"{args.tag}_{args.suffix}.json"
    summary_out = root / "outputs" / f"{args.tag}_{args.suffix}_assembly_summary.json"
    deadline = time.time() + args.timeout_hours * 3600
    while True:
        records, summary = assemble(chunks_dir, args.suffix, args.expected)
        print(f"[STATUS] covered={summary['covered']}/{args.expected} empty={summary['empty_count']} invalid={len(summary['invalid_chunks'])}", flush=True)
        if not summary["missing"]:
            dump_json(full_out, records)
            dump_json(summary_out, {k: v for k, v in summary.items() if k != "sources"})
            print(f"[OK] wrote {full_out}", flush=True)
            print(f"[OK] wrote {summary_out}", flush=True)
            return
        dump_json(summary_out, {k: v for k, v in summary.items() if k != "sources"})
        if not args.wait:
            raise SystemExit(f"missing {len(summary['missing'])} records; see {summary_out}")
        if time.time() >= deadline:
            raise SystemExit(f"timed out waiting for chunks; see {summary_out}")
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
