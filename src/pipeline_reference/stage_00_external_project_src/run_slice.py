"""Run inference on a contiguous slice of the dataset.

Useful for iterative workflows (e.g. single-record retries and manual correction):
`python run_slice.py --input data/dev.json --output outputs/dev_part.json --start 120 --limit 1 ...`
"""
import argparse
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import client  # noqa: E402
from run import process_record, resolve_model  # noqa: E402
from common import load_json, dump_json  # noqa: E402
from retriever import ExemplarRetriever, SemanticExemplarRetriever  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
POOL_PATH = ROOT / "solution" / "data" / "pool.json"
EMBED_CACHE_DIR = ROOT / "solution" / "cache" / "srag_embeddings"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default="main", help="main|cheap|secondary|<raw model>")
    ap.add_argument("--kshot", type=int, default=4)
    ap.add_argument("--retriever", choices=["lexical", "semantic"], default="lexical")
    ap.add_argument("--retriever_model", default=None)
    ap.add_argument("--retriever_cache", default=None)
    ap.add_argument("--start", type=int, default=0, help="start index in the input file")
    ap.add_argument("--limit", type=int, default=None, help="how many records to process")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--max_tokens", type=int, default=4096)
    ap.add_argument("--no_cache", action="store_true")
    ap.add_argument("--conventions", action="store_true")
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--end", type=int, default=None, help="optional end index (exclusive)")
    return ap.parse_args()


def _range(records, start, limit, end):
    n = len(records)
    if start < 0:
        start = 0
    if start > n:
        return []
    if end is None:
        end = n if limit is None else min(n, start + limit)
    else:
        end = max(start, min(end, n))
    return list(range(start, end))


def main():
    args = parse_args()
    model = resolve_model(args.model)
    if client.check_config():
        raise SystemExit("config error in client .env")

    records = load_json(args.input)
    idxs = _range(records, args.start, args.limit, args.end)
    if not idxs:
        raise SystemExit("empty slice")

    pool_exists = POOL_PATH.exists()
    retriever = None
    if args.kshot > 0 and pool_exists:
        pool = load_json(POOL_PATH)
        if args.retriever == "semantic":
            if not args.retriever_model:
                raise SystemExit("--retriever semantic requires --retriever_model")
            cache_path = args.retriever_cache
            if cache_path is None:
                safe_model = re.sub(r'[<>:"/\\|?*]', "_", args.retriever_model)
                cache_path = EMBED_CACHE_DIR / f"{safe_model}.json"
            retriever = SemanticExemplarRetriever(pool, args.retriever_model, cache_path=str(cache_path))
        else:
            retriever = ExemplarRetriever(pool)
    use_cache = not args.no_cache

    selected = [records[i] for i in idxs]
    print(f"[INFO] model={model} start={idxs[0]} end={idxs[-1] + 1} n={len(selected)}")

    def _run_one(rec):
        return process_record(rec, model, retriever, args.kshot, use_cache, args.max_tokens,
                             args.conventions, args.temperature)

    out = [None] * len(selected)
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {ex.submit(_run_one, rec): j for j, rec in enumerate(selected)}
        done = 0
        for fut in as_completed(futs):
            j = futs[fut]
            try:
                out[j] = fut.result()
            except Exception as e:
                print(f"[WARN] local idx {idxs[j]} failed: {e}")
                out[j] = {"text": selected[j]["text"], "entities": [], "relations": []}
            done += 1
            if done % 20 == 0 or done == len(selected):
                print(f"  progress {done}/{len(selected)}")

    # keep local schema; each line corresponds to idxs order
    dump_json(args.output, out)
    print(f"[OK] wrote {args.output}")


if __name__ == "__main__":
    main()
