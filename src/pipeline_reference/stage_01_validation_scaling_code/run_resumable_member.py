import argparse
import json
import shutil
import sys
import time
from pathlib import Path


SOLUTION_ROOT = Path(r"K:\浩然\CCL\solution\solution")
SRC = SOLUTION_ROOT / "src"
sys.path.insert(0, str(SRC))

import run as run_mod  # noqa: E402
from common import dump_json, load_json  # noqa: E402
from retriever import ExemplarRetriever  # noqa: E402


def load_state(path, n):
    if path.exists():
        data = load_json(path)
        if len(data.get("statuses", [])) == n:
            return data
    return {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "updated_at": None,
        "statuses": ["pending"] * n,
        "errors": {},
    }


def save_progress(partial_path, state_path, results, state):
    state["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    dump_json(partial_path, results)
    dump_json(state_path, state)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default="main")
    ap.add_argument("--kshot", type=int, default=4)
    ap.add_argument("--max_tokens", type=int, default=4096)
    ap.add_argument("--pool", default=None)
    ap.add_argument("--conventions", action="store_true")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--stop_after_failures", type=int, default=2)
    ap.add_argument("--sleep_after_failure", type=int, default=0)
    args = ap.parse_args()

    records = load_json(args.input)
    output = Path(args.output)
    partial = output.with_name(output.stem + "_partial.json")
    state_path = output.with_name(output.stem + "_state.json")

    if partial.exists():
        results = load_json(partial)
        if len(results) != len(records):
            results = [None] * len(records)
    else:
        results = [None] * len(records)

    state = load_state(state_path, len(records))
    pool_path = Path(args.pool) if args.pool else SOLUTION_ROOT / "data" / "pool.json"
    pool = load_json(pool_path)
    retriever = ExemplarRetriever(pool) if args.kshot > 0 else None
    model = run_mod.resolve_model(args.model)

    end = len(records) if args.limit is None else min(len(records), args.start + args.limit)
    consecutive_failures = 0
    attempted = 0

    print(
        f"[INFO] resumable model={model} records={len(records)} range={args.start}:{end} "
        f"kshot={args.kshot} max_tokens={args.max_tokens} conventions={args.conventions}",
        flush=True,
    )

    for i in range(args.start, end):
        if state["statuses"][i] == "ok" and results[i] is not None:
            continue
        attempted += 1
        try:
            pred = run_mod.process_record(
                records[i],
                model,
                retriever,
                args.kshot,
                False,
                args.max_tokens,
                conventions=args.conventions,
            )
            results[i] = pred
            state["statuses"][i] = "ok"
            state["errors"].pop(str(i), None)
            consecutive_failures = 0
            print(f"[OK] record {i}", flush=True)
        except Exception as exc:
            state["statuses"][i] = "failed"
            state["errors"][str(i)] = f"{type(exc).__name__}: {exc}"
            consecutive_failures += 1
            print(f"[WARN] record {i} failed: {state['errors'][str(i)]}", flush=True)
            if args.sleep_after_failure:
                time.sleep(args.sleep_after_failure)
        save_progress(partial, state_path, results, state)
        ok_count = sum(1 for s in state["statuses"] if s == "ok")
        if attempted % 5 == 0 or ok_count == len(records):
            print(f"[PROGRESS] ok={ok_count}/{len(records)} attempted={attempted}", flush=True)
        if consecutive_failures >= args.stop_after_failures:
            print(f"[STOP] consecutive_failures={consecutive_failures}", flush=True)
            return 2

    ok_count = sum(1 for s in state["statuses"] if s == "ok")
    if ok_count == len(records):
        shutil.copy2(partial, output)
        print(f"[DONE] wrote complete output {output}", flush=True)
    else:
        print(f"[PARTIAL] ok={ok_count}/{len(records)} partial={partial}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
