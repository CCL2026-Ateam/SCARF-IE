"""Run online multi-member inference and build the current best submit.zip.

This orchestrates several run.py jobs in parallel, then assembles the outputs
with the best online-calibrated voting policy observed so far:

  Entity thresholds:
    ABS:2 BIS:2 CHR:2 CROP:2 CROSS:4 GST:2 MRK:4 QTL:2, default=3

  Relation thresholds:
    AFF:3 CON:1 HAS:3 LOI:2 OCI:2 USE:5, default=2

Examples:
  # smoke test on dev, first 10 records
  py -3.12 src/run_online_best_submit.py --input data/dev.json --tag online_gpt54_dev_limit10 \
    --limit 10 --eval --record_concurrency 1 --member_concurrency 2

  # full Test-A submission, four online members in parallel
  py -3.12 src/run_online_best_submit.py --input dataset/test_A.json --tag online_best_testA \
    --record_concurrency 16 --member_concurrency 4
"""
import argparse
import subprocess
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_submission import to_submission  # noqa: E402
from common import dump_json, evaluate, load_json, print_metrics  # noqa: E402
from ensemble import ensemble  # noqa: E402
from error_guided_postprocess import process_record  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
PY = sys.executable


BEST_ENTITY_THRESHOLDS = {
    "ABS": 2,
    "BIS": 2,
    "CHR": 2,
    "CROP": 2,
    "CROSS": 4,
    "GST": 2,
    "MRK": 4,
    "QTL": 2,
}

BEST_RELATION_THRESHOLDS = {
    "AFF": 3,
    "CON": 1,
    "HAS": 3,
    "LOI": 2,
    "OCI": 2,
    "USE": 5,
}


DEFAULT_MEMBERS = [
    # model alias/raw-name, conventions, output suffix
    ("secondary", False, "secondary_k4"),
    ("main", False, "main_k4"),
    ("secondary", True, "secondary_k4_conv"),
    ("main", True, "main_k4_conv"),
]


def parse_member(item):
    parts = item.split(":")
    if len(parts) != 3:
        raise SystemExit(
            f"bad --member {item!r}; expected MODEL:CONVENTIONS:SUFFIX, "
            "for example main:false:main_k4"
        )
    model, conventions, suffix = parts
    conv = conventions.strip().lower()
    if conv not in {"0", "1", "false", "true", "no", "yes"}:
        raise SystemExit(f"bad conventions value in --member {item!r}")
    return model.strip(), conv in {"1", "true", "yes"}, suffix.strip()


def run_member(input_path, output_path, model, conventions, kshot, limit, record_concurrency,
               max_tokens, temperature, no_cache, eval_member, retriever, retriever_model,
               retriever_cache, member_timeout, pool, cache_dir):
    cmd = [
        PY, str(Path(__file__).parent / "run.py"),
        "--input", str(input_path),
        "--output", str(output_path),
        "--model", model,
        "--kshot", str(kshot),
        "--retriever", retriever,
        "--concurrency", str(record_concurrency),
        "--max_tokens", str(max_tokens),
    ]
    if retriever_model:
        cmd += ["--retriever_model", str(retriever_model)]
    if retriever_cache:
        cmd += ["--retriever_cache", str(retriever_cache)]
    if pool:
        cmd += ["--pool", str(pool)]
    if cache_dir:
        cmd += ["--cache_dir", str(cache_dir)]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    if conventions:
        cmd.append("--conventions")
    if temperature is not None:
        cmd += ["--temperature", str(temperature)]
    if no_cache:
        cmd.append("--no_cache")
    if eval_member:
        cmd.append("--eval")

    print("[RUN]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, timeout=member_timeout)
    return output_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="dataset/test_A.json")
    ap.add_argument("--tag", default="online_best_testA")
    ap.add_argument("--kshot", type=int, default=4)
    ap.add_argument("--retriever", choices=["lexical", "semantic"], default="lexical")
    ap.add_argument("--retriever_model", default=None)
    ap.add_argument("--retriever_cache", default=None)
    ap.add_argument("--pool", default=None)
    ap.add_argument("--cache_dir", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--record_concurrency", type=int, default=16,
                    help="parallel records inside each run.py member job")
    ap.add_argument("--member_concurrency", type=int, default=4,
                    help="parallel member jobs")
    ap.add_argument("--member_timeout", type=int, default=None,
                    help="seconds before one member subprocess is considered failed")
    ap.add_argument("--max_tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--no_cache", action="store_true")
    ap.add_argument("--skip_run", action="store_true",
                    help="reuse existing member outputs when present")
    ap.add_argument("--error_postprocess", action="store_true",
                    help="apply dev error-guided postprocessing after ensemble voting")
    ap.add_argument("--eval", action="store_true",
                    help="evaluate final ensemble against --input, for dev files with gold labels")
    ap.add_argument("--eval_members", action="store_true",
                    help="also evaluate each individual member output")
    ap.add_argument("--member", action="append", default=[],
                    help="MODEL:CONVENTIONS:SUFFIX; repeat to override default members")
    ap.add_argument("--member_out_dir", default=None,
                    help="directory for per-member prediction files; default is pipeline outputs dir")
    ap.add_argument("--ensemble_out", default=None)
    ap.add_argument("--submit_out", default=None)
    ap.add_argument("--zip_out", default=None)
    args = ap.parse_args()

    input_path = Path(args.input)
    members = [parse_member(item) for item in args.member] if args.member else DEFAULT_MEMBERS
    member_out_dir = Path(args.member_out_dir) if args.member_out_dir else OUT
    member_out_dir.mkdir(parents=True, exist_ok=True)

    OUT.mkdir(parents=True, exist_ok=True)
    member_outputs = []
    jobs = []
    for model, conventions, suffix in members:
        out_path = member_out_dir / f"{args.tag}_{suffix}.json"
        member_outputs.append(out_path)
        if args.skip_run and out_path.exists():
            print(f"[SKIP] existing member output: {out_path}")
            continue
        jobs.append((model, conventions, suffix, out_path))

    if jobs:
        with ThreadPoolExecutor(max_workers=args.member_concurrency) as ex:
            futs = {
                ex.submit(
                    run_member,
                    input_path,
                    out_path,
                    model,
                    conventions,
                    args.kshot,
                    args.limit,
                    args.record_concurrency,
                    args.max_tokens,
                    args.temperature,
                    args.no_cache,
                    args.eval_members,
                    args.retriever,
                    args.retriever_model,
                    args.retriever_cache,
                    args.member_timeout,
                    args.pool,
                    args.cache_dir,
                ): suffix
                for model, conventions, suffix, out_path in jobs
            }
            for fut in as_completed(futs):
                suffix = futs[fut]
                try:
                    fut.result()
                    print(f"[OK] member finished: {suffix}", flush=True)
                except Exception as exc:
                    raise SystemExit(f"[FAIL] member {suffix} failed: {exc}") from exc

    preds = [load_json(path) for path in member_outputs]
    lengths = {len(records) for records in preds}
    if len(lengths) != 1:
        raise SystemExit(f"member output lengths differ: {sorted(lengths)}")

    combined = ensemble(
        preds,
        vote_min=3,
        rel_vote_min=2,
        rel_label_thresholds=BEST_RELATION_THRESHOLDS,
        ent_label_thresholds=BEST_ENTITY_THRESHOLDS,
    )
    if args.error_postprocess:
        combined = [
            process_record(
                record,
                split_overexpressing=True,
                expand_qtl_chromosome=True,
                snp_list_chr=True,
                add_gwas_bm=True,
                drop_overexpressing_has=True,
            )
            for record in combined
        ]

    ensemble_out = Path(args.ensemble_out) if args.ensemble_out else OUT / f"{args.tag}_ensemble.json"
    submit_out = Path(args.submit_out) if args.submit_out else OUT / "submit.json"
    zip_out = Path(args.zip_out) if args.zip_out else OUT / "submit.zip"

    dump_json(ensemble_out, combined)
    print(f"[OK] ensemble -> {ensemble_out}")

    if args.eval:
        gold_records = load_json(input_path)
        if args.limit is not None:
            gold_records = gold_records[:args.limit]
        metrics = evaluate(gold_records, combined)
        print_metrics(metrics, title=f"{args.tag} BEST ONLINE-CALIBRATED ENSEMBLE")
        dump_json(ensemble_out.with_name(ensemble_out.stem + "_metrics.json"), metrics)

    submission = to_submission(combined)
    dump_json(submit_out, submission)
    print(f"[OK] submission json -> {submit_out}")

    with zipfile.ZipFile(zip_out, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(submit_out, arcname="submit.json")
    print(f"[OK] submit zip -> {zip_out}")
    print(f"[OK] records={len(submission)} members={len(member_outputs)}")


if __name__ == "__main__":
    main()
