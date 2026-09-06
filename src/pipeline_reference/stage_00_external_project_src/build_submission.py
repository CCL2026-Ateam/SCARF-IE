"""End-to-end submission builder for MGBIE Track-A.

Runs the 4-member ensemble {sonnet, opus, sonnet-conv, opus-conv} on an input file,
combines with voting (entities >=3/4, relations >=3/4), and writes submit.json
(+ optionally zips to submit.zip).

Usage:
  python build_submission.py --input ../../dataset/test_A.json --tag testA
  # then: cd outputs && zip submit.zip submit.json   (or use --zip if 'zip' available)
"""
import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_json, dump_json, evaluate, print_metrics  # noqa
from ensemble import ensemble  # noqa

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "solution" / "outputs"
PY = sys.executable

MEMBERS = [
    ("secondary", False, "sonnet_k4"),
    ("main", False, "opus_k4"),
    ("secondary", True, "sonnet_k4_conv"),
    ("main", True, "opus_k4_conv"),
]


def run_member(input_file, model, conventions, tag, kshot, concurrency, retriever,
               retriever_model, retriever_cache):
    out = OUT / f"{tag}.json"
    cmd = [PY, str(Path(__file__).parent / "run.py"),
           "--input", str(input_file), "--output", str(out),
           "--model", model, "--kshot", str(kshot), "--retriever", retriever,
           "--concurrency", str(concurrency)]
    if retriever_model:
        cmd += ["--retriever_model", str(retriever_model)]
    if retriever_cache:
        cmd += ["--retriever_cache", str(retriever_cache)]
    if conventions:
        cmd.append("--conventions")
    print(f"[RUN] {tag}: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    return out


def to_submission(records):
    clean = []
    for r in records:
        clean.append({
            "text": r["text"],
            "entities": [{"start": e["start"], "end": e["end"], "text": e["text"], "label": e["label"]}
                         for e in r["entities"]],
            "relations": [{
                "head": x["head"], "head_start": x["head_start"], "head_end": x["head_end"], "head_type": x["head_type"],
                "tail": x["tail"], "tail_start": x["tail_start"], "tail_end": x["tail_end"], "tail_type": x["tail_type"],
                "label": x["label"],
            } for x in r["relations"]],
        })
    return clean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--tag", default="testA")
    ap.add_argument("--kshot", type=int, default=4)
    ap.add_argument("--retriever", choices=["lexical", "semantic"], default="lexical")
    ap.add_argument("--retriever_model", default=None)
    ap.add_argument("--retriever_cache", default=None)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--vote_e", type=int, default=3)
    ap.add_argument("--vote_r", type=int, default=3)
    ap.add_argument("--rel_label_threshold", action="append", default=[],
                    help="repeatable LABEL:K e.g. CON:3, USE:4 (per-label relation vote threshold)")
    ap.add_argument("--skip_run", action="store_true", help="reuse existing member outputs")
    ap.add_argument("--gold", default=None, help="evaluate ensemble if gold provided")
    ap.add_argument("--zip", action="store_true", help="also write submit.zip")
    args = ap.parse_args()

    member_files = []
    for model, conv, suffix in MEMBERS:
        f = OUT / f"{args.tag}_{suffix}.json"
        if not args.skip_run or not f.exists():
            run_member(args.input, model, conv, f"{args.tag}_{suffix}", args.kshot,
                       args.concurrency, args.retriever, args.retriever_model,
                       args.retriever_cache)
        member_files.append(f)

    preds = [load_json(f) for f in member_files]
    rel_label_threshold = {}
    for item in args.rel_label_threshold:
        if ":" not in item:
            continue
        label, k = item.split(":", 1)
        try:
            rel_label_threshold[label.strip()] = int(k.strip())
        except ValueError:
            continue

    combined = ensemble(preds, args.vote_e, args.vote_r, rel_label_threshold)
    ens_path = OUT / f"{args.tag}_ensemble.json"
    dump_json(ens_path, combined)
    print(f"[OK] ensemble -> {ens_path}")

    if args.gold:
        m = evaluate(load_json(args.gold), combined)
        print_metrics(m, title=f"{args.tag} ENSEMBLE vote e>={args.vote_e} r>={args.vote_r}")

    submission = to_submission(combined)
    sub_json = OUT / "submit.json"
    dump_json(sub_json, submission)
    print(f"[OK] submission json -> {sub_json}")

    if args.zip:
        zip_path = OUT / "submit.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(sub_json, arcname="submit.json")
        print(f"[OK] submit.zip -> {zip_path}")


if __name__ == "__main__":
    main()
