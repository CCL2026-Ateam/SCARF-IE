"""Test-B project runner.

This file documents the reproducible path used for the Test-B package:
1. place/extract test_B.json under dataset/
2. run the high-quality main model twice, with and without boundary conventions
3. assemble a submission from the completed members

The secondary members are intentionally optional because the relay may stop
long runs when quota or connection limits are hit. Use --full-secondary to try
the original four-member online ensemble.
"""
import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def run(cmd):
    print("[RUN]", " ".join(str(x) for x in cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(ROOT / "dataset" / "test_B.json"))
    ap.add_argument("--tag", default="testB_full")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--full-secondary", action="store_true",
                    help="try the original 4-member online ensemble")
    ap.add_argument("--skip-run", action="store_true")
    args = ap.parse_args()

    out = ROOT / "outputs"
    out.mkdir(parents=True, exist_ok=True)

    if args.full_secondary:
        cmd = [
            PY, str(ROOT / "src" / "run_online_best_submit.py"),
            "--input", args.input,
            "--tag", args.tag,
            "--record_concurrency", str(args.concurrency),
            "--member_concurrency", "4",
            "--ensemble_out", str(out / f"{args.tag}_ensemble.json"),
            "--submit_out", str(out / f"{args.tag}_submit.json"),
            "--zip_out", str(out / f"{args.tag}_submit.zip"),
        ]
        if args.skip_run:
            cmd.append("--skip_run")
        run(cmd)
        return

    members = [
        (False, out / f"{args.tag}_main_k4.json"),
        (True, out / f"{args.tag}_main_k4_conv.json"),
    ]
    for conventions, member_out in members:
        if args.skip_run and member_out.exists():
            print(f"[SKIP] {member_out}")
            continue
        cmd = [
            PY, str(ROOT / "src" / "run.py"),
            "--input", args.input,
            "--output", str(member_out),
            "--model", "main",
            "--kshot", "4",
            "--retriever", "lexical",
            "--concurrency", str(args.concurrency),
            "--max_tokens", "4096",
        ]
        if conventions:
            cmd.append("--conventions")
        run(cmd)

    submit = out / f"{args.tag}_submit.json"
    zip_path = out / f"{args.tag}_submit.zip"
    run([
        PY, str(ROOT / "src" / "assemble_submission.py"),
        "--inputs", *(str(path) for _, path in members),
        "--ensemble_out", str(out / f"{args.tag}_ensemble.json"),
        "--submit_out", str(submit),
        "--vote_e", "1",
        "--vote_r", "1",
        "--zip", str(zip_path),
    ])

    # Compatibility alias expected by older package consumers.
    alias = out / "answer_testB.zip"
    with zipfile.ZipFile(alias, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(submit, arcname="submit.json")
    print(f"[OK] alias -> {alias}")


if __name__ == "__main__":
    main()
