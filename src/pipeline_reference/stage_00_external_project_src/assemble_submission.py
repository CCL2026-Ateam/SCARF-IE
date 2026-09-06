"""Assemble an ensemble output and Track-A submission from arbitrary members.

This is the flexible companion to build_submission.py. It does not run models;
it only combines existing prediction files, evaluates optionally, and writes a
submission-ready JSON/ZIP.

Example:
  python src/assemble_submission.py \
    --inputs outputs/testA_sonnet_k4.json outputs/testA_opus_k4.json \
             outputs/testA_sonnet_k4_conv.json outputs/testA_opus_k4_conv.json \
    --ensemble_out outputs/testA_ensemble.json \
    --submit_out outputs/submit.json \
    --vote_e 3 --vote_r 2 \
    --rel_label_threshold CON:3 --rel_label_threshold USE:4 \
    --rel_label_threshold HAS:4 --rel_label_threshold AFF:4 \
    --rel_label_threshold OCI:2 --rel_label_threshold LOI:2 \
    --zip outputs/submit.zip
"""
import argparse
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_submission import to_submission  # noqa: E402
from common import dump_json, evaluate, load_json, print_metrics  # noqa: E402
from ensemble import ensemble  # noqa: E402


def parse_thresholds(items):
    thresholds = {}
    for item in items:
        if ":" not in item:
            raise SystemExit(f"bad --rel_label_threshold: {item!r}, expected LABEL:K")
        label, value = item.split(":", 1)
        try:
            thresholds[label.strip()] = int(value.strip())
        except ValueError as exc:
            raise SystemExit(f"bad threshold value in {item!r}") from exc
    return thresholds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="aligned member prediction json files")
    ap.add_argument("--ensemble_out", required=True, help="where to write the ensembled prediction json")
    ap.add_argument("--submit_out", required=True, help="where to write the submission json")
    ap.add_argument("--vote_e", type=int, default=3, help="minimum entity votes")
    ap.add_argument("--vote_r", type=int, default=3, help="fallback minimum relation votes")
    ap.add_argument("--ent_label_threshold", action="append", default=[],
                    help="repeatable LABEL:K, e.g. GENE:2")
    ap.add_argument("--rel_label_threshold", action="append", default=[],
                    help="repeatable LABEL:K, e.g. CON:3")
    ap.add_argument("--gold", default=None, help="optional gold json for offline scoring")
    ap.add_argument("--zip", default=None, help="optional submit.zip output path")
    args = ap.parse_args()

    preds = [load_json(path) for path in args.inputs]
    lengths = {len(records) for records in preds}
    if len(lengths) != 1:
        raise SystemExit(f"member output lengths differ: {sorted(lengths)}")

    ent_thresholds = parse_thresholds(args.ent_label_threshold)
    rel_thresholds = parse_thresholds(args.rel_label_threshold)
    combined = ensemble(preds, args.vote_e, args.vote_r, rel_thresholds, ent_thresholds)
    dump_json(args.ensemble_out, combined)
    print(f"[OK] ensemble {len(preds)} members x {len(combined)} records -> {args.ensemble_out}")

    if args.gold:
        metrics = evaluate(load_json(args.gold), combined)
        print_metrics(metrics, title=f"ENSEMBLE vote_e>={args.vote_e} vote_r>={args.vote_r}")

    submission = to_submission(combined)
    dump_json(args.submit_out, submission)
    print(f"[OK] submission -> {args.submit_out}")

    if args.zip:
        with zipfile.ZipFile(args.zip, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(args.submit_out, arcname="submit.json")
        print(f"[OK] zip -> {args.zip}")


if __name__ == "__main__":
    main()
