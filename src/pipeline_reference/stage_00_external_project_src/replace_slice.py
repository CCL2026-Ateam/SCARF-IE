"""Replace a contiguous slice inside a full prediction/submission file.

Use after run_slice.py when iterating record-by-record:
  python src/replace_slice.py \
    --base outputs/testA_sonnet_k4.json \
    --slice outputs/testA_sonnet_k4_i120.json \
    --start 120 \
    --output outputs/testA_sonnet_k4_patched.json
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import dump_json, load_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="full json file to patch")
    ap.add_argument("--slice", required=True, help="json file containing replacement records")
    ap.add_argument("--start", type=int, required=True, help="0-based start index in base")
    ap.add_argument("--output", required=True, help="patched output path")
    args = ap.parse_args()

    base = load_json(args.base)
    patch = load_json(args.slice)
    if args.start < 0:
        raise SystemExit("--start must be >= 0")
    end = args.start + len(patch)
    if end > len(base):
        raise SystemExit(f"slice [{args.start}, {end}) exceeds base length {len(base)}")

    base[args.start:end] = patch
    dump_json(args.output, base)
    print(f"[OK] replaced records [{args.start}, {end}) -> {args.output}")


if __name__ == "__main__":
    main()
