import argparse
import hashlib
import json
from pathlib import Path


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solution_root", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--summary", default=None)
    args = parser.parse_args()

    root = Path(args.solution_root)
    pool_path = root / "data" / "pool.json"
    dev_path = root / "data" / "dev.json"
    out_path = Path(args.out) if args.out else root / "outputs" / "original_1000_pool.json"
    summary_path = Path(args.summary) if args.summary else root / "outputs" / "original_1000_pool_summary.json"

    pool = load_json(pool_path)
    dev = load_json(dev_path)
    combined = pool + dev
    dump_json(out_path, combined)

    summary = {
        "source_pool": str(pool_path),
        "source_pool_records": len(pool),
        "source_pool_sha256": sha256(pool_path),
        "source_dev": str(dev_path),
        "source_dev_records": len(dev),
        "source_dev_sha256": sha256(dev_path),
        "output": str(out_path),
        "output_records": len(combined),
        "output_sha256": sha256(out_path),
    }
    dump_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
