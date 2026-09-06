"""Split train.json into a local train-pool (for few-shot exemplars) and a dev set
(for offline scoring). Deterministic, stratified-ish by record richness."""
import sys
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_json, dump_json  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "dataset" / "train.json"
OUT_DIR = ROOT / "solution" / "data"


def main(dev_size: int = 200, seed: int = 42):
    data = load_json(TRAIN)
    n = len(data)
    idx = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idx)
    dev_idx = sorted(idx[:dev_size])
    pool_idx = sorted(idx[dev_size:])
    dev = [data[i] for i in dev_idx]
    pool = [data[i] for i in pool_idx]
    dump_json(OUT_DIR / "dev.json", dev)
    dump_json(OUT_DIR / "pool.json", pool)
    # also keep the index mapping for reproducibility
    dump_json(OUT_DIR / "split_meta.json", {
        "seed": seed, "dev_size": dev_size,
        "dev_idx": dev_idx, "pool_idx": pool_idx,
    })
    # quick stats
    def stats(recs, name):
        ne = sum(len(r["entities"]) for r in recs)
        nr = sum(len(r["relations"]) for r in recs)
        print(f"  {name}: {len(recs)} records, {ne} entities, {nr} relations")
    print(f"Split train.json ({n} records) -> dev={len(dev)}, pool={len(pool)}")
    stats(dev, "dev ")
    stats(pool, "pool")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev_size", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    main(args.dev_size, args.seed)
