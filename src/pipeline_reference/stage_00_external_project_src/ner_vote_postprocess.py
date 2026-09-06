"""Add high-confidence NER entities from member outputs after ensemble.

This is intentionally conservative: it only adds entities from member vote
agreement and does not add new relations. Existing relations are preserved.
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import dump_json, evaluate, load_json, print_metrics  # noqa: E402
from ensemble import ekey  # noqa: E402


def parse_thresholds(items):
    thresholds = {}
    for item in items:
        if ":" not in item:
            raise SystemExit(f"bad threshold {item!r}, expected LABEL:K")
        label, value = item.split(":", 1)
        thresholds[label.strip()] = int(value.strip())
    return thresholds


def postprocess(base_records, member_records, thresholds):
    out = []
    for i, rec in enumerate(base_records):
        ents = list(rec.get("entities", []))
        seen = {ekey(ent) for ent in ents}
        votes = Counter()
        objects = {}

        for member in member_records:
            member_seen = set()
            for ent in member[i].get("entities", []):
                key = ekey(ent)
                if key in member_seen:
                    continue
                member_seen.add(key)
                votes[key] += 1
                objects[key] = ent

        for key, count in votes.items():
            label = key[2]
            if label in thresholds and count >= thresholds[label] and key not in seen:
                ents.append(objects[key])
                seen.add(key)

        ents.sort(key=lambda ent: (ent["start"], ent["end"], ent["label"]))
        out.append({
            "text": rec["text"],
            "entities": ents,
            "relations": list(rec.get("relations", [])),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="current ensemble prediction json")
    ap.add_argument("--members", nargs="+", required=True, help="aligned member prediction json files")
    ap.add_argument("--output", required=True)
    ap.add_argument("--threshold", action="append", default=[], help="repeatable LABEL:K, e.g. MRK:4")
    ap.add_argument("--gold", default=None)
    args = ap.parse_args()

    base = load_json(args.base)
    members = [load_json(path) for path in args.members]
    lengths = {len(base)} | {len(member) for member in members}
    if len(lengths) != 1:
        raise SystemExit(f"input lengths differ: {sorted(lengths)}")

    out = postprocess(base, members, parse_thresholds(args.threshold))
    dump_json(args.output, out)
    print(f"[OK] NER postprocess {len(out)} records -> {args.output}")

    if args.gold:
        metrics = evaluate(load_json(args.gold), out)
        print_metrics(metrics, title="NER POSTPROCESS")


if __name__ == "__main__":
    main()
