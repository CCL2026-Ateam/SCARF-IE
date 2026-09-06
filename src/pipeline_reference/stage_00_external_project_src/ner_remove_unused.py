"""Remove selected entity labels only when they are unused by relations."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import dump_json, evaluate, load_json, print_metrics  # noqa: E402


def relation_entity_keys(rec):
    keys = set()
    for rel in rec.get("relations", []):
        keys.add((rel["head_start"], rel["head_end"], rel["head_type"]))
        keys.add((rel["tail_start"], rel["tail_end"], rel["tail_type"]))
    return keys


def remove_unused(records, labels):
    labels = set(labels)
    out = []
    for rec in records:
        used = relation_entity_keys(rec)
        ents = []
        for ent in rec.get("entities", []):
            key = (ent["start"], ent["end"], ent["label"])
            if ent["label"] in labels and key not in used:
                continue
            ents.append(ent)
        out.append({
            "text": rec["text"],
            "entities": ents,
            "relations": list(rec.get("relations", [])),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--label", action="append", required=True)
    ap.add_argument("--gold", default=None)
    args = ap.parse_args()

    out = remove_unused(load_json(args.input), args.label)
    dump_json(args.output, out)
    print(f"[OK] removed unused labels {args.label} from {len(out)} records -> {args.output}")
    if args.gold:
        metrics = evaluate(load_json(args.gold), out)
        print_metrics(metrics, title="REMOVE UNUSED NER")


if __name__ == "__main__":
    main()
