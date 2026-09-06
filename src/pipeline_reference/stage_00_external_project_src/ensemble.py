"""Ensemble combiner for multiple prediction files.

Combines N prediction JSON files (same records, same order) by voting:
- entities kept if they appear in >= vote_min member predictions
- relations kept if they appear in >= vote_min members AND both endpoints survive

Matching uses the official keys: entity=(start,end,label),
relation=(head_span+type, tail_span+type, label).
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_json, dump_json, evaluate, print_metrics  # noqa: E402


def ekey(e):
    return (e["start"], e["end"], e["label"])


def rkey(r):
    return (r["head_start"], r["head_end"], r["head_type"],
            r["tail_start"], r["tail_end"], r["tail_type"], r["label"])


def ensemble(pred_lists, vote_min, rel_vote_min=None, rel_label_thresholds=None,
             ent_label_thresholds=None):
    """pred_lists: list of N prediction-record-lists (aligned by index)."""
    if rel_vote_min is None:
        rel_vote_min = vote_min
    rel_label_thresholds = rel_label_thresholds or {}
    ent_label_thresholds = ent_label_thresholds or {}
    n_records = len(pred_lists[0])
    out = []
    for i in range(n_records):
        ent_votes = Counter()
        ent_obj = {}
        rel_votes = Counter()
        rel_obj = {}
        text = pred_lists[0][i]["text"]
        for preds in pred_lists:
            rec = preds[i]
            # dedupe within a member first
            seen_e = set()
            for e in rec["entities"]:
                k = ekey(e)
                if k in seen_e:
                    continue
                seen_e.add(k)
                ent_votes[k] += 1
                ent_obj[k] = e
            seen_r = set()
            for r in rec["relations"]:
                k = rkey(r)
                if k in seen_r:
                    continue
                seen_r.add(k)
                rel_votes[k] += 1
                rel_obj[k] = r
        ents = [
            ent_obj[k]
            for k, v in ent_votes.items()
            if v >= ent_label_thresholds.get(k[2], vote_min)
        ]
        eset = set(ekey(e) for e in ents)
        rels = []
        for k, v in rel_votes.items():
            rel_label_threshold = rel_label_thresholds.get(k[6], rel_vote_min)
            if v < rel_label_threshold:
                continue
            r = rel_obj[k]
            if (r["head_start"], r["head_end"], r["head_type"]) in eset and \
               (r["tail_start"], r["tail_end"], r["tail_type"]) in eset:
                rels.append(r)
        ents.sort(key=lambda e: (e["start"], e["end"], e["label"]))
        rels.sort(key=lambda r: (r["head_start"], r["head_end"], r["tail_start"], r["tail_end"], r["label"]))
        out.append({"text": text, "entities": ents, "relations": rels})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="prediction json files")
    ap.add_argument("--output", required=True)
    ap.add_argument("--vote_min", type=int, default=None, help="min votes for entities (default=ceil(N/2)+... = full intersection N)")
    ap.add_argument("--rel_vote_min", type=int, default=None)
    ap.add_argument("--ent_label_threshold", action="append", default=[], help="repeatable LABEL:K e.g. GENE:2")
    ap.add_argument("--rel_label_threshold", action="append", default=[], help="repeatable LABEL:K e.g. CON:3")
    ap.add_argument("--gold", default=None, help="gold file to evaluate against")
    args = ap.parse_args()

    preds = [load_json(p) for p in args.inputs]
    n = len(preds)
    lengths = set(len(p) for p in preds)
    assert len(lengths) == 1, f"inputs have different lengths: {lengths}"
    vote_min = args.vote_min if args.vote_min is not None else n  # default: full intersection
    rel_vote_min = args.rel_vote_min if args.rel_vote_min is not None else vote_min
    rel_label_thresholds = {}
    ent_label_thresholds = {}
    for item in args.ent_label_threshold:
        if ":" not in item:
            continue
        label, k = item.split(":", 1)
        try:
            ent_label_thresholds[label.strip()] = int(k.strip())
        except ValueError:
            continue
    for item in args.rel_label_threshold:
        if ":" not in item:
            continue
        label, k = item.split(":", 1)
        try:
            rel_label_thresholds[label.strip()] = int(k.strip())
        except ValueError:
            continue

    combined = ensemble(preds, vote_min, rel_vote_min, rel_label_thresholds, ent_label_thresholds)
    dump_json(args.output, combined)
    print(f"[OK] ensemble of {n} files, vote_min={vote_min} rel_vote_min={rel_vote_min} -> {args.output}")

    if args.gold:
        gold = load_json(args.gold)
        m = evaluate(gold, combined)
        print_metrics(m, title=f"ENSEMBLE N={n} vote>={vote_min}/rel>={rel_vote_min}")


if __name__ == "__main__":
    main()
