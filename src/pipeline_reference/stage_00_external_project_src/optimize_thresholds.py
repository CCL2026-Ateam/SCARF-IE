"""Greedy threshold search on a labeled dev set.

This searches per-label entity and relation vote thresholds using existing
member prediction files. It is meant for fast iteration before applying the
same policy to test_A.
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ENTITY_LABELS, RELATION_LABELS, evaluate, load_json, print_metrics  # noqa: E402
from ensemble import ensemble  # noqa: E402


def parse_thresholds(items):
    thresholds = {}
    for item in items:
        if ":" not in item:
            raise SystemExit(f"bad threshold {item!r}, expected LABEL:K")
        label, value = item.split(":", 1)
        thresholds[label.strip()] = int(value.strip())
    return thresholds


def score(gold, preds, vote_e, vote_r, ent_thresholds, rel_thresholds):
    combined = ensemble(preds, vote_e, vote_r, rel_thresholds, ent_thresholds)
    return evaluate(gold, combined), combined


def greedy_search(gold, preds, vote_e, vote_r, ent_base, rel_base, candidates, rounds):
    ent = dict(ent_base)
    rel = dict(rel_base)
    best_metrics, _ = score(gold, preds, vote_e, vote_r, ent, rel)
    best_total = best_metrics["total_score"]
    history = [{
        "step": "baseline",
        "total_score": best_total,
        "score_ner": best_metrics["score_ner"],
        "score_re": best_metrics["score_re"],
        "entity_thresholds": dict(ent),
        "relation_thresholds": dict(rel),
    }]

    labels = [("entity", label) for label in ENTITY_LABELS] + [("relation", label) for label in RELATION_LABELS]
    for round_no in range(1, rounds + 1):
        improved = False
        for kind, label in labels:
            current = ent.get(label, vote_e) if kind == "entity" else rel.get(label, vote_r)
            local_best = (best_total, current, best_metrics)
            for threshold in candidates:
                if threshold == current:
                    continue
                trial_ent = dict(ent)
                trial_rel = dict(rel)
                if kind == "entity":
                    trial_ent[label] = threshold
                else:
                    trial_rel[label] = threshold
                metrics, _ = score(gold, preds, vote_e, vote_r, trial_ent, trial_rel)
                total = metrics["total_score"]
                if total > local_best[0] + 1e-12:
                    local_best = (total, threshold, metrics)
            if local_best[1] != current:
                if kind == "entity":
                    ent[label] = local_best[1]
                else:
                    rel[label] = local_best[1]
                best_total, best_metrics = local_best[0], local_best[2]
                improved = True
                history.append({
                    "step": f"round{round_no}:{kind}:{label}={local_best[1]}",
                    "total_score": best_total,
                    "score_ner": best_metrics["score_ner"],
                    "score_re": best_metrics["score_re"],
                    "entity_thresholds": dict(ent),
                    "relation_thresholds": dict(rel),
                })
        if not improved:
            break
    return ent, rel, best_metrics, history


def grid_relation_only(gold, preds, vote_e, vote_r, ent_thresholds, candidates):
    best = None
    labels = RELATION_LABELS
    for values in itertools.product(candidates, repeat=len(labels)):
        rel = dict(zip(labels, values))
        metrics, _ = score(gold, preds, vote_e, vote_r, ent_thresholds, rel)
        total = metrics["total_score"]
        if best is None or total > best[0]:
            best = (total, rel, metrics)
    return best[1], best[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--vote_e", type=int, default=3)
    ap.add_argument("--vote_r", type=int, default=2)
    ap.add_argument("--ent_label_threshold", action="append", default=[])
    ap.add_argument("--rel_label_threshold", action="append", default=[])
    ap.add_argument("--candidates", nargs="+", type=int, default=[2, 3, 4, 5])
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--output", required=True)
    ap.add_argument("--grid_relations", action="store_true",
                    help="try all relation threshold combinations after greedy entity search")
    args = ap.parse_args()

    preds = [load_json(path) for path in args.inputs]
    lengths = {len(records) for records in preds}
    if len(lengths) != 1:
        raise SystemExit(f"member output lengths differ: {sorted(lengths)}")
    gold = load_json(args.gold)

    ent_base = parse_thresholds(args.ent_label_threshold)
    rel_base = parse_thresholds(args.rel_label_threshold)
    ent, rel, metrics, history = greedy_search(
        gold, preds, args.vote_e, args.vote_r, ent_base, rel_base,
        args.candidates, args.rounds
    )
    if args.grid_relations:
        rel, metrics = grid_relation_only(gold, preds, args.vote_e, args.vote_r, ent, args.candidates)

    result = {
        "vote_e": args.vote_e,
        "vote_r": args.vote_r,
        "entity_thresholds": ent,
        "relation_thresholds": rel,
        "metrics": metrics,
        "history": history,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print_metrics(metrics, title="BEST THRESHOLDS")
    print("entity_thresholds =", " ".join(f"{k}:{v}" for k, v in sorted(ent.items())))
    print("relation_thresholds =", " ".join(f"{k}:{v}" for k, v in sorted(rel.items())))
    print(f"[OK] wrote {args.output}")


if __name__ == "__main__":
    main()
