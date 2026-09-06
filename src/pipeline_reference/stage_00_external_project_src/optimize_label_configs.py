"""Greedy search for per-label member masks and vote thresholds."""
import argparse
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ENTITY_LABELS, RELATION_LABELS, evaluate, load_json, print_metrics  # noqa: E402
from label_config_ensemble import dump_config, flexible_ensemble  # noqa: E402


def parse_thresholds(items, n_members):
    config = {}
    for item in items:
        parts = item.split(":")
        if len(parts) != 2:
            raise SystemExit(f"bad threshold {item!r}, expected LABEL:K")
        label, threshold = parts
        config[label.strip()] = {
            "members": tuple(range(n_members)),
            "threshold": int(threshold.strip()),
        }
    return config


def candidate_rules(n_members, max_threshold):
    rules = []
    for bits in itertools.product([0, 1], repeat=n_members):
        members = tuple(i for i, bit in enumerate(bits) if bit)
        if not members:
            continue
        for threshold in range(1, min(max_threshold, len(members) + 1) + 1):
            rules.append({"members": members, "threshold": threshold})
        rules.append({"members": members, "threshold": len(members) + 1})
    return rules


def rule_key(rule):
    return (tuple(rule["members"]), int(rule["threshold"]))


def score(gold, preds, vote_e, vote_r, ent_config, rel_config):
    combined = flexible_ensemble(preds, vote_e, vote_r, ent_config, rel_config)
    return evaluate(gold, combined)


def search(gold, preds, vote_e, vote_r, ent_config, rel_config, rounds, max_threshold):
    n_members = len(preds)
    candidates = candidate_rules(n_members, max_threshold)
    best = score(gold, preds, vote_e, vote_r, ent_config, rel_config)
    best_total = best["total_score"]
    history = [{
        "step": "baseline",
        "total_score": best_total,
        "score_ner": best["score_ner"],
        "score_re": best["score_re"],
    }]

    labels = [("entity", label) for label in ENTITY_LABELS] + [("relation", label) for label in RELATION_LABELS]
    for round_no in range(1, rounds + 1):
        improved = False
        for kind, label in labels:
            config = ent_config if kind == "entity" else rel_config
            current = config.get(label, {
                "members": tuple(range(n_members)),
                "threshold": vote_e if kind == "entity" else vote_r,
            })
            local_best = (best_total, current, best)
            seen = {rule_key(current)}
            for rule in candidates:
                key = rule_key(rule)
                if key in seen:
                    continue
                seen.add(key)
                trial_ent = dict(ent_config)
                trial_rel = dict(rel_config)
                if kind == "entity":
                    trial_ent[label] = rule
                else:
                    trial_rel[label] = rule
                metrics = score(gold, preds, vote_e, vote_r, trial_ent, trial_rel)
                total = metrics["total_score"]
                if total > local_best[0] + 1e-12:
                    local_best = (total, rule, metrics)
            if rule_key(local_best[1]) != rule_key(current):
                config[label] = local_best[1]
                best_total = local_best[0]
                best = local_best[2]
                improved = True
                history.append({
                    "step": f"round{round_no}:{kind}:{label}",
                    "label": label,
                    "kind": kind,
                    "members": list(local_best[1]["members"]),
                    "threshold": local_best[1]["threshold"],
                    "total_score": best_total,
                    "score_ner": best["score_ner"],
                    "score_re": best["score_re"],
                })
        if not improved:
            break
    return ent_config, rel_config, best, history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--vote_e", type=int, default=3)
    ap.add_argument("--vote_r", type=int, default=2)
    ap.add_argument("--ent_label_threshold", action="append", default=[])
    ap.add_argument("--rel_label_threshold", action="append", default=[])
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--max_threshold", type=int, default=5)
    ap.add_argument("--output_config", required=True)
    ap.add_argument("--output_history", required=True)
    args = ap.parse_args()

    preds = [load_json(path) for path in args.inputs]
    lengths = {len(records) for records in preds}
    if len(lengths) != 1:
        raise SystemExit(f"member output lengths differ: {sorted(lengths)}")
    gold = load_json(args.gold)
    n_members = len(preds)

    ent_config = parse_thresholds(args.ent_label_threshold, n_members)
    rel_config = parse_thresholds(args.rel_label_threshold, n_members)
    ent_config, rel_config, metrics, history = search(
        gold, preds, args.vote_e, args.vote_r, ent_config, rel_config,
        args.rounds, args.max_threshold
    )

    dump_config(args.output_config, ent_config, rel_config)
    Path(args.output_history).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_history, "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "history": history}, f, ensure_ascii=False, indent=2)

    print_metrics(metrics, title="BEST LABEL CONFIG")
    print(f"[OK] config -> {args.output_config}")
    print(f"[OK] history -> {args.output_history}")


if __name__ == "__main__":
    main()
