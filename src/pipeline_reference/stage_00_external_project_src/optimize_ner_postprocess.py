"""Search add-only NER postprocess rules on a labeled dev set.

Rules add entities from selected member outputs when enough selected members
agree. Relations are left unchanged, so this is a conservative way to improve
NER without directly perturbing RE predictions.
"""
import argparse
import itertools
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ENTITY_LABELS, dump_json, evaluate, load_json, print_metrics  # noqa: E402
from ensemble import ekey  # noqa: E402
from ner_vote_postprocess import postprocess  # noqa: E402


def candidate_rules(n_members):
    rules = []
    for bits in itertools.product([0, 1], repeat=n_members):
        members = tuple(i for i, bit in enumerate(bits) if bit)
        if not members:
            continue
        for threshold in range(1, len(members) + 1):
            rules.append({"members": members, "threshold": threshold})
    return rules


def materialize_selected_members(all_members, config):
    selected = []
    for label, rule in config.items():
        label_members = []
        for idx in rule["members"]:
            label_members.append(all_members[idx])
        selected.append((label, label_members, rule["threshold"]))
    return selected


def apply_config(base, all_members, config):
    out = []
    selected = materialize_selected_members(all_members, config)
    for i, rec in enumerate(base):
        ents = list(rec.get("entities", []))
        seen = {ekey(ent) for ent in ents}
        for label, label_members, threshold in selected:
            votes = Counter()
            objects = {}
            for member in label_members:
                member_seen = set()
                for ent in member[i].get("entities", []):
                    key = ekey(ent)
                    if key[2] != label or key in member_seen:
                        continue
                    member_seen.add(key)
                    votes[key] += 1
                    objects[key] = ent
            for key, count in votes.items():
                if count >= threshold and key not in seen:
                    ents.append(objects[key])
                    seen.add(key)
        ents.sort(key=lambda ent: (ent["start"], ent["end"], ent["label"]))
        out.append({
            "text": rec["text"],
            "entities": ents,
            "relations": list(rec.get("relations", [])),
        })
    return out


def parse_seed(items, n_members):
    config = {}
    for item in items:
        parts = item.split(":")
        if len(parts) != 3:
            raise SystemExit(f"bad seed {item!r}, expected LABEL:MASK:K")
        label, mask, threshold = parts
        if mask.isdigit() and len(mask) == n_members and set(mask) <= {"0", "1"}:
            members = tuple(i for i, bit in enumerate(mask) if bit == "1")
        else:
            members = tuple(int(part) for part in mask.split(",") if part.strip())
        config[label.strip()] = {"members": members, "threshold": int(threshold)}
    return config


def clean_config(config):
    return {
        label: {"members": list(rule["members"]), "threshold": rule["threshold"]}
        for label, rule in sorted(config.items())
    }


def search(gold, base, members, seed, metric, rounds):
    n_members = len(members)
    rules = candidate_rules(n_members)
    config = dict(seed)
    best_pred = apply_config(base, members, config)
    best_metrics = evaluate(gold, best_pred)
    best_value = best_metrics["score_ner" if metric == "ner" else "total_score"]
    history = [{
        "step": "baseline",
        "metric": best_value,
        "total_score": best_metrics["total_score"],
        "score_ner": best_metrics["score_ner"],
        "score_re": best_metrics["score_re"],
        "config": clean_config(config),
    }]

    for round_no in range(1, rounds + 1):
        improved = False
        for label in ENTITY_LABELS:
            current = config.get(label)
            local_best = (best_value, current, best_metrics)
            for rule in rules:
                if current and tuple(current["members"]) == tuple(rule["members"]) and current["threshold"] == rule["threshold"]:
                    continue
                trial = dict(config)
                trial[label] = rule
                metrics = evaluate(gold, apply_config(base, members, trial))
                value = metrics["score_ner" if metric == "ner" else "total_score"]
                if value > local_best[0] + 1e-12:
                    local_best = (value, rule, metrics)
            if local_best[1] != current:
                config[label] = local_best[1]
                best_value = local_best[0]
                best_metrics = local_best[2]
                improved = True
                history.append({
                    "step": f"round{round_no}:{label}",
                    "metric": best_value,
                    "total_score": best_metrics["total_score"],
                    "score_ner": best_metrics["score_ner"],
                    "score_re": best_metrics["score_re"],
                    "label": label,
                    "rule": {"members": list(local_best[1]["members"]), "threshold": local_best[1]["threshold"]},
                })
        if not improved:
            break
    return config, best_metrics, history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--members", nargs="+", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--seed", action="append", default=[], help="LABEL:MASK:K, e.g. MRK:1111:4")
    ap.add_argument("--metric", choices=["total", "ner"], default="total")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--output_config", required=True)
    ap.add_argument("--output_history", required=True)
    ap.add_argument("--output_pred", required=True)
    args = ap.parse_args()

    base = load_json(args.base)
    members = [load_json(path) for path in args.members]
    lengths = {len(base)} | {len(member) for member in members}
    if len(lengths) != 1:
        raise SystemExit(f"input lengths differ: {sorted(lengths)}")

    gold = load_json(args.gold)
    seed = parse_seed(args.seed, len(members))
    config, metrics, history = search(gold, base, members, seed, args.metric, args.rounds)
    pred = apply_config(base, members, config)

    dump_json(args.output_config, clean_config(config))
    dump_json(args.output_history, {"metrics": metrics, "history": history})
    dump_json(args.output_pred, pred)
    print_metrics(metrics, title="BEST NER POSTPROCESS")
    print(f"[OK] config -> {args.output_config}")
    print(f"[OK] pred -> {args.output_pred}")


if __name__ == "__main__":
    main()
