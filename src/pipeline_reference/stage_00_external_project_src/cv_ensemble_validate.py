"""Cross-validate ensemble policies on dev without using test labels or APIs.

This script evaluates fixed voting/config policies across deterministic folds.
It is intended to distinguish genuine policy improvements from full-dev tuning.

Example:
  python src/cv_ensemble_validate.py \
    --gold data/dev.json \
    --members outputs/dev_sonnet_k4.json outputs/dev_opus_k4.json \
              outputs/dev_sonnet_k4_conv.json outputs/dev_opus_k4_conv.json \
    --config outputs/dev_opt_label_config_ens4.json
"""
import argparse
import itertools
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RELATION_LABELS, evaluate, load_json  # noqa: E402
from ensemble import ensemble  # noqa: E402
from label_config_ensemble import flexible_ensemble  # noqa: E402


ONLINE_ENTITY_THRESHOLDS = {
    "ABS": 2,
    "BIS": 2,
    "CHR": 2,
    "CROP": 2,
    "CROSS": 4,
    "GST": 2,
    "MRK": 4,
    "QTL": 2,
}

ONLINE_RELATION_THRESHOLDS = {
    "AFF": 3,
    "CON": 1,
    "HAS": 3,
    "LOI": 2,
    "OCI": 2,
    "USE": 5,
}


def subset(records, indices):
    return [records[i] for i in indices]


def subset_members(members, indices):
    return [subset(member, indices) for member in members]


def folds_for(n_records, n_folds):
    return [list(range(fold, n_records, n_folds)) for fold in range(n_folds)]


def normalize_config(raw):
    return {
        label: {"members": tuple(rule["members"]), "threshold": int(rule["threshold"])}
        for label, rule in raw.items()
    }


def eval_policy(name, gold, members, folds, make_pred):
    scores = []
    for fold_no, valid in enumerate(folds):
        train = [idx for idx in range(len(gold)) if idx not in set(valid)]
        pred = make_pred(train, valid)
        metrics = evaluate(subset(gold, valid), pred)
        scores.append(metrics["total_score"])
        print(
            f"{name}\tfold={fold_no}\t"
            f"total={metrics['total_score']:.6f}\t"
            f"ner={metrics['score_ner']:.6f}\t"
            f"re={metrics['score_re']:.6f}"
        )
    mean = statistics.mean(scores)
    std = statistics.pstdev(scores) if len(scores) > 1 else 0.0
    print(f"{name}\tmean={mean:.6f}\tstd={std:.6f}\tfolds={','.join(f'{s:.6f}' for s in scores)}")
    return mean


def search_global_policy(gold, members, train_indices):
    train_gold = subset(gold, train_indices)
    train_members = subset_members(members, train_indices)
    best = None
    for vote_e in (2, 3, 4):
        for vote_r in (1, 2, 3, 4):
            pred = ensemble(train_members, vote_e, vote_r)
            total = evaluate(train_gold, pred)["total_score"]
            if best is None or total > best[0]:
                best = (total, vote_e, vote_r)
    return best[1], best[2]


def search_relation_label_policy(gold, members, train_indices):
    train_gold = subset(gold, train_indices)
    train_members = subset_members(members, train_indices)
    best = None
    for values in itertools.product((2, 3, 4, 5), repeat=len(RELATION_LABELS)):
        rel_thresholds = dict(zip(RELATION_LABELS, values))
        pred = ensemble(train_members, vote_min=3, rel_vote_min=3, rel_label_thresholds=rel_thresholds)
        total = evaluate(train_gold, pred)["total_score"]
        if best is None or total > best[0]:
            best = (total, rel_thresholds)
    return best[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--members", nargs="+", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--config", default=None, help="optional label_config_ensemble JSON to evaluate as fixed policy")
    ap.add_argument("--search_rel_label", action="store_true",
                    help="also train-fold-search relation label thresholds; slow and often overfits")
    args = ap.parse_args()

    gold = load_json(args.gold)
    members = [load_json(path) for path in args.members]
    lengths = {len(gold)} | {len(member) for member in members}
    if len(lengths) != 1:
        raise SystemExit(f"input lengths differ: {sorted(lengths)}")
    folds = folds_for(len(gold), args.folds)

    def uniform_3_3(_train, valid):
        return ensemble(subset_members(members, valid), vote_min=3, rel_vote_min=3)

    def online_calibrated(_train, valid):
        return ensemble(
            subset_members(members, valid),
            vote_min=3,
            rel_vote_min=2,
            rel_label_thresholds=ONLINE_RELATION_THRESHOLDS,
            ent_label_thresholds=ONLINE_ENTITY_THRESHOLDS,
        )

    def searched_global(train, valid):
        vote_e, vote_r = search_global_policy(gold, members, train)
        return ensemble(subset_members(members, valid), vote_min=vote_e, rel_vote_min=vote_r)

    eval_policy("uniform_3_3", gold, members, folds, uniform_3_3)
    eval_policy("online_calibrated_fixed", gold, members, folds, online_calibrated)
    eval_policy("search_global_on_train", gold, members, folds, searched_global)

    if args.config:
        raw = load_json(args.config)
        ent_config = normalize_config(raw.get("entity", {}))
        rel_config = normalize_config(raw.get("relation", {}))

        def fixed_label_config(_train, valid):
            return flexible_ensemble(
                subset_members(members, valid),
                vote_e=3,
                vote_r=2,
                ent_config=ent_config,
                rel_config=rel_config,
            )

        eval_policy("fixed_label_config", gold, members, folds, fixed_label_config)

    if args.search_rel_label:
        def searched_rel_label(train, valid):
            rel_thresholds = search_relation_label_policy(gold, members, train)
            return ensemble(
                subset_members(members, valid),
                vote_min=3,
                rel_vote_min=3,
                rel_label_thresholds=rel_thresholds,
            )

        eval_policy("search_rel_label_on_train", gold, members, folds, searched_rel_label)


if __name__ == "__main__":
    main()
