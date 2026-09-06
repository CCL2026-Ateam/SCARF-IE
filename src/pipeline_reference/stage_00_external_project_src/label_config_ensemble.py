"""Flexible per-label member selection ensemble.

Each label can choose which member files are allowed to vote and how many votes
are required. This is useful when dev data shows that a member is helpful for
some labels but noisy for others.
"""
import argparse
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_submission import to_submission  # noqa: E402
from common import dump_json, evaluate, load_json, print_metrics  # noqa: E402
from ensemble import ekey, rkey  # noqa: E402


def parse_mask(mask, n_members):
    if mask.isdigit() and len(mask) == n_members and set(mask) <= {"0", "1"}:
        return tuple(i for i, bit in enumerate(mask) if bit == "1")
    idxs = []
    for part in mask.split(","):
        part = part.strip()
        if part:
            idxs.append(int(part))
    if not idxs:
        raise ValueError("empty member mask")
    if min(idxs) < 0 or max(idxs) >= n_members:
        raise ValueError(f"member index out of range for {n_members} members: {mask}")
    return tuple(sorted(set(idxs)))


def parse_config_items(items, n_members):
    config = {}
    for item in items:
        parts = item.split(":")
        if len(parts) != 3:
            raise SystemExit(f"bad config {item!r}, expected LABEL:MASK:THRESHOLD")
        label, mask, threshold = parts
        try:
            config[label.strip()] = {
                "members": parse_mask(mask.strip(), n_members),
                "threshold": int(threshold.strip()),
            }
        except ValueError as exc:
            raise SystemExit(f"bad config {item!r}: {exc}") from exc
    return config


def normalize_config(config, n_members):
    normalized = {}
    for label, value in config.items():
        members = value.get("members", range(n_members))
        normalized[label] = {
            "members": tuple(members),
            "threshold": int(value["threshold"]),
        }
    return normalized


def load_config(path, n_members):
    raw = load_json(path)
    return {
        "entity": normalize_config(raw.get("entity", {}), n_members),
        "relation": normalize_config(raw.get("relation", {}), n_members),
    }


def dump_config(path, ent_config, rel_config):
    def clean(config):
        return {
            label: {"members": list(value["members"]), "threshold": value["threshold"]}
            for label, value in sorted(config.items())
        }
    dump_json(path, {"entity": clean(ent_config), "relation": clean(rel_config)})


def default_rule(config, label, n_members, default_threshold):
    return config.get(label, {"members": tuple(range(n_members)), "threshold": default_threshold})


def flexible_ensemble(preds, vote_e, vote_r, ent_config=None, rel_config=None):
    ent_config = ent_config or {}
    rel_config = rel_config or {}
    n_members = len(preds)
    n_records = len(preds[0])
    out = []
    for i in range(n_records):
        ent_votes = Counter()
        ent_obj = {}
        rel_votes = Counter()
        rel_obj = {}
        text = preds[0][i]["text"]

        for member_idx, member in enumerate(preds):
            rec = member[i]
            seen_e = set()
            for ent in rec.get("entities", []):
                key = ekey(ent)
                if key in seen_e:
                    continue
                seen_e.add(key)
                rule = default_rule(ent_config, key[2], n_members, vote_e)
                if member_idx in rule["members"]:
                    ent_votes[key] += 1
                    ent_obj[key] = ent

            seen_r = set()
            for rel in rec.get("relations", []):
                key = rkey(rel)
                if key in seen_r:
                    continue
                seen_r.add(key)
                rule = default_rule(rel_config, key[6], n_members, vote_r)
                if member_idx in rule["members"]:
                    rel_votes[key] += 1
                    rel_obj[key] = rel

        ents = []
        for key, votes in ent_votes.items():
            rule = default_rule(ent_config, key[2], n_members, vote_e)
            if votes >= rule["threshold"]:
                ents.append(ent_obj[key])
        eset = set(ekey(ent) for ent in ents)

        rels = []
        for key, votes in rel_votes.items():
            rule = default_rule(rel_config, key[6], n_members, vote_r)
            if votes < rule["threshold"]:
                continue
            rel = rel_obj[key]
            if (rel["head_start"], rel["head_end"], rel["head_type"]) in eset and \
               (rel["tail_start"], rel["tail_end"], rel["tail_type"]) in eset:
                rels.append(rel)

        ents.sort(key=lambda ent: (ent["start"], ent["end"], ent["label"]))
        rels.sort(key=lambda rel: (rel["head_start"], rel["head_end"], rel["tail_start"], rel["tail_end"], rel["label"]))
        out.append({"text": text, "entities": ents, "relations": rels})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--ensemble_out", required=True)
    ap.add_argument("--submit_out", required=True)
    ap.add_argument("--vote_e", type=int, default=3)
    ap.add_argument("--vote_r", type=int, default=2)
    ap.add_argument("--ent_config", action="append", default=[], help="LABEL:MASK:K, e.g. CROP:1110:2")
    ap.add_argument("--rel_config", action="append", default=[], help="LABEL:MASK:K, e.g. CON:1111:1")
    ap.add_argument("--config_json", default=None)
    ap.add_argument("--gold", default=None)
    ap.add_argument("--zip", default=None)
    args = ap.parse_args()

    preds = [load_json(path) for path in args.inputs]
    lengths = {len(records) for records in preds}
    if len(lengths) != 1:
        raise SystemExit(f"member output lengths differ: {sorted(lengths)}")

    n_members = len(preds)
    ent_config = parse_config_items(args.ent_config, n_members)
    rel_config = parse_config_items(args.rel_config, n_members)
    if args.config_json:
        loaded = load_config(args.config_json, n_members)
        ent_config.update(loaded["entity"])
        rel_config.update(loaded["relation"])

    combined = flexible_ensemble(preds, args.vote_e, args.vote_r, ent_config, rel_config)
    dump_json(args.ensemble_out, combined)
    print(f"[OK] flexible ensemble {n_members} members x {len(combined)} records -> {args.ensemble_out}")

    if args.gold:
        metrics = evaluate(load_json(args.gold), combined)
        print_metrics(metrics, title="FLEXIBLE ENSEMBLE")

    submission = to_submission(combined)
    dump_json(args.submit_out, submission)
    print(f"[OK] submission -> {args.submit_out}")

    if args.zip:
        with zipfile.ZipFile(args.zip, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(args.submit_out, arcname="submit.json")
        print(f"[OK] zip -> {args.zip}")


if __name__ == "__main__":
    main()
