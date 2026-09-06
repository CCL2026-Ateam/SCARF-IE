"""Rank records by ensemble disagreement for targeted iteration.

The score is a triage heuristic, not an official metric. Records with many
borderline entities/relations are good candidates for one-by-one reruns or
manual review because a small correction can change the final ensemble.

Example:
  python src/rank_uncertain_records.py \
    --inputs outputs/testA_sonnet_k4.json outputs/testA_opus_k4.json \
             outputs/testA_sonnet_k4_conv.json outputs/testA_opus_k4_conv.json \
    --output_csv outputs/testA_uncertain_top80.csv \
    --output_json outputs/testA_uncertain_full.json \
    --vote_e 3 --vote_r 2 \
    --rel_label_threshold CON:3 --rel_label_threshold USE:4 \
    --rel_label_threshold HAS:4 --rel_label_threshold AFF:4 \
    --rel_label_threshold OCI:2 --rel_label_threshold LOI:2 \
    --top 80
"""
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_json  # noqa: E402
from ensemble import ekey, rkey  # noqa: E402


def parse_thresholds(items):
    thresholds = {}
    for item in items:
        if ":" not in item:
            raise SystemExit(f"bad --rel_label_threshold: {item!r}, expected LABEL:K")
        label, value = item.split(":", 1)
        try:
            thresholds[label.strip()] = int(value.strip())
        except ValueError as exc:
            raise SystemExit(f"bad threshold value in {item!r}") from exc
    return thresholds


def short_text(text, limit=180):
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[:limit - 3] + "..."


def summarize_candidate(obj, kind, votes):
    if kind == "entity":
        return {
            "votes": votes,
            "text": obj.get("text", ""),
            "label": obj.get("label", ""),
            "start": obj.get("start"),
            "end": obj.get("end"),
        }
    return {
        "votes": votes,
        "label": obj.get("label", ""),
        "head": obj.get("head", ""),
        "head_type": obj.get("head_type", ""),
        "head_start": obj.get("head_start"),
        "head_end": obj.get("head_end"),
        "tail": obj.get("tail", ""),
        "tail_type": obj.get("tail_type", ""),
        "tail_start": obj.get("tail_start"),
        "tail_end": obj.get("tail_end"),
    }


def rank_records(preds, vote_e, vote_r, ent_thresholds, rel_thresholds):
    n_records = len(preds[0])
    rows = []
    for idx in range(n_records):
        ent_votes = Counter()
        rel_votes = Counter()
        ent_obj = {}
        rel_obj = {}
        member_ent_counts = []
        member_rel_counts = []
        text = preds[0][idx]["text"]

        for member in preds:
            rec = member[idx]
            member_ent_counts.append(len(rec.get("entities", [])))
            member_rel_counts.append(len(rec.get("relations", [])))

            seen_e = set()
            for ent in rec.get("entities", []):
                key = ekey(ent)
                if key in seen_e:
                    continue
                seen_e.add(key)
                ent_votes[key] += 1
                ent_obj[key] = ent

            seen_r = set()
            for rel in rec.get("relations", []):
                key = rkey(rel)
                if key in seen_r:
                    continue
                seen_r.add(key)
                rel_votes[key] += 1
                rel_obj[key] = rel

        kept_entities = sum(1 for votes in ent_votes.values() if votes >= vote_e)
        borderline_entities = []
        for key, votes in ent_votes.items():
            threshold = ent_thresholds.get(key[2], vote_e)
            if votes == threshold - 1:
                borderline_entities.append(summarize_candidate(ent_obj[key], "entity", votes))

        kept_relations = 0
        borderline_relations = []
        for key, votes in rel_votes.items():
            threshold = rel_thresholds.get(key[6], vote_r)
            if votes >= threshold:
                kept_relations += 1
            elif votes == threshold - 1:
                borderline_relations.append(summarize_candidate(rel_obj[key], "relation", votes))

        unique_entities = len(ent_votes)
        unique_relations = len(rel_votes)
        ent_spread = max(member_ent_counts) - min(member_ent_counts)
        rel_spread = max(member_rel_counts) - min(member_rel_counts)

        score = (
            4 * len(borderline_relations)
            + 2 * len(borderline_entities)
            + 1.5 * rel_spread
            + 0.7 * ent_spread
            + 0.2 * unique_relations
            + 0.1 * unique_entities
        )

        rows.append({
            "index": idx,
            "score": round(score, 3),
            "text_len": len(text),
            "text_preview": short_text(text),
            "member_entity_counts": member_ent_counts,
            "member_relation_counts": member_rel_counts,
            "unique_entities": unique_entities,
            "unique_relations": unique_relations,
            "kept_entities": kept_entities,
            "kept_relations": kept_relations,
            "borderline_entities": borderline_entities,
            "borderline_relations": borderline_relations,
            "borderline_entity_count": len(borderline_entities),
            "borderline_relation_count": len(borderline_relations),
        })

    rows.sort(key=lambda row: (-row["score"], row["index"]))
    return rows


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "rank", "index", "score", "text_len", "borderline_relation_count",
        "borderline_entity_count", "kept_relations", "kept_entities",
        "unique_relations", "unique_entities", "member_relation_counts",
        "member_entity_counts", "text_preview",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for rank, row in enumerate(rows, 1):
            out = {field: row.get(field) for field in fields}
            out["rank"] = rank
            out["member_relation_counts"] = json.dumps(row["member_relation_counts"], ensure_ascii=False)
            out["member_entity_counts"] = json.dumps(row["member_entity_counts"], ensure_ascii=False)
            writer.writerow(out)


def write_json(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="aligned member prediction json files")
    ap.add_argument("--output_csv", required=True)
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--vote_e", type=int, default=3)
    ap.add_argument("--vote_r", type=int, default=3)
    ap.add_argument("--ent_label_threshold", action="append", default=[])
    ap.add_argument("--rel_label_threshold", action="append", default=[])
    ap.add_argument("--top", type=int, default=80, help="number of rows to write to CSV")
    args = ap.parse_args()

    preds = [load_json(path) for path in args.inputs]
    lengths = {len(records) for records in preds}
    if len(lengths) != 1:
        raise SystemExit(f"member output lengths differ: {sorted(lengths)}")

    rows = rank_records(
        preds, args.vote_e, args.vote_r,
        parse_thresholds(args.ent_label_threshold),
        parse_thresholds(args.rel_label_threshold),
    )
    write_csv(args.output_csv, rows[:args.top])
    write_json(args.output_json, rows)
    print(f"[OK] ranked {len(rows)} records -> {args.output_csv}, {args.output_json}")
    for rank, row in enumerate(rows[:10], 1):
        print(f"#{rank:02d} idx={row['index']} score={row['score']} "
              f"borderline_rel={row['borderline_relation_count']} "
              f"borderline_ent={row['borderline_entity_count']} "
              f"rel_counts={row['member_relation_counts']}")


if __name__ == "__main__":
    main()
