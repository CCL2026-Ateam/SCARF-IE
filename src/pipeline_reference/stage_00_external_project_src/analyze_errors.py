"""Print focused error examples for labeled MGBIE data.

This script compares a gold file and a prediction file aligned by record index.
It is meant for quick diagnosis after running the online ensemble on dev/train.
"""
import argparse
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import evaluate, load_json, print_metrics  # noqa: E402


def ent_key(e):
    return (e["start"], e["end"], e["label"])


def rel_key(r):
    return (
        r["head_start"], r["head_end"], r["head_type"],
        r["tail_start"], r["tail_end"], r["tail_type"],
        r["label"],
    )


def ent_label(k):
    return k[2]


def rel_label(k):
    return k[6]


def short_text(text, width=220):
    text = " ".join(text.split())
    return text if len(text) <= width else text[:width - 3] + "..."


def fmt_ent(e):
    return f"{e.get('text', '')!r}/{e['label']}[{e['start']}:{e['end']}]"


def fmt_rel(r):
    return (
        f"{r.get('head', '')!r}/{r['head_type']}[{r['head_start']}:{r['head_end']}] "
        f"-{r['label']}-> "
        f"{r.get('tail', '')!r}/{r['tail_type']}[{r['tail_start']}:{r['tail_end']}]"
    )


def top_items(items, n):
    return items[:n] if n > 0 else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--examples", type=int, default=8)
    ap.add_argument("--per_kind", type=int, default=6)
    args = ap.parse_args()

    gold = load_json(args.gold)
    pred = load_json(args.pred)
    if args.limit is not None:
        gold = gold[:args.limit]
        pred = pred[:args.limit]

    metrics = evaluate(gold, pred)
    print_metrics(metrics, title="ERROR ANALYSIS INPUT")
    print()

    ent_fp_labels, ent_fn_labels = Counter(), Counter()
    rel_fp_labels, rel_fn_labels = Counter(), Counter()
    rows = []

    for i, (g, p) in enumerate(zip(gold, pred)):
        g_ent_map = {ent_key(e): e for e in g.get("entities", [])}
        p_ent_map = {ent_key(e): e for e in p.get("entities", [])}
        g_rel_map = {rel_key(r): r for r in g.get("relations", [])}
        p_rel_map = {rel_key(r): r for r in p.get("relations", [])}

        g_ents, p_ents = set(g_ent_map), set(p_ent_map)
        g_rels, p_rels = set(g_rel_map), set(p_rel_map)

        ent_fp = sorted(p_ents - g_ents)
        ent_fn = sorted(g_ents - p_ents)
        rel_fp = sorted(p_rels - g_rels)
        rel_fn = sorted(g_rels - p_rels)

        for k in ent_fp:
            ent_fp_labels[ent_label(k)] += 1
        for k in ent_fn:
            ent_fn_labels[ent_label(k)] += 1
        for k in rel_fp:
            rel_fp_labels[rel_label(k)] += 1
        for k in rel_fn:
            rel_fn_labels[rel_label(k)] += 1

        error_count = len(ent_fp) + len(ent_fn) + 2 * len(rel_fp) + 2 * len(rel_fn)
        if error_count:
            rows.append({
                "idx": i,
                "text": g["text"],
                "ent_fp": [p_ent_map[k] for k in ent_fp],
                "ent_fn": [g_ent_map[k] for k in ent_fn],
                "rel_fp": [p_rel_map[k] for k in rel_fp],
                "rel_fn": [g_rel_map[k] for k in rel_fn],
                "error_count": error_count,
            })

    print("[LABEL ERRORS]")
    print("  NER false positive:", dict(ent_fp_labels.most_common()))
    print("  NER false negative:", dict(ent_fn_labels.most_common()))
    print("  RE  false positive:", dict(rel_fp_labels.most_common()))
    print("  RE  false negative:", dict(rel_fn_labels.most_common()))
    print()

    rows.sort(key=lambda x: x["error_count"], reverse=True)
    print(f"[TOP {min(args.examples, len(rows))} ERROR EXAMPLES]")
    for row in rows[:args.examples]:
        print("=" * 100)
        print(f"IDX={row['idx']} weighted_errors={row['error_count']}")
        print(f"TEXT: {short_text(row['text'])}")
        if row["ent_fp"]:
            print("  NER FP:")
            for e in top_items(row["ent_fp"], args.per_kind):
                print("   +", fmt_ent(e))
        if row["ent_fn"]:
            print("  NER FN:")
            for e in top_items(row["ent_fn"], args.per_kind):
                print("   -", fmt_ent(e))
        if row["rel_fp"]:
            print("  RE FP:")
            for r in top_items(row["rel_fp"], args.per_kind):
                print("   +", fmt_rel(r))
        if row["rel_fn"]:
            print("  RE FN:")
            for r in top_items(row["rel_fn"], args.per_kind):
                print("   -", fmt_rel(r))


if __name__ == "__main__":
    main()
