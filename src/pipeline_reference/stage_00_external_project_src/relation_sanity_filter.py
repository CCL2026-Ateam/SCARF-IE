"""Post-filter low-confidence relation edges in a submit.json / submit.zip.

The filter is intentionally relation-first: entities are preserved by default
because NER is scored separately, while RE carries 60% of the leaderboard score.

Typical use:
  python src/relation_sanity_filter.py \
    --input outputs/testB_augmented_pool_4member_best_noempty_submit.zip \
    --output outputs/testB_best_relfilter_balanced_submit.json \
    --zip_out outputs/testB_best_relfilter_balanced_submit.zip \
    --mode balanced
"""
import argparse
import json
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import dump_json, evaluate, load_json, print_metrics  # noqa: E402


CORE_TYPE_PAIRS = {
    "CON": {
        ("CROP", "VAR"), ("CROP", "CROP"), ("GENE", "GENE"),
        ("VAR", "VAR"), ("CROSS", "VAR"), ("QTL", "QTL"),
        ("TRT", "TRT"), ("BM", "BM"), ("CHR", "CHR"),
    },
    "HAS": {
        ("VAR", "TRT"), ("CROP", "TRT"), ("QTL", "TRT"),
        ("GENE", "TRT"), ("TRT", "TRT"),
    },
    "AFF": {
        ("GENE", "TRT"), ("ABS", "TRT"), ("ABS", "GENE"),
        ("QTL", "TRT"), ("TRT", "TRT"), ("GENE", "GENE"),
        ("BIS", "TRT"),
    },
    "LOI": {
        ("QTL", "TRT"), ("QTL", "CHR"), ("GENE", "TRT"),
        ("GENE", "QTL"), ("GENE", "CHR"), ("MRK", "QTL"),
        ("MRK", "CHR"), ("MRK", "TRT"), ("MRK", "GENE"),
        ("QTL", "MRK"), ("QTL", "GENE"), ("QTL", "QTL"),
    },
    "OCI": {
        ("TRT", "GST"), ("ABS", "GST"), ("BIS", "GST"),
    },
    "USE": {
        ("BM", "GENE"), ("BM", "QTL"), ("BM", "TRT"),
        ("BM", "MRK"), ("BM", "BM"), ("VAR", "BM"),
        ("CROP", "BM"), ("GENE", "BM"),
    },
}


MEDIUM_TYPE_PAIRS = {
    "CON": {
        ("CROP", "GENE"), ("CROP", "QTL"), ("CROP", "CROSS"),
        ("CROSS", "CROP"), ("VAR", "CROSS"), ("GENE", "VAR"),
        ("VAR", "GENE"), ("MRK", "MRK"), ("ABS", "ABS"),
        ("BIS", "BIS"),
    },
    "HAS": {
        ("VAR", "VAR"), ("CROP", "ABS"), ("ABS", "TRT"),
    },
    "AFF": {
        ("BM", "TRT"), ("VAR", "TRT"), ("VAR", "ABS"),
        ("ABS", "CROP"), ("GENE", "ABS"), ("GENE", "BIS"),
        ("BIS", "GENE"), ("BIS", "BIS"),
    },
    "LOI": {
        ("CHR", "TRT"), ("CHR", "QTL"), ("TRT", "CHR"),
        ("TRT", "MRK"), ("TRT", "GST"), ("VAR", "TRT"),
        ("BM", "GENE"),
    },
    "OCI": {
        ("GST", "TRT"), ("TRT", "TRT"), ("TRT", "ABS"),
        ("ABS", "TRT"),
    },
    "USE": {
        ("BM", "CROSS"), ("BM", "ABS"), ("MRK", "BM"),
        ("MRK", "TRT"), ("QTL", "BM"), ("VAR", "CROSS"),
        ("VAR", "QTL"), ("CROSS", "BM"),
    },
}


# High-risk patterns seen in over-generated submissions.
BLOCKED_TYPE_PAIRS = {
    ("USE", "MRK", "CROP"),
    ("USE", "MRK", "QTL"),
    ("USE", "GENE", "CROP"),
    ("USE", "GENE", "TRT"),
    ("HAS", "BM", "VAR"),
    ("HAS", "CROP", "VAR"),
    ("LOI", "QTL", "ABS"),
    ("LOI", "GENE", "ABS"),
    ("LOI", "TRT", "TRT"),
    ("CON", "CROP", "BM"),
    ("CON", "CROP", "MRK"),
    ("CON", "TRT", "VAR"),
    ("CON", "QTL", "CROP"),
}


DISTANCE_LIMITS = {
    "conservative": {"CON": 230, "HAS": 180, "AFF": 220, "LOI": 180, "OCI": 160, "USE": 150},
    "balanced": {"CON": 170, "HAS": 140, "AFF": 180, "LOI": 140, "OCI": 120, "USE": 120},
    "aggressive": {"CON": 130, "HAS": 110, "AFF": 145, "LOI": 115, "OCI": 95, "USE": 90},
}


LIST_CAPS = {
    "conservative": {"CON": 10, "HAS": 6, "AFF": 6, "LOI": 7, "OCI": 4, "USE": 3},
    "balanced": {"CON": 8, "HAS": 4, "AFF": 4, "LOI": 5, "OCI": 3, "USE": 2},
    "aggressive": {"CON": 6, "HAS": 3, "AFF": 3, "LOI": 4, "OCI": 2, "USE": 1},
}


TRIGGER_PATTERNS = {
    "USE": re.compile(
        r"\b(used?|using|employ(?:ed|s|ing)?|utili[sz](?:ed|es|ing)?|"
        r"adopt(?:ed|s|ing)?|appl(?:ied|y|ies)|method|analysis|"
        r"sequenc(?:ing|ed)|GWAS|mapping|association analysis|model)\b",
        re.I,
    ),
    "LOI": re.compile(
        r"\b(locat(?:ed|es|ing|ion)|map(?:ped|s|ping)|chromosome|chr\.?|"
        r"linkage group|interval|locus|loci|QTL|marker|region)\b",
        re.I,
    ),
    "AFF": re.compile(
        r"\b(affect(?:ed|s|ing)?|regulat(?:ed|es|ing|ion)|improv(?:ed|es|ing)?|"
        r"increas(?:ed|es|ing)?|decreas(?:ed|es|ing)?|associated with|"
        r"confer(?:red|s|ring)?|control(?:led|s|ling)?|response|tolerance|"
        r"resistance|sensitivity|susceptib(?:le|ility)|enhanc(?:ed|es|ing)?)\b",
        re.I,
    ),
    "HAS": re.compile(
        r"\b(has|have|having|had|show(?:ed|s|ing)?|exhibit(?:ed|s|ing)?|"
        r"trait|quality|yield|resistance|tolerance|susceptib(?:le|ility))\b",
        re.I,
    ),
    "CON": re.compile(
        r"\b(including|include(?:d|s)?|contain(?:ed|s|ing)?|belong(?:ed|s)?|"
        r"genotype|cultivar|variet(?:y|ies)|cross|parent|cluster|haplotype)\b",
        re.I,
    ),
    "OCI": re.compile(
        r"\b(stage|period|phase|seedling|matur(?:e|ity)|flowering|heading|"
        r"germination|jointing|booting|grain filling|adult plants?)\b",
        re.I,
    ),
}


def load_submit(path):
    path = Path(path)
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            return json.loads(zf.read("submit.json").decode("utf-8"))
    return load_json(path)


def write_zip(json_path, zip_path):
    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(json_path, arcname="submit.json")


def entity_span_map(record):
    return {
        (int(e["start"]), int(e["end"]), str(e["text"])): str(e["label"])
        for e in record.get("entities", [])
    }


def relation_key(rel):
    return (
        int(rel["head_start"]), int(rel["head_end"]), str(rel["head_type"]),
        int(rel["tail_start"]), int(rel["tail_end"]), str(rel["tail_type"]),
        str(rel["label"]),
    )


def endpoint_exists(rel, side, spans):
    key = (int(rel[f"{side}_start"]), int(rel[f"{side}_end"]), str(rel[side]))
    return key in spans


def fill_types(rel, spans):
    out = dict(rel)
    for side in ("head", "tail"):
        key = (int(out[f"{side}_start"]), int(out[f"{side}_end"]), str(out[side]))
        if not out.get(f"{side}_type") and key in spans:
            out[f"{side}_type"] = spans[key]
    return out


def sorted_relations(relations):
    return sorted(
        relations,
        key=lambda r: (
            int(r["head_start"]), int(r["head_end"]),
            int(r["tail_start"]), int(r["tail_end"]),
            str(r["label"]),
        ),
    )


def distance(rel):
    return abs(int(rel["head_start"]) - int(rel["tail_start"]))


def context_window(text, rel, margin=80):
    start = max(0, min(int(rel["head_start"]), int(rel["tail_start"])) - margin)
    end = min(len(text), max(int(rel["head_end"]), int(rel["tail_end"])) + margin)
    return text[start:end]


def between_text(text, rel):
    left_end = min(int(rel["head_end"]), int(rel["tail_end"]))
    right_start = max(int(rel["head_start"]), int(rel["tail_start"]))
    if left_end > right_start:
        return ""
    return text[left_end:right_start]


def same_sentence(text, rel):
    span = between_text(text, rel)
    return not re.search(r"[.;!?]\s+[A-Z0-9]", span)


def trigger_present(text, rel):
    label = str(rel["label"])
    pat = TRIGGER_PATTERNS.get(label)
    if not pat:
        return True
    return bool(pat.search(context_window(text, rel)))


def type_pair_rank(label, head_type, tail_type):
    pair = (head_type, tail_type)
    if (label, head_type, tail_type) in BLOCKED_TYPE_PAIRS:
        return "blocked"
    if pair in CORE_TYPE_PAIRS.get(label, set()):
        return "core"
    if pair in MEDIUM_TYPE_PAIRS.get(label, set()):
        return "medium"
    return "rare"


def generic_relation_guard(rel, mode):
    """Hard semantic guards that are stable across modes."""
    label = str(rel["label"])
    ht = str(rel["head_type"])
    tt = str(rel["tail_type"])

    # In conservative mode, keep rare-but-gold-observed patterns such as
    # VAR->BM USE or GST->TRT OCI. Those are risky to prune without votes.
    if mode == "conservative":
        if label == "AFF" and ht in {"CROP", "CHR", "MRK"}:
            return False, "aff_bad_head_type"
        if label == "LOI" and ht in {"CROP", "CROSS", "BIS"}:
            return False, "loi_bad_head_type"
        return True, ""

    if label == "USE" and ht != "BM":
        return False, "use_head_not_bm"
    if label == "HAS" and tt not in {"TRT", "VAR", "ABS", "GENE", "GST"}:
        return False, "has_bad_tail_type"
    if label == "OCI" and tt != "GST" and (ht, tt) not in {("TRT", "TRT"), ("ABS", "TRT")}:
        return False, "oci_bad_tail_type"
    if label == "AFF" and ht in {"CROP", "CHR", "MRK"}:
        return False, "aff_bad_head_type"
    if label == "LOI" and ht in {"CROP", "VAR", "CROSS", "BIS"}:
        return False, "loi_bad_head_type"
    return True, ""


def score_relation_for_list_cap(text, rel, rank):
    score = 0
    if rank == "core":
        score += 4
    elif rank == "medium":
        score += 2
    if same_sentence(text, rel):
        score += 2
    if trigger_present(text, rel):
        score += 2
    score -= min(distance(rel) // 60, 4)
    return score


def should_keep_relation(text, rel, mode):
    ok, reason = generic_relation_guard(rel, mode)
    if not ok:
        return False, reason

    label = str(rel["label"])
    ht = str(rel["head_type"])
    tt = str(rel["tail_type"])
    rank = type_pair_rank(label, ht, tt)

    if rank == "blocked":
        return False, "blocked_type_pair"

    dist = distance(rel)
    limit = DISTANCE_LIMITS[mode][label]
    has_trigger = trigger_present(text, rel)
    in_same_sentence = same_sentence(text, rel)

    if mode == "conservative":
        return True, ""

    if dist > limit:
        if mode == "conservative" and (rank in {"core", "medium"} or has_trigger) and dist <= int(limit * 1.35):
            return True, ""
        return False, f"long_distance_{label}"

    if rank == "rare":
        if mode == "conservative":
            return True, ""
        else:
            return False, "rare_type_pair"

    if rank == "medium" and mode == "aggressive":
        if not (has_trigger and in_same_sentence):
            return False, "medium_type_pair_weak_evidence"

    if label in {"USE", "AFF", "LOI"} and not has_trigger:
        if mode != "conservative" or rank != "core":
            return False, f"missing_{label.lower()}_trigger"

    return True, ""


def apply_list_caps(text, relations, mode, stats):
    if mode == "conservative":
        return relations

    caps = LIST_CAPS[mode]
    grouped = defaultdict(list)
    for rel in relations:
        for side in ("head", "tail"):
            grouped[(rel["label"], side, rel[f"{side}_start"], rel[f"{side}_end"], rel[f"{side}_type"])].append(rel)

    keep_ids = {id(rel) for rel in relations}
    for (label, _side, _start, _end, _typ), items in grouped.items():
        cap = caps.get(label, 99)
        if len(items) <= cap:
            continue
        ranked = sorted(
            items,
            key=lambda r: (
                score_relation_for_list_cap(text, r, type_pair_rank(r["label"], r["head_type"], r["tail_type"])),
                -distance(r),
            ),
            reverse=True,
        )
        for rel in ranked[cap:]:
            if id(rel) in keep_ids:
                keep_ids.remove(id(rel))
                stats["dropped_by_reason"]["list_explosion_cap"] += 1
                stats["dropped_by_label"][rel["label"]] += 1

    return [rel for rel in relations if id(rel) in keep_ids]


def filter_record(record, mode, stats):
    text = record["text"]
    spans = entity_span_map(record)
    kept = []
    seen = set()

    for raw in record.get("relations", []):
        rel = fill_types(raw, spans)
        label = str(rel.get("label", ""))
        if label not in CORE_TYPE_PAIRS:
            stats["dropped_by_reason"]["bad_relation_label"] += 1
            continue

        if not endpoint_exists(rel, "head", spans) or not endpoint_exists(rel, "tail", spans):
            stats["dropped_by_reason"]["missing_endpoint_entity"] += 1
            stats["dropped_by_label"][label] += 1
            continue

        if text[int(rel["head_start"]):int(rel["head_end"])] != rel["head"]:
            stats["dropped_by_reason"]["bad_head_span_text"] += 1
            stats["dropped_by_label"][label] += 1
            continue
        if text[int(rel["tail_start"]):int(rel["tail_end"])] != rel["tail"]:
            stats["dropped_by_reason"]["bad_tail_span_text"] += 1
            stats["dropped_by_label"][label] += 1
            continue

        keep, reason = should_keep_relation(text, rel, mode)
        if not keep:
            stats["dropped_by_reason"][reason] += 1
            stats["dropped_by_label"][label] += 1
            continue

        key = relation_key(rel)
        if key in seen:
            stats["dropped_by_reason"]["duplicate_relation"] += 1
            stats["dropped_by_label"][label] += 1
            continue
        seen.add(key)
        kept.append(rel)

    kept = apply_list_caps(text, kept, mode, stats)
    out = dict(record)
    out["entities"] = list(record.get("entities", []))
    out["relations"] = sorted_relations(kept)
    return out


def make_summary(before, after, stats, mode):
    before_labels = Counter()
    after_labels = Counter()
    for rec in before:
        before_labels.update(r.get("label", "") for r in rec.get("relations", []))
    for rec in after:
        after_labels.update(r.get("label", "") for r in rec.get("relations", []))
    return {
        "mode": mode,
        "records": len(after),
        "entities_before": sum(len(r.get("entities", [])) for r in before),
        "entities_after": sum(len(r.get("entities", [])) for r in after),
        "relations_before": sum(len(r.get("relations", [])) for r in before),
        "relations_after": sum(len(r.get("relations", [])) for r in after),
        "relations_dropped": sum(len(r.get("relations", [])) for r in before) - sum(len(r.get("relations", [])) for r in after),
        "relations_before_by_label": dict(sorted(before_labels.items())),
        "relations_after_by_label": dict(sorted(after_labels.items())),
        "dropped_by_label": dict(sorted(stats["dropped_by_label"].items())),
        "dropped_by_reason": dict(stats["dropped_by_reason"].most_common()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="submit.json or submit.zip with submit.json inside")
    ap.add_argument("--output", required=True, help="filtered submit.json")
    ap.add_argument("--zip_out", default=None, help="optional filtered submit.zip")
    ap.add_argument("--summary", default=None, help="optional JSON summary")
    ap.add_argument("--mode", choices=["conservative", "balanced", "aggressive"], default="conservative")
    ap.add_argument("--gold", default=None, help="optional gold json for dev evaluation")
    args = ap.parse_args()

    records = load_submit(args.input)
    stats = {"dropped_by_reason": Counter(), "dropped_by_label": Counter()}
    filtered = [filter_record(rec, args.mode, stats) for rec in records]
    dump_json(args.output, filtered)
    if args.zip_out:
        write_zip(args.output, args.zip_out)

    summary = make_summary(records, filtered, stats, args.mode)
    summary_path = args.summary
    if summary_path is None:
        summary_path = str(Path(args.output).with_suffix(".filter_summary.json"))
    dump_json(summary_path, summary)

    print(f"[OK] wrote filtered json -> {args.output}")
    if args.zip_out:
        print(f"[OK] wrote filtered zip  -> {args.zip_out}")
    print(f"[OK] wrote summary       -> {summary_path}")
    print(
        f"[SUMMARY] mode={args.mode} relations "
        f"{summary['relations_before']} -> {summary['relations_after']} "
        f"(dropped {summary['relations_dropped']})"
    )
    if summary["dropped_by_label"]:
        print(f"[DROPPED BY LABEL] {summary['dropped_by_label']}")
    if args.gold:
        metrics = evaluate(load_json(args.gold), filtered)
        print_metrics(metrics, title=f"RELATION SANITY FILTER ({args.mode})")


if __name__ == "__main__":
    main()
