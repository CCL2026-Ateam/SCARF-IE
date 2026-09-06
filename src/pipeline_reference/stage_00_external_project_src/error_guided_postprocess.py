"""Conservative post-processing rules derived from dev error examples.

The rules target high-confidence, recurring errors:
  1. GENE boundary suffixes such as "AsaPP2C genes" -> "AsaPP2C".
  2. Optionally, transgenic VAR spans such as "GsNAC2-overexpressing sorghum plants"
     -> add "GsNAC2"/GENE and keep "overexpressing sorghum plants"/VAR.
  3. Optionally, generic QTL/loci mentions are expanded when the local grammar is clear,
     e.g. "QTL on chromosome 5" and "Four loci".
  4. Relations are synchronized with changed entity endpoints and orphan
     relations are removed.
"""
import argparse
import re
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import dump_json, evaluate, load_json, print_metrics  # noqa: E402


GENE_SUFFIXES = (
    " genes",
    " gene",
    " proteins",
    " protein",
    " family",
)

COUNT_WORD_RE = re.compile(
    r"(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|"
    r"twenty|\d+)",
    re.I,
)


def entity_key(e):
    return (e["start"], e["end"], e["label"])


def endpoint_key_from_relation(r, side):
    return (r[f"{side}_start"], r[f"{side}_end"], r[f"{side}_type"])


def gene_like_single_token(text):
    if " " in text:
        return False
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]*", text):
        return False
    return bool(re.search(r"[A-Z]", text) and re.search(r"\d", text))


def make_entity(text, start, end, label):
    return {"start": start, "end": end, "text": text[start:end], "label": label}


def trim_gene_suffix(text, ent):
    value = ent["text"]
    lower = value.lower()
    for suffix in GENE_SUFFIXES:
        if lower.endswith(suffix):
            prefix = value[: -len(suffix)]
            if gene_like_single_token(prefix):
                end = ent["start"] + len(prefix)
                return make_entity(text, ent["start"], end, ent["label"])
    return ent


def split_overexpressing_var(text, ent):
    value = ent["text"]
    m = re.match(r"^([A-Za-z][A-Za-z0-9_.-]{2,})-(overexpressing .+)$", value)
    if not m:
        return None
    gene_text, var_tail = m.groups()
    if not gene_like_single_token(gene_text):
        return None
    gene = make_entity(text, ent["start"], ent["start"] + len(gene_text), "GENE")
    var_start = ent["start"] + len(gene_text) + 1
    var = make_entity(text, var_start, var_start + len(var_tail), "VAR")
    return gene, var


def expand_qtl_like(text, ent):
    if ent["label"] != "QTL":
        return ent
    value = ent["text"]
    lower = value.lower()
    if lower not in {"qtl", "qtls", "loci"}:
        return ent

    start, end = ent["start"], ent["end"]

    # "Four loci" / "19 QTLs" when the count is directly attached before it.
    prefix = text[max(0, start - 24):start]
    m = re.search(r"((?:%s)\s+)$" % COUNT_WORD_RE.pattern, prefix, re.I)
    if m:
        start = start - len(m.group(1))

    # "QTL on chromosome 5" / "QTL on chromosome 3".
    suffix = text[end:end + 48]
    m = re.match(r"( on chromosome(?: arm| arms)? [A-Za-z0-9]+)", suffix, re.I)
    if m:
        end = end + len(m.group(1))
        return make_entity(text, start, end, "QTL")

    # "QTLs for branch number on the main stem"; keep only short local spans.
    m = re.match(r"( for [^.;,()]{3,70})", suffix, re.I)
    if m and lower in {"qtl", "qtls", "loci"}:
        phrase = m.group(1)
        if not re.search(r"\b(one|two|three|four|five|six|seven|eight|nine|\d+)\s+QTLs?\b", phrase, re.I):
            end = end + len(phrase.rstrip())

    return make_entity(text, start, end, "QTL")


def expand_qtl_chromosome_only(text, ent):
    if ent["label"] != "QTL":
        return ent
    if ent["text"].lower() not in {"qtl", "qtls", "loci"}:
        return ent
    suffix = text[ent["end"]:ent["end"] + 48]
    m = re.match(r"( on chromosome(?: arm| arms)? [A-Za-z0-9]+)", suffix, re.I)
    if not m:
        return ent
    return make_entity(text, ent["start"], ent["end"] + len(m.group(1)), "QTL")


def maybe_snp_list_chr(text, ent):
    if ent["label"] != "MRK":
        return ent
    if not re.fullmatch(r"SNP\d+", ent["text"]):
        return ent
    if "SNP markers" not in text:
        return ent
    out = dict(ent)
    out["label"] = "CHR"
    return out


def add_gwas_bm_if_needed(text, entities):
    if "GWAS" not in text:
        return entities
    existing = {(e["start"], e["end"], e["label"]) for e in entities}
    out = list(entities)
    for m in re.finditer(r"\bGWAS\b", text):
        key = (m.start(), m.end(), "BM")
        if key not in existing:
            out.append(make_entity(text, m.start(), m.end(), "BM"))
            existing.add(key)
    return out


def dedup_entities(entities):
    seen = set()
    out = []
    for e in sorted(entities, key=lambda x: (x["start"], x["end"], x["label"], x["text"])):
        k = entity_key(e)
        if k in seen:
            continue
        seen.add(k)
        out.append(e)
    return out


def process_record(
    record,
    split_overexpressing=False,
    expand_qtl=False,
    expand_qtl_chromosome=False,
    snp_list_chr=False,
    add_gwas_bm=False,
    drop_overexpressing_has=False,
):
    text = record["text"]
    old_to_new = {}
    new_entities = []

    for ent in record.get("entities", []):
        original_key = entity_key(ent)
        current = dict(ent)

        if current["label"] == "GENE":
            current = trim_gene_suffix(text, current)

        if snp_list_chr:
            current = maybe_snp_list_chr(text, current)

        split = (
            split_overexpressing_var(text, current)
            if split_overexpressing and current["label"] == "VAR"
            else None
        )
        if split:
            gene, var = split
            new_entities.extend([gene, var])
            old_to_new[original_key] = entity_key(var)
            continue

        if expand_qtl:
            current = expand_qtl_like(text, current)
        elif expand_qtl_chromosome:
            current = expand_qtl_chromosome_only(text, current)
        new_entities.append(current)
        old_to_new[original_key] = entity_key(current)

    if add_gwas_bm:
        new_entities = add_gwas_bm_if_needed(text, new_entities)

    new_entities = dedup_entities(new_entities)
    entity_by_key = {entity_key(e): e for e in new_entities}

    new_relations = []
    seen_rel = set()
    for rel in record.get("relations", []):
        head_key = endpoint_key_from_relation(rel, "head")
        tail_key = endpoint_key_from_relation(rel, "tail")
        head_key = old_to_new.get(head_key, head_key)
        tail_key = old_to_new.get(tail_key, tail_key)

        if head_key not in entity_by_key or tail_key not in entity_by_key:
            continue

        head = entity_by_key[head_key]
        tail = entity_by_key[tail_key]
        if (
            drop_overexpressing_has
            and rel.get("label") == "HAS"
            and (
                head["text"].startswith("overexpressing ")
                or tail["text"].startswith("overexpressing ")
            )
        ):
            continue
        new_rel = dict(rel)
        new_rel.update({
            "head": head["text"],
            "head_start": head["start"],
            "head_end": head["end"],
            "head_type": head["label"],
            "tail": tail["text"],
            "tail_start": tail["start"],
            "tail_end": tail["end"],
            "tail_type": tail["label"],
        })
        rel_key = (
            new_rel["head_start"], new_rel["head_end"], new_rel["head_type"],
            new_rel["tail_start"], new_rel["tail_end"], new_rel["tail_type"],
            new_rel["label"],
        )
        if rel_key in seen_rel:
            continue
        seen_rel.add(rel_key)
        new_relations.append(new_rel)

    out = dict(record)
    out["entities"] = new_entities
    out["relations"] = new_relations
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--gold", default=None)
    ap.add_argument("--split_overexpressing", action="store_true")
    ap.add_argument("--expand_qtl", action="store_true")
    ap.add_argument("--expand_qtl_chromosome", action="store_true")
    ap.add_argument("--snp_list_chr", action="store_true")
    ap.add_argument("--add_gwas_bm", action="store_true")
    ap.add_argument("--drop_overexpressing_has", action="store_true")
    args = ap.parse_args()

    records = load_json(args.input)
    processed = [
        process_record(
            r,
            split_overexpressing=args.split_overexpressing,
            expand_qtl=args.expand_qtl,
            expand_qtl_chromosome=args.expand_qtl_chromosome,
            snp_list_chr=args.snp_list_chr,
            add_gwas_bm=args.add_gwas_bm,
            drop_overexpressing_has=args.drop_overexpressing_has,
        )
        for r in records
    ]
    dump_json(args.output, processed)
    print(f"[OK] wrote {args.output}")

    if args.gold:
        gold = load_json(args.gold)
        metrics = evaluate(gold[:len(processed)], processed)
        print_metrics(metrics, title="ERROR-GUIDED POSTPROCESS")


if __name__ == "__main__":
    main()
