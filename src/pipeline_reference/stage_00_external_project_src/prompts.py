"""Prompt construction for entity + relation extraction (Track-A, no fine-tune)."""
import json
from typing import Any, Dict, List

SYSTEM_PROMPT = (
    "You are an expert information-extraction system for cereal/millet breeding "
    "literature (foxtail millet, sorghum, oat, buckwheat, barley, broomcorn millet, "
    "various food legumes, etc.). You extract named entities and the semantic "
    "relations between them, strictly grounded in the given English text."
)

# Compact label guide. Examples chosen to match dataset distribution.
ENTITY_GUIDE = """ENTITY TYPES (12). Extract spans EXACTLY as they appear in the text (do not translate, normalize, expand abbreviations, or merge):
- CROP : crop species / class / genus. e.g. sorghum, foxtail millet, Tartary buckwheat, Setaria italica, oat, barley
- VAR  : a specific cultivar / variety / accession / line / genotype / mutant / transgenic material / germplasm name or code. e.g. Jingu 21, An04-4783, LN-sensitive genotype, near-isogenic lines, transgenic lines, diploid sorghum inbreds
- TRT  : a phenotypic / agronomic / quality / resistance / physiological trait or measured indicator. e.g. plant height, seed yield, kernel weight, drought tolerance, chlorophyll content, milling traits
- GST  : a growth / developmental / measurement stage. e.g. seedling stage, maturity, grain-filling stage, heading stage
- GENE : a gene, candidate gene, gene-family member, gene symbol. e.g. Waxy, Dw6, SiCST1, SbSNAC1, FtMYB10
- QTL  : a QTL, locus, mapping interval, MQTL, resistance locus. e.g. qPH3.1, drought tolerance QTL, QTL for SDW, 6H QTL
- MRK  : a molecular marker (SSR/SNP/InDel/KASP/DArT...). e.g. GBM1498, KASP_Chr2_15Mb, SB3344
- CHR  : a chromosome / linkage group / chromosome segment. e.g. Chr1, chromosome 2, chromosome 2H, LG10, SBI-03, 6H
- BM   : a breeding / screening / detection / omics / mapping method or technique. e.g. hybridization, QTL mapping, RNA-Seq, GWAS, positional cloning, marker-assisted selection
- CROSS: a parent/cross combination expression. e.g. A x B, Prisma x Apex, Yugu 5 x Jigu 31
- ABS  : an abiotic stress / treatment / non-living adverse condition. e.g. drought stress, salt stress, heat stress, low nitrogen (LN), PEG treatment
- BIS  : a biotic stress / pest / pathogen / disease. e.g. stem rust, aphids, southern root-knot nematodes (RKN), stalk rot"""

RELATION_GUIDE = """RELATION TYPES (6 coarse classes). A relation is directional head -> tail. Only extract a relation when the text gives DIRECT evidence (a verb/phrase linking the two); do NOT infer from mere co-occurrence:
- CON (contains / belongs-to / is-a / alias / abbreviation / synonym / apposition). Typical: CROP->VAR (a variety belongs to a crop), GENE<->GENE alias, full name<->abbreviation, scientific<->common name. e.g. "barley cultivar 'Tadmor'" => barley CON Tadmor
- USE (uses / employs / applies / via / by / through a method/technique/material). Typical: VAR->BM, BM->GENE/QTL/MRK, applied-via. e.g. "identified via positional cloning" => positional cloning USE <target>
- HAS (an object has / shows / exhibits / possesses a trait). Typical: VAR->TRT, CROP->TRT. Triggers: showed, had, exhibited, possesses, is tolerant/resistant, higher/lower <trait>. e.g. "sorghum inbreds have higher kernel weight" => inbreds HAS kernel weight
- AFF (affects / regulates / promotes / suppresses / induces / responds / sensitive / tolerant / resistant — a directional effect). Typical: ABS->TRT, GENE->TRT, BIS->TRT, ABS->GENE, QTL->TRT. Triggers: affect, regulate, enhance, improve, reduce, increase, decrease, inhibit, induce, responsive to, sensitive to, tolerant to, resistant to, damaged by, infected by. e.g. "LN improved root growth" => LN AFF root growth
- OCI (occurs-in / measured-at / investigated-at / treated-at a growth stage). Typical: TRT->GST, ABS->GST, BIS->GST. e.g. "kernel weight measured at maturity" => kernel weight OCI maturity
- LOI (located / mapped / linked / associated / candidate / co-localized — molecular-evidence positioning). Typical: QTL->CHR, QTL->TRT, MRK->QTL, GENE->CHR, MRK->CHR, GENE->TRT. Triggers: located on, mapped to, linked to, associated with, flanked by, candidate for, co-localized with, on chromosome. e.g. "QTL on chromosome 2" => QTL LOI chromosome 2"""

OUTPUT_SPEC = """OUTPUT FORMAT — output ONLY a single JSON object, no markdown, no commentary:
{
  "entities": [{"start": <int>, "end": <int>, "text": "<exact span>", "label": "<TYPE>"}],
  "relations": [{"head": "<exact span>", "head_type": "<TYPE>", "tail": "<exact span>", "tail_type": "<TYPE>", "label": "<REL>"}]
}
RULES:
1. "text" / "head" / "tail" MUST be copied character-for-character from the input text (a contiguous substring). Do not paraphrase.
2. start/end are character offsets; if unsure, give your best estimate — they will be re-aligned automatically, so prioritize EXACT span text over exact offsets.
3. Every head/tail in relations MUST also appear as an entity (same text & type).
4. If a span appears multiple times, you may extract the most relevant occurrence; offsets need not be perfect.
5. Extract relations aggressively where evidence exists — relations are weighted heavily. But never fabricate relations without textual support.
6. If no entities/relations, output empty arrays."""

# Boundary & selection conventions derived from gold error analysis. Toggleable.
CONVENTIONS = """SPAN-BOUNDARY CONVENTIONS (the gold standard annotates spans a specific way — follow these to match exact boundaries):
- Extract the MINIMAL but COMPLETE term. Drop leading descriptive/quantitative modifiers that are not part of the term name: "abnormal X"->"X", "increased/reduced/higher/lower X content"->"X content" only if the change-word is not intrinsic; "major QTL"/"minor QTL"->"QTL", "14 QTL hotspot regions"->"QTL hotspot regions", "domesticated chickpea"->"chickpea", "400 mM salinity"->"salinity".
- Keep an entity together with its parenthetical abbreviation as ONE span when written as "full name (ABBR)": e.g. "spike length (SL)" is a single TRT span (do NOT also emit "SL" and "spike length" separately). When only the abbreviation appears alone elsewhere, that standalone occurrence is its own span.
- Do NOT merge two coordinated items into one span: "chromosomes 3 and 7" -> annotate "chromosomes 3" (and "7"/"chromosome 7" if it is a real chromosome mention), not the whole coordinated phrase. "qPH7.1/qBX7.1" are two separate QTLs.
- A leading category word can be part of the name in this domain: "Gene Si1g06530", "QTL QSc/Sl.cib-7H" — when the text writes the type word immediately before the identifier, include it.
- Be conservative with generic/abstract mentions: only extract a term as an entity when it denotes a concrete object in context. Plural generic mentions (e.g. "markers", "traits", "genes") are usually NOT annotated unless they are the specific subject; specific plurals like "QTLs", "SNPs" may be annotated when they are the focus.
- Precision matters as much as recall (scoring weights F1 0.5 + P 0.25 + R 0.25). Prefer not to emit a span you are unsure about."""


def _fmt_example(rec: Dict[str, Any]) -> str:
    ents = [{"start": e["start"], "end": e["end"], "text": e["text"], "label": e["label"]}
            for e in rec["entities"]]
    rels = [{"head": r["head"], "head_type": r["head_type"],
             "tail": r["tail"], "tail_type": r["tail_type"], "label": r["label"]}
            for r in rec["relations"]]
    out = {"entities": ents, "relations": rels}
    return (f'TEXT:\n{rec["text"]}\n'
            f'OUTPUT:\n{json.dumps(out, ensure_ascii=False)}')


def build_prompt(text: str, exemplars: List[Dict[str, Any]] = None, conventions: bool = False) -> str:
    parts = [ENTITY_GUIDE, "", RELATION_GUIDE, ""]
    if conventions:
        parts += [CONVENTIONS, ""]
    parts += [OUTPUT_SPEC, ""]
    if exemplars:
        parts.append("=== EXAMPLES ===")
        for ex in exemplars:
            parts.append(_fmt_example(ex))
            parts.append("")
    parts.append("=== NOW EXTRACT FROM THIS TEXT ===")
    parts.append(f"TEXT:\n{text}")
    parts.append("OUTPUT:")
    return "\n".join(parts)
