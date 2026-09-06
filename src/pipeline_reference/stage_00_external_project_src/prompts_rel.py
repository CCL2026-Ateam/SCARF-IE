"""Stage-2 relation extraction: given a FIXED entity list, ask the model to
extract only relations among those entities. Used to boost RE recall/precision
by decoupling it from entity extraction."""
import json
from typing import Any, Dict, List
from prompts import RELATION_GUIDE

REL_SYSTEM = (
    "You are an expert relation-extraction system for cereal/millet breeding "
    "literature. Given a text and a fixed list of entities, you extract the "
    "directional semantic relations that hold between those entities, strictly "
    "grounded in the text."
)

REL_OUTPUT_SPEC = """OUTPUT FORMAT — output ONLY a JSON object:
{"relations": [{"head_id": <int>, "tail_id": <int>, "label": "<REL>"}]}
RULES:
1. head_id / tail_id refer to the "id" field of the provided entities.
2. label is one of: CON, USE, HAS, AFF, OCI, LOI.
3. Only output a relation when the text gives DIRECT evidence linking the two entities (a verb/phrase). Do NOT infer from mere co-occurrence.
4. Relations are directional (head -> tail). Choose the correct direction per the type definitions.
5. The same entity pair may have at most one relation label unless the text clearly supports more. Output an empty array if there are no well-supported relations.
6. Relations are weighted heavily in scoring — extract every well-supported relation, but do not fabricate."""


def build_relation_prompt(text: str, entities: List[Dict[str, Any]], exemplars=None) -> str:
    ent_lines = []
    for i, e in enumerate(entities):
        ent_lines.append(f'  {{"id": {i}, "text": {json.dumps(e["text"], ensure_ascii=False)}, "type": "{e["label"]}"}}')
    ent_block = "[\n" + ",\n".join(ent_lines) + "\n]"
    parts = [RELATION_GUIDE, "", REL_OUTPUT_SPEC, ""]
    if exemplars:
        parts.append("=== EXAMPLES ===")
        for ex in exemplars:
            parts.append(_fmt_rel_example(ex))
            parts.append("")
    parts.append("=== NOW EXTRACT RELATIONS ===")
    parts.append(f"TEXT:\n{text}")
    parts.append(f"ENTITIES:\n{ent_block}")
    parts.append("OUTPUT:")
    return "\n".join(parts)


def _fmt_rel_example(rec: Dict[str, Any]) -> str:
    # Build entity id map from the gold record
    ents = rec["entities"]
    key2id = {(e["start"], e["end"], e["label"]): i for i, e in enumerate(ents)}
    ent_lines = [f'  {{"id": {i}, "text": {json.dumps(e["text"], ensure_ascii=False)}, "type": "{e["label"]}"}}'
                 for i, e in enumerate(ents)]
    rels = []
    for r in rec["relations"]:
        hid = key2id.get((r["head_start"], r["head_end"], r["head_type"]))
        tid = key2id.get((r["tail_start"], r["tail_end"], r["tail_type"]))
        if hid is not None and tid is not None:
            rels.append({"head_id": hid, "tail_id": tid, "label": r["label"]})
    out = {"relations": rels}
    return (f'TEXT:\n{rec["text"]}\n'
            f'ENTITIES:\n[\n' + ",\n".join(ent_lines) + "\n]\n"
            f'OUTPUT:\n{json.dumps(out, ensure_ascii=False)}')


def relations_from_ids(parsed: Dict[str, Any], entities: List[Dict[str, Any]], text: str) -> List[Dict[str, Any]]:
    """Convert id-based relation output to full relation dicts using the entity list."""
    out = []
    seen = set()
    raw = parsed.get("relations", []) if isinstance(parsed, dict) else []
    n = len(entities)
    for r in raw:
        if not isinstance(r, dict):
            continue
        try:
            hid = int(r.get("head_id"))
            tid = int(r.get("tail_id"))
        except Exception:
            continue
        label = str(r.get("label", "")).strip()
        from common import RELATION_LABEL_SET
        if label not in RELATION_LABEL_SET:
            continue
        if not (0 <= hid < n and 0 <= tid < n) or hid == tid:
            continue
        h = entities[hid]
        t = entities[tid]
        item = {
            "head": h["text"], "head_start": h["start"], "head_end": h["end"], "head_type": h["label"],
            "tail": t["text"], "tail_start": t["start"], "tail_end": t["end"], "tail_type": t["label"],
            "label": label,
        }
        k = (h["start"], h["end"], h["label"], t["start"], t["end"], t["label"], label)
        if k in seen:
            continue
        seen.add(k)
        out.append(item)
    out.sort(key=lambda r: (r["head_start"], r["head_end"], r["tail_start"], r["tail_end"], r["label"]))
    return out
