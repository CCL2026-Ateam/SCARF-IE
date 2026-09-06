"""Robust span alignment & post-processing.

The official scorer matches entities on (start, end, label) and relations on the
two endpoint spans + types. LLM-provided character offsets are unreliable, so we
re-derive offsets from the entity *text* by searching the source string.

Key finding from train.json (oracle analysis):
  - choosing the FIRST occurrence of each (text,label) plus any occurrence that is
    referenced by a relation gives the best official NER score (~0.96 oracle).
"""
from typing import Any, Dict, List, Tuple
from common import ENTITY_LABEL_SET, RELATION_LABEL_SET


_TOKEN_CHARS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_+-./")


def _needs_embedded_filter(needle: str) -> bool:
    """Short codes/numbers often match inside longer identifiers (e.g. BX in qBX7.1)."""
    if len(needle) <= 4:
        return True
    compact = needle.replace("-", "").replace(".", "")
    return bool(compact) and (compact.isupper() or compact.isdigit())


def _is_embedded_match(text: str, start: int, end: int) -> bool:
    left = start > 0 and text[start - 1] in _TOKEN_CHARS
    right = end < len(text) and text[end] in _TOKEN_CHARS
    return left or right


def _filter_embedded_spans(text: str, needle: str, spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not spans or not _needs_embedded_filter(needle):
        return spans
    clean = [sp for sp in spans if not _is_embedded_match(text, sp[0], sp[1])]
    return clean or spans


def find_all_spans(text: str, needle: str) -> List[Tuple[int, int]]:
    if not needle:
        return []
    spans = []
    start = 0
    while True:
        i = text.find(needle, start)
        if i < 0:
            return _filter_embedded_spans(text, needle, spans)
        spans.append((i, i + len(needle)))
        start = i + 1


def _choose_span(spans: List[Tuple[int, int]], preferred_start) -> Tuple[int, int]:
    """Pick the occurrence closest to the model-suggested start (fallback: first)."""
    if preferred_start is None:
        return spans[0]
    return min(spans, key=lambda sp: (abs(sp[0] - preferred_start), sp[0]))


def _to_int(v):
    try:
        return int(v)
    except Exception:
        return None


def align_entities(raw_entities: List[Dict[str, Any]], text: str):
    """Resolve each raw entity to a concrete (start,end,text,label).

    Returns (entities, lookup) where lookup maps (norm_text, label) -> list of
    candidate entity dicts, used by relation alignment.

    Strategy per unique (text,label):
      - always emit the FIRST occurrence in the source text
      - the model's suggested start is used to disambiguate which single occurrence
        to emit when the model clearly points at a specific one (kept minimal to
        protect precision). Additional occurrences referenced by relations are
        added later in `align_relations`.
    """
    # Collect requested (text,label) with the model's suggested starts.
    by_tl: Dict[Tuple[str, str], List] = {}
    order: List[Tuple[str, str]] = []
    for e in raw_entities or []:
        if not isinstance(e, dict):
            continue
        label = str(e.get("label", "")).strip()
        if label not in ENTITY_LABEL_SET:
            continue
        etext = str(e.get("text", "")).strip()
        if not etext:
            continue
        key = (etext, label)
        if key not in by_tl:
            by_tl[key] = []
            order.append(key)
        by_tl[key].append(_to_int(e.get("start")))

    entities = []
    seen = set()
    # lookup for relation resolution: (text,label) -> all spans found in source
    span_index: Dict[Tuple[str, str], List[Tuple[int, int]]] = {}
    for key in order:
        etext, label = key
        spans = find_all_spans(text, etext)
        if not spans:
            continue
        span_index[key] = spans
        # Emit the first occurrence, PLUS any occurrence the model's suggested
        # offset points at (recovers recall when gold annotated a later occurrence;
        # validated on dev: +0.006 total with no precision loss).
        chosen = {spans[0]}
        for sug in by_tl[key]:
            if sug is not None and sug >= 0:
                chosen.add(min(spans, key=lambda sp: (abs(sp[0] - sug), sp[0])))
        for s, en in chosen:
            k = (s, en, label)
            if k not in seen:
                seen.add(k)
                entities.append({"start": s, "end": en, "text": text[s:en], "label": label})
    return entities, span_index, seen


def align_relations(raw_relations, text, entities, span_index, seen_entity_keys):
    """Resolve relations; may add extra entity occurrences that relations reference.

    For each relation we resolve head/tail by (text,label). If a relation points at
    a non-first occurrence (via suggested offset), we add that occurrence as an
    entity too (matching the 'first+relref' oracle strategy).
    """
    relations = []
    seen_rel = set()
    extra_entities = []

    def candidate_spans(rtext, rtype):
        rtype = str(rtype).strip()
        rtext = str(rtext).strip()
        if rtype not in ENTITY_LABEL_SET or not rtext:
            return None, None
        spans = span_index.get((rtext, rtype))
        if not spans:
            spans = find_all_spans(text, rtext)
            if spans:
                span_index[(rtext, rtype)] = spans
            else:
                return None, None
        return spans, rtype

    def resolve_pair(h_spans, h_type, h_start, t_spans, t_type, t_start):
        """Jointly choose head & tail occurrences.

        If the model gave usable offsets, honor them (closest occurrence). Otherwise
        pick the (head, tail) occurrence pair that minimizes the textual gap between
        the two mentions — relation endpoints are almost always near each other.
        """
        # If offsets provided, resolve each independently by proximity to suggestion.
        h_has = h_start is not None and h_start >= 0
        t_has = t_start is not None and t_start >= 0
        if h_has or t_has:
            h = _choose_span(h_spans, h_start if h_has else None)
            t = _choose_span(t_spans, t_start if t_has else None)
            return h, t
        # No offsets: minimize gap between the two mentions.
        best = None
        best_gap = None
        for hs in h_spans:
            for ts in t_spans:
                if hs == ts:
                    continue
                # gap = distance between the nearest edges
                if hs[1] <= ts[0]:
                    gap = ts[0] - hs[1]
                elif ts[1] <= hs[0]:
                    gap = hs[0] - ts[1]
                else:
                    gap = 0  # overlap
                if best_gap is None or gap < best_gap or (gap == best_gap and hs[0] < best[0][0]):
                    best_gap = gap
                    best = (hs, ts)
        if best is None:
            return h_spans[0], (t_spans[0] if t_spans[0] != h_spans[0] else (t_spans[1] if len(t_spans) > 1 else t_spans[0]))
        return best[0], best[1]

    for r in raw_relations or []:
        if not isinstance(r, dict):
            continue
        label = str(r.get("label", "")).strip()
        if label not in RELATION_LABEL_SET:
            continue
        h_spans, h_type = candidate_spans(r.get("head"), r.get("head_type"))
        t_spans, t_type = candidate_spans(r.get("tail"), r.get("tail_type"))
        if not h_spans or not t_spans:
            continue
        h_sp, t_sp = resolve_pair(
            h_spans, h_type, _to_int(r.get("head_start")),
            t_spans, t_type, _to_int(r.get("tail_start")),
        )
        head = (h_sp[0], h_sp[1], h_type)
        tail = (t_sp[0], t_sp[1], t_type)
        if head is None or tail is None:
            continue
        if (head[0], head[1]) == (tail[0], tail[1]):
            continue
        # ensure endpoint entities exist
        for (s, en, lab) in (head, tail):
            k = (s, en, lab)
            if k not in seen_entity_keys:
                seen_entity_keys.add(k)
                extra_entities.append({"start": s, "end": en, "text": text[s:en], "label": lab})
        item = {
            "head": text[head[0]:head[1]], "head_start": head[0], "head_end": head[1], "head_type": head[2],
            "tail": text[tail[0]:tail[1]], "tail_start": tail[0], "tail_end": tail[1], "tail_type": tail[2],
            "label": label,
        }
        rk = (head[0], head[1], head[2], tail[0], tail[1], tail[2], label)
        if rk in seen_rel:
            continue
        seen_rel.add(rk)
        relations.append(item)

    return relations, extra_entities


def postprocess(raw: Dict[str, Any], text: str) -> Dict[str, Any]:
    """Full pipeline: align entities then relations, return clean record."""
    raw_entities = raw.get("entities", []) if isinstance(raw, dict) else []
    raw_relations = raw.get("relations", []) if isinstance(raw, dict) else []

    entities, span_index, seen = align_entities(raw_entities, text)
    relations, extra = align_relations(raw_relations, text, entities, span_index, seen)
    entities.extend(extra)

    entities.sort(key=lambda e: (e["start"], e["end"], e["label"]))
    relations.sort(key=lambda r: (r["head_start"], r["head_end"], r["tail_start"], r["tail_end"], r["label"]))
    return {"text": text, "entities": entities, "relations": relations}
