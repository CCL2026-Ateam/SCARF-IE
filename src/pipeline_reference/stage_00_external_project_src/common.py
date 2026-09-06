"""Common utilities: data IO, label sets, scoring (official-equivalent)."""
import json
from pathlib import Path
from typing import Any, Dict, List

# ---- Label sets (from README / baseline) ----
ENTITY_LABELS = ["CROP", "VAR", "TRT", "GST", "GENE", "QTL", "MRK", "CHR", "BM", "CROSS", "ABS", "BIS"]
RELATION_LABELS = ["CON", "USE", "HAS", "AFF", "OCI", "LOI"]
ENTITY_LABEL_SET = set(ENTITY_LABELS)
RELATION_LABEL_SET = set(RELATION_LABELS)


def load_json(path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def append_jsonl(path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def load_jsonl(path) -> List[Dict[str, Any]]:
    path = Path(path)
    items = []
    if not path.exists():
        return items
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                continue
    return items


# ---- Scoring keys (MUST match baseline/evaluate_mgbie_qwen_lora.py) ----
def entity_key(e: Dict[str, Any]):
    return (e["start"], e["end"], e["label"])


def relation_key(r: Dict[str, Any]):
    return (
        r["head_start"], r["head_end"], r["head_type"],
        r["tail_start"], r["tail_end"], r["tail_type"],
        r["label"],
    )


def compute_prf(gold_set, pred_set) -> Dict[str, float]:
    tp = len(gold_set & pred_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f1}


def task_score(prf: Dict[str, float]) -> float:
    """Score_NER / Score_RE = 0.5*F1 + 0.25*P + 0.25*R."""
    return 0.5 * prf["f1"] + 0.25 * prf["precision"] + 0.25 * prf["recall"]


def evaluate(gold_records: List[Dict], pred_records: List[Dict]) -> Dict[str, Any]:
    """Compute the official metrics. Records are matched by index (must be aligned)."""
    assert len(gold_records) == len(pred_records), \
        f"length mismatch gold={len(gold_records)} pred={len(pred_records)}"
    g_ent, p_ent, g_rel, p_rel = set(), set(), set(), set()
    for i, (g, p) in enumerate(zip(gold_records, pred_records)):
        for e in g.get("entities", []):
            g_ent.add((i, *entity_key(e)))
        for e in p.get("entities", []):
            p_ent.add((i, *entity_key(e)))
        for r in g.get("relations", []):
            g_rel.add((i, *relation_key(r)))
        for r in p.get("relations", []):
            p_rel.add((i, *relation_key(r)))
    ent = compute_prf(g_ent, p_ent)
    rel = compute_prf(g_rel, p_rel)
    s_ner = task_score(ent)
    s_re = task_score(rel)
    total = 0.4 * s_ner + 0.6 * s_re
    return {
        "entity": ent,
        "relation": rel,
        "score_ner": s_ner,
        "score_re": s_re,
        "total_score": total,
    }


def print_metrics(m: Dict[str, Any], title: str = "") -> None:
    if title:
        print(f"===== {title} =====")
    e, r = m["entity"], m["relation"]
    print(f"  NER  P={e['precision']:.4f} R={e['recall']:.4f} F1={e['f1']:.4f} "
          f"(tp={e['tp']} fp={e['fp']} fn={e['fn']})  Score_NER={m['score_ner']:.4f}")
    print(f"  RE   P={r['precision']:.4f} R={r['recall']:.4f} F1={r['f1']:.4f} "
          f"(tp={r['tp']} fp={r['fp']} fn={r['fn']})  Score_RE ={m['score_re']:.4f}")
    print(f"  >>> TOTAL = 0.4*NER + 0.6*RE = {m['total_score']:.4f}")
