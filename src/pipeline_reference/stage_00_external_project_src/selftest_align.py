"""Offline self-test: feed gold entities/relations (with offsets stripped) through
the alignment postprocessor and score. This verifies the align+scoring pipeline
WITHOUT spending any API tokens, and reveals the achievable ceiling of our
span-alignment strategy when the model's text/label predictions are perfect."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_json, evaluate, print_metrics  # noqa: E402
from align import postprocess  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def strip_offsets(rec):
    """Simulate a 'perfect-text, wrong-offset' model output."""
    ents = [{"start": -1, "end": -1, "text": e["text"], "label": e["label"]}
            for e in rec["entities"]]
    rels = [{"head": r["head"], "head_type": r["head_type"],
             "tail": r["tail"], "tail_type": r["tail_type"], "label": r["label"]}
            for r in rec["relations"]]
    return {"entities": ents, "relations": rels}


def main(path):
    gold = load_json(path)
    preds = [postprocess(strip_offsets(r), r["text"]) for r in gold]
    m = evaluate(gold, preds)
    print_metrics(m, title=f"ORACLE-TEXT CEILING (offsets stripped) on {Path(path).name}")


if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "solution" / "data" / "dev.json")
    main(p)
