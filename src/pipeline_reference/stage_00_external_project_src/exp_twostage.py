"""Experiment: two-stage relation extraction.
Stage-1 entities come from a given prediction file (ideally the ensemble entity set).
Stage-2 re-extracts relations among those fixed entities with a chosen model.
Evaluates RE against gold."""
import sys
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import client  # noqa
from common import load_json, dump_json, evaluate, print_metrics  # noqa
from prompts_rel import build_relation_prompt, relations_from_ids, REL_SYSTEM  # noqa
from retriever import ExemplarRetriever  # noqa
import run as R  # noqa

ROOT = Path(__file__).resolve().parents[2]


def stage2(rec_text, entities, model, retriever, kshot, max_tokens=2048):
    exemplars = retriever.retrieve(rec_text, k=kshot) if (retriever and kshot > 0) else None
    prompt = build_relation_prompt(rec_text, entities, exemplars)
    cm = f"REL::{model}"
    raw = R.get_cached(cm, prompt)
    if raw is None:
        raw = client.chat(prompt, model=model, system=REL_SYSTEM, use_json=True, max_tokens=max_tokens)
        R.set_cached(cm, prompt, raw)
    parsed = R.parse_json_lenient(raw) or {"relations": []}
    return relations_from_ids(parsed, entities, rec_text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--entity_pred", required=True, help="prediction file providing stage-1 entities")
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default="main")
    ap.add_argument("--kshot", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--concurrency", type=int, default=16)
    args = ap.parse_args()

    model = R.resolve_model(args.model)
    gold = load_json(args.gold)
    ent_src = load_json(args.entity_pred)
    if args.limit:
        gold = gold[:args.limit]
        ent_src = ent_src[:args.limit]
    pool = load_json(ROOT / "solution" / "data" / "pool.json")
    retriever = ExemplarRetriever(pool) if args.kshot > 0 else None

    results = [None] * len(gold)
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {ex.submit(stage2, gold[i]["text"], ent_src[i]["entities"], model, retriever, args.kshot): i
                for i in range(len(gold))}
        done = 0
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                rels = fut.result()
            except Exception as e:
                print(f"[WARN] {i}: {e}")
                rels = []
            results[i] = {"text": gold[i]["text"], "entities": ent_src[i]["entities"], "relations": rels}
            done += 1
            if done % 20 == 0 or done == len(gold):
                print(f"  progress {done}/{len(gold)}", flush=True)

    dump_json(args.output, results)
    m = evaluate(gold, results)
    print_metrics(m, title=f"TWO-STAGE RE  model={model} kshot={args.kshot} entities_from={Path(args.entity_pred).name}")


if __name__ == "__main__":
    main()
