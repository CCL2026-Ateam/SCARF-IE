"""Multi-member ensemble with optional AI arbitration for disagreement cases.

The pipeline:
  1. Run multiple extraction members, e.g. main/secondary/cheap and convention variants.
  2. Build a normal vote-threshold ensemble.
  3. For non-unanimous candidates near the decision boundary, ask a judge model
     to keep/reject them using retrieved training examples as evidence.

The judge does not invent new entities/relations. It only discusses candidates
proposed by at least one member and writes a separate discussion log with
training-example-based reasons.
"""
import argparse
import json
import re
import subprocess
import sys
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import client  # noqa: E402
from build_submission import to_submission  # noqa: E402
from common import dump_json, evaluate, load_json, print_metrics  # noqa: E402
from ensemble import ensemble, ekey, rkey  # noqa: E402
from run import parse_json_lenient, resolve_model  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
POOL_PATH = ROOT / "data" / "pool.json"
PY = sys.executable


BEST_ENTITY_THRESHOLDS = {
    "ABS": 2,
    "BIS": 2,
    "CHR": 2,
    "CROP": 2,
    "CROSS": 4,
    "GST": 2,
    "MRK": 4,
    "QTL": 2,
}

BEST_RELATION_THRESHOLDS = {
    "AFF": 3,
    "CON": 1,
    "HAS": 3,
    "LOI": 2,
    "OCI": 2,
    "USE": 5,
}


DEFAULT_MEMBERS = [
    ("secondary", False, "secondary_k4"),
    ("main", False, "main_k4"),
    ("cheap", False, "cheap_k4"),
    ("secondary", True, "secondary_k4_conv"),
    ("main", True, "main_k4_conv"),
    ("cheap", True, "cheap_k4_conv"),
]


TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-]*")


def tokenize(text):
    return set(t.lower() for t in TOKEN_RE.findall(text))


def parse_member(item):
    parts = item.split(":")
    if len(parts) != 3:
        raise SystemExit(
            f"bad --member {item!r}; expected MODEL:CONVENTIONS:SUFFIX, "
            "for example deepseek-chat:false:deepseek_k4"
        )
    model, conventions, suffix = parts
    conv = conventions.strip().lower()
    if conv not in {"0", "1", "false", "true", "no", "yes"}:
        raise SystemExit(f"bad conventions value in --member {item!r}")
    return model.strip(), conv in {"1", "true", "yes"}, suffix.strip()


def run_member(input_path, output_path, model, conventions, kshot, limit,
               record_concurrency, max_tokens, temperature, no_cache,
               retriever, retriever_model, retriever_cache):
    cmd = [
        PY, str(Path(__file__).parent / "run.py"),
        "--input", str(input_path),
        "--output", str(output_path),
        "--model", model,
        "--kshot", str(kshot),
        "--retriever", retriever,
        "--concurrency", str(record_concurrency),
        "--max_tokens", str(max_tokens),
    ]
    if retriever_model:
        cmd += ["--retriever_model", str(retriever_model)]
    if retriever_cache:
        cmd += ["--retriever_cache", str(retriever_cache)]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    if conventions:
        cmd.append("--conventions")
    if temperature is not None:
        cmd += ["--temperature", str(temperature)]
    if no_cache:
        cmd.append("--no_cache")

    print("[RUN]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    return output_path


def format_entity(e):
    return {
        "start": e["start"],
        "end": e["end"],
        "text": e["text"],
        "label": e["label"],
    }


def format_relation(r):
    return {
        "head": r["head"],
        "head_start": r["head_start"],
        "head_end": r["head_end"],
        "head_type": r["head_type"],
        "tail": r["tail"],
        "tail_start": r["tail_start"],
        "tail_end": r["tail_end"],
        "tail_type": r["tail_type"],
        "label": r["label"],
    }


def retrieve_examples(pool, text, k):
    if not pool or k <= 0:
        return []
    q = tokenize(text)
    scored = []
    for i, rec in enumerate(pool):
        pt = tokenize(rec["text"])
        if not pt:
            continue
        inter = len(q & pt)
        if inter == 0:
            continue
        score = inter / len(q | pt)
        if rec.get("relations"):
            score += 0.03
        scored.append((score, i))
    scored.sort(reverse=True)
    chosen = []
    used = set()
    for _score, i in scored:
        if len(chosen) >= k:
            break
        if i in used:
            continue
        used.add(i)
        rec = pool[i]
        chosen.append({
            "example_id": i,
            "text": rec["text"],
            "entities": [format_entity(e) for e in rec.get("entities", [])],
            "relations": [format_relation(r) for r in rec.get("relations", [])],
        })
    return chosen


def collect_votes(pred_lists, member_names, idx):
    ent_votes = Counter()
    rel_votes = Counter()
    ent_obj = {}
    rel_obj = {}
    ent_voters = defaultdict(list)
    rel_voters = defaultdict(list)
    for name, preds in zip(member_names, pred_lists):
        rec = preds[idx]
        seen_e = set()
        for e in rec.get("entities", []):
            k = ekey(e)
            if k in seen_e:
                continue
            seen_e.add(k)
            ent_votes[k] += 1
            ent_obj[k] = e
            ent_voters[k].append(name)
        seen_r = set()
        for r in rec.get("relations", []):
            k = rkey(r)
            if k in seen_r:
                continue
            seen_r.add(k)
            rel_votes[k] += 1
            rel_obj[k] = r
            rel_voters[k].append(name)
    return ent_votes, rel_votes, ent_obj, rel_obj, ent_voters, rel_voters


def threshold_for_entity(k, default_vote):
    return BEST_ENTITY_THRESHOLDS.get(k[2], default_vote)


def threshold_for_relation(k, default_vote):
    return BEST_RELATION_THRESHOLDS.get(k[6], default_vote)


def candidate_priority(item):
    # Prefer relations because RE is 60% of the score; then higher vote count.
    typ = item["type"]
    votes = item["votes"]
    threshold = item["threshold"]
    near = -abs(votes - threshold)
    return (1 if typ == "relation" else 0, near, votes)


def build_candidates(pred_lists, member_names, base_rec, idx, default_ent_vote,
                     default_rel_vote, min_votes, max_candidates):
    ent_votes, rel_votes, ent_obj, rel_obj, ent_voters, rel_voters = collect_votes(pred_lists, member_names, idx)
    base_entities = {ekey(e) for e in base_rec.get("entities", [])}
    base_relations = {rkey(r) for r in base_rec.get("relations", [])}
    n = len(member_names)
    items = []

    for k, votes in ent_votes.items():
        if votes < min_votes or votes == n:
            continue
        threshold = threshold_for_entity(k, default_ent_vote)
        e = ent_obj[k]
        items.append({
            "id": f"E{len(items)}",
            "type": "entity",
            "votes": votes,
            "threshold": threshold,
            "default_keep": k in base_entities,
            "voters": ent_voters[k],
            "object": format_entity(e),
            "_key": k,
        })

    for k, votes in rel_votes.items():
        if votes < min_votes or votes == n:
            continue
        threshold = threshold_for_relation(k, default_rel_vote)
        r = rel_obj[k]
        items.append({
            "id": f"R{len(items)}",
            "type": "relation",
            "votes": votes,
            "threshold": threshold,
            "default_keep": k in base_relations,
            "voters": rel_voters[k],
            "object": format_relation(r),
            "_key": k,
        })

    items.sort(key=candidate_priority, reverse=True)
    return items[:max_candidates]


def build_judge_prompt(text, member_names, candidates, examples):
    public_candidates = []
    for c in candidates:
        public_candidates.append({
            "id": c["id"],
            "type": c["type"],
            "votes": c["votes"],
            "threshold": c["threshold"],
            "default_keep": c["default_keep"],
            "voters": c["voters"],
            "object": c["object"],
        })

    return f"""You are an adjudicator for an information extraction ensemble.

Task:
Decide whether each disputed candidate should be kept. Use ONLY the current text,
the candidate vote pattern, and the retrieved labeled training examples.
Do not invent new entities or relations.

Decision principles:
- Prefer exact span boundaries and labels matching the training examples.
- A relation should be kept only when the text gives direct evidence.
- If a candidate conflicts with the training examples, reject it even if it has votes.
- If a low-vote candidate is strongly supported by text and examples, keep it.
- Reasons MUST cite one or more training example IDs when useful.

Members:
{json.dumps(member_names, ensure_ascii=False)}

Current text:
{text}

Retrieved labeled training examples:
{json.dumps(examples, ensure_ascii=False)}

Disputed candidates:
{json.dumps(public_candidates, ensure_ascii=False)}

Return ONLY JSON:
{{
  "decisions": [
    {{
      "id": "E0 or R0",
      "keep": true,
      "reason": "short reason citing example_id(s) and text evidence"
    }}
  ]
}}
"""


def judge_one_record(idx, text, member_names, candidates, examples, judge_model, temperature, max_tokens):
    prompt = build_judge_prompt(text, member_names, candidates, examples)
    raw = client.chat(
        prompt,
        model=resolve_model(judge_model),
        system="You are a careful IE ensemble adjudicator. Return strict JSON only.",
        use_json=True,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    parsed = parse_json_lenient(raw) or {}
    decisions = parsed.get("decisions", [])
    by_id = {}
    for d in decisions:
        if isinstance(d, dict) and "id" in d and "keep" in d:
            by_id[str(d["id"])] = {
                "keep": bool(d["keep"]),
                "reason": str(d.get("reason", ""))[:1000],
            }
    return {"record_index": idx, "decisions": by_id, "raw": raw[:4000]}


def apply_decisions(base_rec, candidates, discussion):
    ents = {ekey(e): e for e in base_rec.get("entities", [])}
    rels = {rkey(r): r for r in base_rec.get("relations", [])}
    decisions = discussion.get("decisions", {})
    for c in candidates:
        decision = decisions.get(c["id"])
        if decision is None:
            continue
        keep = decision["keep"]
        if c["type"] == "entity":
            if keep:
                ents[c["_key"]] = c["object"]
            else:
                ents.pop(c["_key"], None)
        else:
            if keep:
                rels[c["_key"]] = c["object"]
            else:
                rels.pop(c["_key"], None)

    kept_entity_keys = set(ents)
    clean_rels = {}
    for k, r in rels.items():
        head = (r["head_start"], r["head_end"], r["head_type"])
        tail = (r["tail_start"], r["tail_end"], r["tail_type"])
        if head in kept_entity_keys and tail in kept_entity_keys:
            clean_rels[k] = r

    out = dict(base_rec)
    out["entities"] = sorted(ents.values(), key=lambda e: (e["start"], e["end"], e["label"]))
    out["relations"] = sorted(clean_rels.values(), key=lambda r: (r["head_start"], r["head_end"], r["tail_start"], r["tail_end"], r["label"]))
    return out


def discussion_ensemble(pred_lists, member_names, base_records, input_records, pool, args):
    final = list(base_records)
    jobs = []
    meta = {}

    for idx, base_rec in enumerate(base_records):
        text = base_rec["text"]
        candidates = build_candidates(
            pred_lists,
            member_names,
            base_rec,
            idx,
            args.vote_min,
            args.rel_vote_min,
            args.judge_min_votes,
            args.judge_max_candidates_per_record,
        )
        if not candidates:
            continue
        if args.judge_max_records is not None and len(jobs) >= args.judge_max_records:
            break
        examples = retrieve_examples(pool, text, args.example_k)
        meta[idx] = {"candidates": candidates, "examples": examples}
        jobs.append(idx)

    print(f"[JUDGE] records_with_disputes={len(jobs)} model={resolve_model(args.judge_model)}", flush=True)
    discussions = []
    with ThreadPoolExecutor(max_workers=args.judge_concurrency) as ex:
        futs = {
            ex.submit(
                judge_one_record,
                idx,
                final[idx]["text"],
                member_names,
                meta[idx]["candidates"],
                meta[idx]["examples"],
                args.judge_model,
                args.judge_temperature,
                args.judge_max_tokens,
            ): idx
            for idx in jobs
        }
        done = 0
        for fut in as_completed(futs):
            idx = futs[fut]
            try:
                discussion = fut.result()
                if args.judge_apply:
                    final[idx] = apply_decisions(final[idx], meta[idx]["candidates"], discussion)
            except Exception as exc:
                discussion = {"record_index": idx, "error": str(exc), "decisions": {}}
            discussions.append({
                "record_index": idx,
                "text": final[idx]["text"],
                "candidates": [
                    {k: v for k, v in c.items() if not k.startswith("_")}
                    for c in meta[idx]["candidates"]
                ],
                "examples": meta[idx]["examples"],
                "discussion": discussion,
            })
            done += 1
            if done % 20 == 0 or done == len(jobs):
                print(f"  judge progress {done}/{len(jobs)}", flush=True)
    discussions.sort(key=lambda x: x["record_index"])
    return final, discussions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="dataset/test_A.json")
    ap.add_argument("--tag", default="discussion_testA")
    ap.add_argument("--kshot", type=int, default=4)
    ap.add_argument("--retriever", choices=["lexical", "semantic"], default="lexical")
    ap.add_argument("--retriever_model", default=None)
    ap.add_argument("--retriever_cache", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--record_concurrency", type=int, default=8)
    ap.add_argument("--member_concurrency", type=int, default=3)
    ap.add_argument("--max_tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--no_cache", action="store_true")
    ap.add_argument("--skip_run", action="store_true")
    ap.add_argument("--member", action="append", default=[],
                    help="MODEL:CONVENTIONS:SUFFIX; repeat to override defaults")
    ap.add_argument("--vote_min", type=int, default=3)
    ap.add_argument("--rel_vote_min", type=int, default=2)
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--judge_apply", action="store_true",
                    help="apply judge decisions to final predictions; default only writes discussion reasons")
    ap.add_argument("--judge_model", default="main")
    ap.add_argument("--judge_temperature", type=float, default=0.0)
    ap.add_argument("--judge_max_tokens", type=int, default=4096)
    ap.add_argument("--judge_concurrency", type=int, default=4)
    ap.add_argument("--judge_min_votes", type=int, default=1)
    ap.add_argument("--judge_max_records", type=int, default=None)
    ap.add_argument("--judge_max_candidates_per_record", type=int, default=10)
    ap.add_argument("--example_k", type=int, default=3)
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--ensemble_out", default=None)
    ap.add_argument("--discussion_out", default=None)
    ap.add_argument("--submit_out", default=None)
    ap.add_argument("--zip_out", default=None)
    args = ap.parse_args()

    input_path = Path(args.input)
    members = [parse_member(item) for item in args.member] if args.member else DEFAULT_MEMBERS
    member_names = [suffix for _model, _conv, suffix in members]
    OUT.mkdir(parents=True, exist_ok=True)

    jobs = []
    member_outputs = []
    for model, conventions, suffix in members:
        out_path = OUT / f"{args.tag}_{suffix}.json"
        member_outputs.append(out_path)
        if args.skip_run and out_path.exists():
            print(f"[SKIP] existing member output: {out_path}")
            continue
        jobs.append((model, conventions, suffix, out_path))

    if jobs:
        with ThreadPoolExecutor(max_workers=args.member_concurrency) as ex:
            futs = {
                ex.submit(
                    run_member,
                    input_path,
                    out_path,
                    model,
                    conventions,
                    args.kshot,
                    args.limit,
                    args.record_concurrency,
                    args.max_tokens,
                    args.temperature,
                    args.no_cache,
                    args.retriever,
                    args.retriever_model,
                    args.retriever_cache,
                ): suffix
                for model, conventions, suffix, out_path in jobs
            }
            for fut in as_completed(futs):
                suffix = futs[fut]
                try:
                    fut.result()
                    print(f"[OK] member finished: {suffix}", flush=True)
                except Exception as exc:
                    raise SystemExit(f"[FAIL] member {suffix} failed: {exc}") from exc

    preds = [load_json(path) for path in member_outputs]
    lengths = {len(records) for records in preds}
    if len(lengths) != 1:
        raise SystemExit(f"member output lengths differ: {sorted(lengths)}")

    combined = ensemble(
        preds,
        vote_min=args.vote_min,
        rel_vote_min=args.rel_vote_min,
        rel_label_thresholds=BEST_RELATION_THRESHOLDS,
        ent_label_thresholds=BEST_ENTITY_THRESHOLDS,
    )

    discussions = []
    if args.judge:
        pool = load_json(POOL_PATH) if POOL_PATH.exists() else []
        input_records = load_json(input_path)
        if args.limit is not None:
            input_records = input_records[:args.limit]
        combined, discussions = discussion_ensemble(preds, member_names, combined, input_records, pool, args)

    ensemble_out = Path(args.ensemble_out) if args.ensemble_out else OUT / f"{args.tag}_discussion_ensemble.json"
    discussion_out = Path(args.discussion_out) if args.discussion_out else OUT / f"{args.tag}_discussions.json"
    submit_out = Path(args.submit_out) if args.submit_out else OUT / f"{args.tag}_submit.json"
    zip_out = Path(args.zip_out) if args.zip_out else OUT / f"{args.tag}_submit.zip"

    dump_json(ensemble_out, combined)
    print(f"[OK] ensemble -> {ensemble_out}")
    if args.judge:
        dump_json(discussion_out, discussions)
        print(f"[OK] discussions -> {discussion_out}")

    if args.eval:
        gold = load_json(input_path)
        if args.limit is not None:
            gold = gold[:args.limit]
        metrics = evaluate(gold, combined)
        print_metrics(metrics, title=f"{args.tag} DISCUSSION ENSEMBLE")
        dump_json(ensemble_out.with_name(ensemble_out.stem + "_metrics.json"), metrics)

    submission = to_submission(combined)
    dump_json(submit_out, submission)
    with zipfile.ZipFile(zip_out, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(submit_out, arcname="submit.json")
    print(f"[OK] submit zip -> {zip_out}")
    print(f"[OK] records={len(submission)} members={len(member_outputs)} judge={args.judge}")


if __name__ == "__main__":
    main()
