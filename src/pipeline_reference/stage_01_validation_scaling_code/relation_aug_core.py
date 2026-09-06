"""Core helpers for AI-audited union-only relation augmentation."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import union_only_entity_ai_augmenter as ent_aug
from env_loader import default_env_paths, load_env_defaults


PROJECT = Path(__file__).resolve().parent
load_env_defaults(*default_env_paths(PROJECT))
PAIR_ROOT = PROJECT / "outputs" / "record_level_filter_5x5" / "pairs"
S100_T10 = PROJECT / "outputs" / "s100_t10"
OUT_DIR = Path(os.getenv("RELATION_AUG_OUT_DIR") or (PROJECT / "outputs" / "union_only_relation_ai_augmenter"))
SOLUTION_SRC = Path(os.getenv("SOLUTION_SRC") or (PROJECT.parent / "stage_00_external_project_src"))

DEFAULT_TRIALS = [0, 1, 4, 6, 7]
RELATION_LABELS = ["CON", "USE", "HAS", "AFF", "OCI", "LOI"]

SYSTEM = (
    "You are a strict information-extraction auditor for cereal/millet breeding "
    "literature. You judge whether candidate relations should be added to an "
    "existing high-precision extraction. Return strict JSON only."
)

GUIDE = """Relation labels:
- CON contains/belongs-to/is-a/alias/abbreviation/synonym/apposition
- USE uses/employs/applies/via/by/through method/technique/material
- HAS object has/shows/exhibits/possesses a trait
- AFF affects/regulates/promotes/suppresses/induces/responds/sensitive/tolerant/resistant
- OCI occurs-in/measured-at/investigated-at/treated-at a growth stage
- LOI located/mapped/linked/associated/candidate/co-localized molecular positioning

Audit rules:
1. Judge each ADD_RELATION_CANDIDATE independently against TEXT.
2. Keep only if the relation has direct textual evidence and the label direction is correct.
3. Drop relations supported only by co-occurrence, broad topic, or biological plausibility.
4. Endpoints are already accepted entities; do not reject just because an endpoint is newly added.
5. If a candidate is real but label/direction/endpoints are slightly wrong, action="fix" only when the corrected relation can be formed from BASE_ENTITIES.
6. Prefer precision over recall. A weak candidate should be dropped.
"""


def dump_json_atomic(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def load_json(path):
    return ent_aug.load_json(path)


def load_jsonl(path):
    return ent_aug.load_jsonl(path)


def append_jsonl(path, obj):
    return ent_aug.append_jsonl(path, obj)


def pair_dir(trial, route):
    return PAIR_ROOT / f"trial_{trial:02d}__{route}"


def entity_key(entity):
    return int(entity["start"]), int(entity["end"]), str(entity["label"])


def relation_key(relation):
    return (
        int(relation["head_start"]),
        int(relation["head_end"]),
        str(relation["head_type"]),
        int(relation["tail_start"]),
        int(relation["tail_end"]),
        str(relation["tail_type"]),
        str(relation["label"]),
    )


def relation_triple(relation):
    return str(relation["label"]), str(relation["head_type"]), str(relation["tail_type"])


def relation_id(trial, record_index, relation):
    raw = "\0".join(
        [
            str(trial),
            str(record_index),
            str(relation["head_start"]),
            str(relation["head_end"]),
            str(relation["head_type"]),
            str(relation["tail_start"]),
            str(relation["tail_end"]),
            str(relation["tail_type"]),
            str(relation["label"]),
            str(relation.get("head", "")),
            str(relation.get("tail", "")),
        ]
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def endpoints_exist(relation, entity_keys):
    return (
        (int(relation["head_start"]), int(relation["head_end"]), str(relation["head_type"])) in entity_keys
        and (int(relation["tail_start"]), int(relation["tail_end"]), str(relation["tail_type"])) in entity_keys
    )


def sorted_relations(relations):
    return sorted(
        relations,
        key=lambda r: (
            int(r["head_start"]),
            int(r["head_end"]),
            int(r["tail_start"]),
            int(r["tail_end"]),
            str(r["label"]),
        ),
    )


def judgment_path():
    return OUT_DIR / "union_only_relation_ai_judgments.jsonl"


def candidate_json_path():
    return OUT_DIR / "union_only_relation_candidates.json"


def candidate_summary_path():
    return OUT_DIR / "candidate_summary.json"


def build_entity_augmented_base(trials, keep_threshold=0.92, fix_threshold=0.92, entity_labels=None):
    """Build strict3 + AI-reviewed union-only entity additions."""
    entity_rows, _, _ = ent_aug.make_candidates(trials, entity_labels)
    entity_decisions = ent_aug.collect_decisions()
    outputs = {}
    summary = {
        "entity_labels": entity_labels or "all",
        "changed_entities": 0,
        "added_by_label": Counter(),
        "missing_entity_judgments": 0,
    }

    for trial in trials:
        strict = load_json(pair_dir(trial, "strict3") / "filtered_pred.json")
        additions = [[] for _ in strict]
        for row in entity_rows:
            if row["trial"] != trial:
                continue
            decisions = entity_decisions.get(row["candidate_id"], [])
            if not decisions:
                summary["missing_entity_judgments"] += 1
                continue
            decision = ent_aug.choose_decision(decisions, keep_threshold, fix_threshold)
            action = ent_aug.normalize_action(decision.get("action"))
            entity = None
            if action == "keep":
                entity = {"start": row["start"], "end": row["end"], "text": row["text"], "label": row["label"]}
            elif action == "fix" and ent_aug.valid_corrected_entity(decision.get("corrected"), strict[row["record_index"]]["text"]):
                entity = ent_aug.resolve_corrected_entity(decision["corrected"], strict[row["record_index"]]["text"])
            if entity:
                additions[row["record_index"]].append(entity)
                summary["added_by_label"][entity["label"]] += 1
        outputs[trial], changed = ent_aug.add_entities(strict, additions)
        summary["changed_entities"] += changed

    summary["added_by_label"] = dict(sorted(summary["added_by_label"].items()))
    return outputs, summary


def make_candidates(trials, base_by_trial, relation_labels=None):
    label_allow = set(relation_labels or [])
    rows = []
    by_record = defaultdict(list)
    stats = Counter()
    skipped_missing_endpoint = Counter()

    for trial in trials:
        strict = load_json(pair_dir(trial, "strict3") / "filtered_pred.json")
        union = load_json(pair_dir(trial, "union") / "filtered_pred.json")
        base = base_by_trial[trial]
        for record_index, (strict_row, union_row, base_row) in enumerate(zip(strict, union, base)):
            existing_relations = {relation_key(r) for r in strict_row.get("relations", []) or []}
            entity_keys = {entity_key(e) for e in base_row.get("entities", []) or []}
            seen = set(existing_relations)
            for relation in union_row.get("relations", []) or []:
                key = relation_key(relation)
                if key in seen:
                    continue
                seen.add(key)
                label = str(relation.get("label", ""))
                if label_allow and label not in label_allow:
                    continue
                if not endpoints_exist(relation, entity_keys):
                    skipped_missing_endpoint[label] += 1
                    continue
                row = {
                    "candidate_id": relation_id(trial, record_index, relation),
                    "trial": trial,
                    "record_index": record_index,
                    "label": label,
                    "head": relation.get("head", ""),
                    "head_start": int(relation["head_start"]),
                    "head_end": int(relation["head_end"]),
                    "head_type": str(relation["head_type"]),
                    "tail": relation.get("tail", ""),
                    "tail_start": int(relation["tail_start"]),
                    "tail_end": int(relation["tail_end"]),
                    "tail_type": str(relation["tail_type"]),
                }
                rows.append(row)
                by_record[(trial, record_index)].append(row)
                stats[label] += 1

    return rows, by_record, stats, skipped_missing_endpoint


def write_candidate_files(
    trials,
    relation_labels,
    entity_keep_threshold=0.92,
    entity_fix_threshold=0.92,
    entity_labels=None,
):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base_by_trial, entity_summary = build_entity_augmented_base(
        trials,
        entity_keep_threshold,
        entity_fix_threshold,
        entity_labels,
    )
    rows, by_record, stats, skipped = make_candidates(trials, base_by_trial, relation_labels)
    dump_json_atomic(candidate_json_path(), rows)
    dump_json_atomic(
        candidate_summary_path(),
        {
            "trials": trials,
            "entity_labels": entity_labels or "all",
            "relation_labels": relation_labels or "all",
            "candidates": len(rows),
            "records_with_candidates": len(by_record),
            "by_label": dict(sorted(stats.items())),
            "skipped_missing_endpoint": dict(sorted(skipped.items())),
            "entity_base": entity_summary,
        },
    )
    return rows, by_record, stats, skipped, base_by_trial, entity_summary


def build_relation_example_index(trials):
    examples = defaultdict(list)
    for trial in trials:
        gold = load_json(S100_T10 / f"trial_{trial:02d}" / "gold.json")
        for record_index, record in enumerate(gold):
            text = record.get("text", "")
            for relation in record.get("relations", []) or []:
                label = str(relation.get("label", ""))
                if label not in RELATION_LABELS:
                    continue
                start = min(int(relation["head_start"]), int(relation["tail_start"]))
                end = max(int(relation["head_end"]), int(relation["tail_end"]))
                examples[label].append(
                    {
                        "trial": trial,
                        "record_index": record_index,
                        "label": label,
                        "head": relation.get("head", ""),
                        "head_type": relation.get("head_type", ""),
                        "tail": relation.get("tail", ""),
                        "tail_type": relation.get("tail_type", ""),
                        "context": ent_aug.context_snippet(text, start, end, window=120),
                    }
                )
    return examples


def select_relation_examples(example_index, labels, current_trial, current_record_index, per_label):
    selected = []
    for label in sorted(labels):
        count = 0
        for example in example_index.get(label, []):
            if example["trial"] == current_trial and example["record_index"] == current_record_index:
                continue
            selected.append(
                {
                    "label": label,
                    "head": example["head"],
                    "head_type": example["head_type"],
                    "tail": example["tail"],
                    "tail_type": example["tail_type"],
                    "context": example["context"],
                }
            )
            count += 1
            if count >= per_label:
                break
    return selected


def build_prompt(text, base_entities, base_relations, candidates, examples=None):
    compact_entities = [
        {"text": e.get("text", ""), "start": e.get("start"), "end": e.get("end"), "label": e.get("label", "")}
        for e in base_entities
    ]
    compact_relations = [
        {
            "head": r.get("head", ""),
            "head_type": r.get("head_type", ""),
            "tail": r.get("tail", ""),
            "tail_type": r.get("tail_type", ""),
            "label": r.get("label", ""),
        }
        for r in base_relations
    ]
    compact_candidates = [
        {
            "id": c["candidate_id"],
            "head": c["head"],
            "head_type": c["head_type"],
            "tail": c["tail"],
            "tail_type": c["tail_type"],
            "label": c["label"],
        }
        for c in candidates
    ]
    examples_block = ""
    if examples:
        examples_block = (
            "\nVALIDATION_GOLD_RELATION_EXAMPLES:\n"
            + json.dumps(examples, ensure_ascii=False)
            + "\nUse these as label semantics examples only. They are not candidates for the current TEXT.\n"
        )
    return f"""{GUIDE}

TEXT:
{text}

BASE_ENTITIES:
{json.dumps(compact_entities, ensure_ascii=False)}

EXISTING_BASE_RELATIONS:
{json.dumps(compact_relations, ensure_ascii=False)}
{examples_block}

ADD_RELATION_CANDIDATES:
{json.dumps(compact_candidates, ensure_ascii=False)}

Return this JSON shape:
{{
  "decisions": [
    {{
      "id": "<candidate id>",
      "action": "keep|drop|fix",
      "confidence": 0.0,
      "reason": "short reason",
      "corrected": null
    }}
  ]
}}

For fixed relations, corrected must be {{"head": "...", "head_type": "...", "tail": "...", "tail_type": "...", "label": "..."}}.
"""


def done_candidate_ids(path):
    done = set()
    for row in load_jsonl(path):
        for decision in row.get("decisions", []) or []:
            cid = str(decision.get("id", "")).strip()
            if cid:
                done.add(cid)
    return done


def run_online(args):
    rows, by_record, stats, skipped, base_by_trial, entity_summary = write_candidate_files(
        args.trials,
        args.relation_labels,
        args.entity_keep_threshold,
        args.entity_fix_threshold,
        getattr(args, "entity_labels", None),
    )
    done = done_candidate_ids(judgment_path())
    example_index = None if args.no_label_examples else build_relation_example_index(args.trials)
    jobs = []
    planned_items = 0
    for (trial, record_index), candidates in sorted(by_record.items()):
        pending = [c for c in candidates if c["candidate_id"] not in done]
        if not pending:
            continue
        planned_items += len(pending)
        for batch_index, batch in ent_aug.chunked(pending, args.batch_items):
            if args.max_batches is not None and len(jobs) >= args.max_batches:
                break
            jobs.append((trial, record_index, batch_index, batch))
        if args.max_batches is not None and len(jobs) >= args.max_batches:
            break

    print(
        f"[PLAN] candidates={len(rows)} labels={dict(sorted(stats.items()))} "
        f"skipped_missing_endpoint={dict(sorted(skipped.items()))} "
        f"pending_batches={len(jobs)} pending_items={planned_items} model={args.model}",
        flush=True,
    )
    if args.dry_run:
        return

    api_key = os.getenv(args.api_key_env) or os.getenv("API_KEY")
    if not api_key:
        raise SystemExit(f"missing API key env: {args.api_key_env} or API_KEY")
    client = ent_aug.ChatClient(api_key, args.base_url, args.model, args.timeout, args.retries, system=SYSTEM)
    write_lock = threading.Lock()

    def audit_job(job):
        trial, record_index, batch_index, batch = job
        base_row = base_by_trial[trial][record_index]
        examples = []
        if example_index is not None and args.examples_per_label > 0:
            examples = select_relation_examples(
                example_index,
                {c["label"] for c in batch},
                trial,
                record_index,
                args.examples_per_label,
            )
        prompt = build_prompt(
            base_row["text"],
            base_row.get("entities", []) or [],
            base_row.get("relations", []) or [],
            batch,
            examples,
        )
        raw = client.chat(prompt, args.max_tokens)
        parsed = ent_aug.parse_json_lenient(raw)
        if not isinstance(parsed, dict) or not isinstance(parsed.get("decisions"), list):
            parsed = {"decisions": [], "parse_error": True}
        out = {
            "trial": trial,
            "record_index": record_index,
            "batch_index": batch_index,
            "prompt_hash": ent_aug.prompt_hash(prompt, args.model),
            "model": args.model,
            "base_url": args.base_url,
            "examples_per_label": 0 if args.no_label_examples else args.examples_per_label,
            "candidate_ids": [c["candidate_id"] for c in batch],
            "decisions": parsed.get("decisions", []),
        }
        if parsed.get("parse_error"):
            out["parse_error"] = True
        if args.keep_raw:
            out["raw"] = raw
        return out

    completed = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(audit_job, job): job for job in jobs}
        for fut in as_completed(futures):
            job = futures[fut]
            try:
                row = fut.result()
            except Exception as exc:
                failed += 1
                trial, record_index, batch_index, _ = job
                print(
                    json.dumps({"trial": trial, "record_index": record_index, "batch_index": batch_index, "error": repr(exc)}, ensure_ascii=False),
                    flush=True,
                )
                continue
            with write_lock:
                append_jsonl(judgment_path(), row)
            completed += 1
            if completed % 10 == 0 or completed + failed == len(jobs):
                print(f"  progress completed={completed} failed={failed}/{len(jobs)} -> {judgment_path()}", flush=True)


def collect_decisions():
    latest = {}
    for row in load_jsonl(judgment_path()):
        for decision in row.get("decisions", []) or []:
            cid = str(decision.get("id", "")).strip()
            if cid:
                latest[cid] = decision
    out = defaultdict(list)
    for cid, decision in latest.items():
        out[cid].append(decision)
    return out


def choose_decision(decisions, keep_threshold, fix_threshold):
    if not decisions:
        return {"action": "drop", "confidence": 0.0, "reason": "missing judgment"}
    fixes = [d for d in decisions if ent_aug.normalize_action(d.get("action")) == "fix" and ent_aug.confidence(d) >= fix_threshold]
    keeps = [d for d in decisions if ent_aug.normalize_action(d.get("action")) == "keep" and ent_aug.confidence(d) >= keep_threshold]
    drops = [d for d in decisions if ent_aug.normalize_action(d.get("action")) == "drop"]
    if fixes:
        return max(fixes, key=ent_aug.confidence)
    if keeps:
        return max(keeps, key=ent_aug.confidence)
    if drops:
        return max(drops, key=ent_aug.confidence)
    return max(decisions, key=ent_aug.confidence)


def resolve_corrected_relation(corrected, base_entities):
    if not isinstance(corrected, dict):
        return None
    label = str(corrected.get("label", ""))
    if label not in RELATION_LABELS:
        return None
    head = str(corrected.get("head", ""))
    head_type = str(corrected.get("head_type", ""))
    tail = str(corrected.get("tail", ""))
    tail_type = str(corrected.get("tail_type", ""))
    if not all([head, head_type, tail, tail_type]):
        return None
    heads = [e for e in base_entities if e.get("text") == head and e.get("label") == head_type]
    tails = [e for e in base_entities if e.get("text") == tail and e.get("label") == tail_type]
    if len(heads) != 1 or len(tails) != 1:
        return None
    h = heads[0]
    t = tails[0]
    return {
        "head": head,
        "head_start": int(h["start"]),
        "head_end": int(h["end"]),
        "head_type": head_type,
        "tail": tail,
        "tail_start": int(t["start"]),
        "tail_end": int(t["end"]),
        "tail_type": tail_type,
        "label": label,
    }


def add_relations(base, additions):
    out = []
    changed = 0
    for record, rels in zip(base, additions):
        row = dict(record)
        existing = {relation_key(r) for r in row.get("relations", []) or []}
        merged = list(row.get("relations", []) or [])
        for relation in rels:
            key = relation_key(relation)
            if key in existing:
                continue
            merged.append(relation)
            existing.add(key)
            changed += 1
        row["relations"] = sorted_relations(merged)
        out.append(row)
    return out, changed


def evaluate_outputs(trials, outputs_by_trial):
    sys.path.insert(0, str(SOLUTION_SRC))
    from common import evaluate  # noqa: E402

    rows = []
    for trial in trials:
        gold = load_json(S100_T10 / f"trial_{trial:02d}" / "gold.json")
        pred = outputs_by_trial[trial]
        m = evaluate(gold, pred)
        rows.append(
            {
                "trial": trial,
                "total": m["total_score"],
                "ner": m["score_ner"],
                "re": m["score_re"],
                "entity": m["entity"],
                "relation": m["relation"],
            }
        )
    return {
        "rows": rows,
        "mean_total": sum(r["total"] for r in rows) / len(rows),
        "mean_ner": sum(r["ner"] for r in rows) / len(rows),
        "mean_re": sum(r["re"] for r in rows) / len(rows),
        "entity_tp": sum(r["entity"]["tp"] for r in rows),
        "entity_fp": sum(r["entity"]["fp"] for r in rows),
        "entity_fn": sum(r["entity"]["fn"] for r in rows),
        "relation_tp": sum(r["relation"]["tp"] for r in rows),
        "relation_fp": sum(r["relation"]["fp"] for r in rows),
        "relation_fn": sum(r["relation"]["fn"] for r in rows),
    }


def write_zip(json_path, zip_path):
    import zipfile

    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(json_path, "submit.json")


def apply_augments(args):
    rows, by_record, stats, skipped, base_by_trial, entity_summary = write_candidate_files(
        args.trials,
        args.relation_labels,
        args.entity_keep_threshold,
        args.entity_fix_threshold,
        getattr(args, "entity_labels", None),
    )
    decisions = collect_decisions()
    outputs_by_trial = {}
    apply_stats = Counter()
    by_label = Counter()
    decision_rows = []

    for trial in args.trials:
        base = base_by_trial[trial]
        additions = [[] for _ in base]
        for row in rows:
            if row["trial"] != trial:
                continue
            decision = choose_decision(decisions.get(row["candidate_id"], []), args.keep_threshold, args.fix_threshold)
            action = ent_aug.normalize_action(decision.get("action"))
            relation = None
            if action == "keep":
                relation = {
                    "head": row["head"],
                    "head_start": row["head_start"],
                    "head_end": row["head_end"],
                    "head_type": row["head_type"],
                    "tail": row["tail"],
                    "tail_start": row["tail_start"],
                    "tail_end": row["tail_end"],
                    "tail_type": row["tail_type"],
                    "label": row["label"],
                }
            elif action == "fix":
                relation = resolve_corrected_relation(decision.get("corrected"), base[row["record_index"]].get("entities", []) or [])
            output_allow = set(args.output_relation_labels or [])
            if relation and output_allow and relation["label"] not in output_allow:
                relation = None
            if relation and endpoints_exist(relation, {entity_key(e) for e in base[row["record_index"]].get("entities", []) or []}):
                additions[row["record_index"]].append(relation)
                apply_stats["added"] += 1
                by_label[relation["label"]] += 1
            else:
                apply_stats["rejected"] += 1
            decision_rows.append(
                {
                    **row,
                    "action": action,
                    "confidence": ent_aug.confidence(decision),
                    "reason": decision.get("reason", ""),
                    "applied": bool(relation),
                    "applied_relation": relation,
                }
            )
        outputs_by_trial[trial], changed = add_relations(base, additions)
        apply_stats["changed_relations"] += changed
        entity_mode = "entity_ai_partial" if getattr(args, "entity_labels", None) else "entity_ai_all"
        relation_mode = "relation_ai_partial" if args.output_relation_labels else "relation_ai_all"
        out_json = OUT_DIR / f"trial_{trial:02d}_strict3_{entity_mode}_{relation_mode}_augmented.json"
        out_zip = OUT_DIR / f"trial_{trial:02d}_strict3_{entity_mode}_{relation_mode}_augmented.zip"
        dump_json_atomic(out_json, outputs_by_trial[trial])
        write_zip(out_json, out_zip)

    dump_json_atomic(OUT_DIR / "applied_candidate_decisions.json", decision_rows)
    summary = {
        "trials": args.trials,
        "entity_labels": getattr(args, "entity_labels", None) or "all",
        "relation_labels": args.relation_labels or "all",
        "output_relation_labels": args.output_relation_labels or "all",
        "candidate_count": len(rows),
        "judged_candidates": sum(1 for row in rows if row["candidate_id"] in decisions),
        "candidate_by_label": dict(sorted(stats.items())),
        "skipped_missing_endpoint": dict(sorted(skipped.items())),
        "entity_base": entity_summary,
        "apply_stats": dict(sorted(apply_stats.items())),
        "added_by_label": dict(sorted(by_label.items())),
        "evaluation": evaluate_outputs(args.trials, outputs_by_trial),
    }
    dump_json_atomic(OUT_DIR / "augment_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def summarize_judgments(args):
    candidate_rows = {r["candidate_id"]: r for r in load_json(candidate_json_path())} if candidate_json_path().exists() else {}
    latest = {}
    parse_errors = 0
    for row in load_jsonl(judgment_path()):
        if row.get("parse_error"):
            parse_errors += 1
        for decision in row.get("decisions", []) or []:
            cid = str(decision.get("id", "")).strip()
            if cid:
                latest[cid] = decision
    actions = Counter()
    labels = Counter()
    for cid, decision in latest.items():
        action = ent_aug.normalize_action(decision.get("action"))
        actions[action] += 1
        if cid in candidate_rows:
            labels[candidate_rows[cid]["label"] + ":" + action] += 1
    summary = {
        "judgment_batches": len(load_jsonl(judgment_path())),
        "unique_decision_ids": len(latest),
        "parse_errors": parse_errors,
        "decisions_by_action": dict(sorted(actions.items())),
        "label_action_counts": dict(sorted(labels.items())),
    }
    dump_json_atomic(OUT_DIR / "judgment_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
