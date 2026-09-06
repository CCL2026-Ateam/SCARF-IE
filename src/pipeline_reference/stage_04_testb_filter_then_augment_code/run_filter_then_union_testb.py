"""Run filter-then-union augmentation on the final test_b prediction set.

This wrapper keeps the validated experiment code untouched. It reuses:
- filter_then_union_augment.py for candidate generation and final apply
- union_only_entity_ai_augmenter.py / relation_aug_core.py for prompts and chat client
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent
VALIDATION = PIPELINE_ROOT / "stage_01_validation_scaling_code"
BASE = PIPELINE_ROOT / "outputs/from_scratch/base_submit.json"
POOL = PIPELINE_ROOT / "outputs/from_scratch/union_pool.json"
SOLUTION_SRC = PIPELINE_ROOT / "stage_00_external_project_src"
WORK_OUT = PIPELINE_ROOT / "work/test_b_filter_then_augment"
BASE_WORK_OUT = WORK_OUT
DELIVER_OUT = PIPELINE_ROOT / "outputs/test_b_filter_then_augment"
TRIALS_FOR_EXAMPLES = [0, 1, 4, 6, 7]
ENTITY_LABELS = ["CROP", "VAR", "TRT", "GST", "GENE", "QTL", "MRK", "CHR", "BM", "CROSS", "ABS", "BIS"]
RELATION_LABELS = ["CON", "LOI"]
RUN_NAME = ""
OUTPUT_PREFIX = "test_b_filter_then_union_augmented"
ENTITY_DEDUP = "default"
MENTION_REPEAT_LABELS = {"TRT", "VAR", "CROP", "GENE"}

sys.path.insert(0, str(VALIDATION))
import env_loader  # noqa: E402
import filter_then_union_augment as ftu  # noqa: E402
import relation_aug_core as rel_aug  # noqa: E402
import union_only_entity_ai_augmenter as ent_aug  # noqa: E402


def load_env() -> None:
    env_loader.load_env_defaults(*env_loader.default_env_paths(PIPELINE_ROOT))
    env_loader.load_env_defaults(*env_loader.default_env_paths(SOLUTION_SRC))


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def append_jsonl(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def entity_candidate_path() -> Path:
    return WORK_OUT / "entity_candidates.json"


def entity_judgment_path() -> Path:
    return WORK_OUT / "entity_ai_judgments.jsonl"


def relation_candidate_path() -> Path:
    return relation_work_dir() / "relation_candidates_after_entity.json"


def relation_judgment_path() -> Path:
    return relation_work_dir() / "relation_ai_judgments.jsonl"


def entity_augmented_path() -> Path:
    return WORK_OUT / "entity_augmented_base.json"


def summary_path(name: str) -> Path:
    return WORK_OUT / f"{name}_summary.json"


def relation_work_dir() -> Path:
    return WORK_OUT / RUN_NAME if RUN_NAME else WORK_OUT


def entity_span_key(entity: dict) -> tuple[int, int]:
    return int(entity["start"]), int(entity["end"])


def make_entity_candidates_span_only(base: list[dict], pool: list[dict], labels: list[str]) -> tuple[list[dict], Counter]:
    allow = set(labels or ENTITY_LABELS)
    rows, by_label = [], Counter()
    for i, (base_row, pool_row) in enumerate(zip(base, pool)):
        seen = {entity_span_key(e) for e in base_row.get("entities", []) or []}
        for ent in pool_row.get("entities", []) or []:
            label = str(ent.get("label", ""))
            if label not in allow:
                continue
            span = entity_span_key(ent)
            if span in seen:
                continue
            row = {
                "id": ftu.entity_id(i, ent),
                "kind": "entity",
                "record_index": i,
                "text": str(ent.get("text", "")),
                "start": int(ent["start"]),
                "end": int(ent["end"]),
                "label": label,
            }
            rows.append(row)
            by_label[label] += 1
            seen.add(span)
    return rows, by_label


def make_entity_candidates_label_span_only(
    base: list[dict],
    pool: list[dict],
    labels: list[str],
    span_only_labels: set[str],
) -> tuple[list[dict], Counter]:
    allow = set(labels or ENTITY_LABELS)
    rows, by_label = [], Counter()
    for i, (base_row, pool_row) in enumerate(zip(base, pool)):
        existing = {ftu.entity_key(e) for e in base_row.get("entities", []) or []}
        text_seen = {ftu.entity_text_key(e) for e in base_row.get("entities", []) or []}
        spans_seen = {entity_span_key(e) for e in base_row.get("entities", []) or []}
        seen = set(existing)
        for ent in pool_row.get("entities", []) or []:
            label = str(ent.get("label", ""))
            if label not in allow:
                continue
            if label in span_only_labels:
                span = entity_span_key(ent)
                if span in spans_seen:
                    continue
            elif ftu.entity_key(ent) in seen or ftu.entity_text_key(ent) in text_seen:
                continue
            row = {
                "id": ftu.entity_id(i, ent),
                "kind": "entity",
                "record_index": i,
                "text": str(ent.get("text", "")),
                "start": int(ent["start"]),
                "end": int(ent["end"]),
                "label": label,
            }
            rows.append(row)
            by_label[label] += 1
            if label in span_only_labels:
                spans_seen.add(entity_span_key(ent))
            else:
                seen.add(ftu.entity_key(ent))
    return rows, by_label


def make_entity_candidates_variant(base: list[dict], pool: list[dict]) -> tuple[list[dict], Counter]:
    if ENTITY_DEDUP == "span_only":
        return make_entity_candidates_span_only(base, pool, ENTITY_LABELS)
    if ENTITY_DEDUP == "label_span_only":
        return make_entity_candidates_label_span_only(base, pool, ENTITY_LABELS, MENTION_REPEAT_LABELS)
    return ftu.make_entity_candidates(base, pool, ENTITY_LABELS)


def add_entities_span_only(base, candidates, decisions, keep_threshold, fix_threshold, missing_action):
    additions = [[] for _ in base]
    stats, by_label, applied_rows = Counter(), Counter(), []
    for row in candidates:
        rec = base[row["record_index"]]
        decision = ftu.choose_decision(
            decisions.get(row["id"], []),
            keep_threshold,
            fix_threshold,
            missing_action,
        )
        action = ftu.normalize_action(decision.get("action"))
        ent = None
        if action == "keep":
            ent = {"start": row["start"], "end": row["end"], "text": row["text"], "label": row["label"]}
        elif action == "fix":
            ent = ftu.resolve_corrected_entity(decision.get("corrected"), rec.get("text", ""))
        if ent and ftu.valid_entity(ent, rec.get("text", "")):
            additions[row["record_index"]].append(ent)
            stats["accepted"] += 1
            by_label[ent["label"]] += 1
        else:
            stats["rejected"] += 1
        applied_rows.append(
            {
                **row,
                "action": action,
                "confidence": ftu.confidence(decision),
                "applied": bool(ent),
            }
        )

    out = []
    changed = 0
    for rec, ents in zip(base, additions):
        row = dict(rec)
        existing = {entity_span_key(e) for e in row.get("entities", []) or []}
        merged = list(row.get("entities", []) or [])
        for ent in ents:
            span = entity_span_key(ent)
            if span in existing:
                continue
            merged.append(ent)
            existing.add(span)
            changed += 1
        row["entities"] = sorted(merged, key=lambda e: (int(e["start"]), int(e["end"]), str(e["label"])))
        out.append(row)
    stats["changed_entities"] = changed
    return out, stats, by_label, applied_rows


def add_entities_label_span_only(base, candidates, decisions, keep_threshold, fix_threshold, missing_action):
    additions = [[] for _ in base]
    stats, by_label, applied_rows = Counter(), Counter(), []
    for row in candidates:
        rec = base[row["record_index"]]
        decision = ftu.choose_decision(
            decisions.get(row["id"], []),
            keep_threshold,
            fix_threshold,
            missing_action,
        )
        action = ftu.normalize_action(decision.get("action"))
        ent = None
        if action == "keep":
            ent = {"start": row["start"], "end": row["end"], "text": row["text"], "label": row["label"]}
        elif action == "fix":
            ent = ftu.resolve_corrected_entity(decision.get("corrected"), rec.get("text", ""))
        if ent and ftu.valid_entity(ent, rec.get("text", "")):
            additions[row["record_index"]].append(ent)
            stats["accepted"] += 1
            by_label[ent["label"]] += 1
        else:
            stats["rejected"] += 1
        applied_rows.append(
            {
                **row,
                "action": action,
                "confidence": ftu.confidence(decision),
                "applied": bool(ent),
            }
        )

    out = []
    changed = 0
    for rec, ents in zip(base, additions):
        row = dict(rec)
        existing = {ftu.entity_key(e) for e in row.get("entities", []) or []}
        spans_seen = {entity_span_key(e) for e in row.get("entities", []) or []}
        merged = list(row.get("entities", []) or [])
        for ent in ents:
            if ent["label"] in MENTION_REPEAT_LABELS:
                span = entity_span_key(ent)
                if span in spans_seen:
                    continue
                merged.append(ent)
                spans_seen.add(span)
            else:
                key = ftu.entity_key(ent)
                if key in existing:
                    continue
                merged.append(ent)
                existing.add(key)
            changed += 1
        row["entities"] = sorted(merged, key=lambda e: (int(e["start"]), int(e["end"]), str(e["label"])))
        out.append(row)
    stats["changed_entities"] = changed
    return out, stats, by_label, applied_rows


def as_entity_prompt_row(row: dict) -> dict:
    out = dict(row)
    out["candidate_id"] = row["id"]
    return out


def as_relation_prompt_row(row: dict) -> dict:
    out = dict(row)
    out["candidate_id"] = row["id"]
    return out


def done_ids(path: Path) -> set[str]:
    done: set[str] = set()
    for row in load_jsonl(path):
        for cid in row.get("candidate_ids", []) or []:
            done.add(str(cid))
        for decision in row.get("decisions", []) or []:
            cid = str(decision.get("id") or decision.get("candidate_id") or "").strip()
            if cid:
                done.add(cid)
    return done


def collect_decisions(path: Path) -> dict[str, list[dict]]:
    latest = {}
    for row in load_jsonl(path):
        for decision in row.get("decisions", []) or []:
            cid = str(decision.get("id") or decision.get("candidate_id") or "").strip()
            if cid:
                latest[cid] = decision
    grouped: dict[str, list[dict]] = defaultdict(list)
    for cid, decision in latest.items():
        grouped[cid].append(decision)
    return grouped


def group_entity_candidates(rows: list[dict], records: list[dict], completed: set[str]) -> list[tuple[int, int, list[dict]]]:
    by_record: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        if row["id"] not in completed:
            by_record[int(row["record_index"])].append(as_entity_prompt_row(row))
    jobs = []
    for record_index, candidates in sorted(by_record.items()):
        if not candidates:
            continue
        if not records[record_index].get("text"):
            raise RuntimeError(f"record {record_index} has no text")
        jobs.append((record_index, 0, candidates))
    return jobs


def group_relation_candidates(rows: list[dict], records: list[dict], completed: set[str]) -> list[tuple[int, int, list[dict]]]:
    by_record: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        if row["id"] not in completed:
            by_record[int(row["record_index"])].append(as_relation_prompt_row(row))
    jobs = []
    for record_index, candidates in sorted(by_record.items()):
        if not candidates:
            continue
        if not records[record_index].get("text"):
            raise RuntimeError(f"record {record_index} has no text")
        jobs.append((record_index, 0, candidates))
    return jobs


def split_batches(jobs: list[tuple[int, int, list[dict]]], batch_items: int, max_batches: int | None):
    batches = []
    for record_index, _, candidates in jobs:
        for batch_index, batch in ent_aug.chunked(candidates, batch_items):
            batches.append((record_index, batch_index, batch))
            if max_batches is not None and len(batches) >= max_batches:
                return batches
    return batches


def client_from_args(args, system=None):
    load_env()
    model = args.model or os.getenv("OPUS_MODEL", os.getenv("MODEL_MAIN", "gpt-5.4"))
    base_url = args.base_url or os.getenv("OPUS_BASE_URL", os.getenv("BASE_URL", "https://zjapi.com"))
    api_key_env = args.api_key_env or env_loader.api_key_env_for_model(model)
    api_key = os.getenv(api_key_env) or os.getenv("API_KEY")
    if not api_key:
        raise SystemExit(f"missing API key env: {api_key_env} or API_KEY")
    return ent_aug.ChatClient(api_key, base_url, model, args.timeout, args.retries, system=system), model, base_url


def prepare_entity_candidates() -> dict:
    WORK_OUT.mkdir(parents=True, exist_ok=True)
    base = ftu.load_records(BASE)
    pool = ftu.load_records(POOL)
    rows, by_label = make_entity_candidates_variant(base, pool)
    dump_json(entity_candidate_path(), rows)
    summary = {
        "base": str(BASE),
        "pool": str(POOL),
        "entity_dedup": ENTITY_DEDUP,
        "mention_repeat_labels": sorted(MENTION_REPEAT_LABELS) if ENTITY_DEDUP == "label_span_only" else [],
        "base_stats": ftu.count_records(base),
        "pool_stats": ftu.count_records(pool),
        "entity_candidates": len(rows),
        "entity_candidate_by_label": dict(sorted(by_label.items())),
    }
    dump_json(summary_path("entity_prepare"), summary)
    return summary


def judge_entities(args) -> dict:
    if not entity_candidate_path().exists():
        prepare_entity_candidates()
    rows = load_json(entity_candidate_path())
    base = ftu.load_records(BASE)
    completed = done_ids(entity_judgment_path())
    jobs = group_entity_candidates(rows, base, completed)
    batches = split_batches(jobs, args.entity_batch_items, args.max_batches)
    candidate_count = len(rows)
    pending_items = sum(len(batch) for _, _, batch in batches)
    by_label = Counter(row["label"] for row in rows)
    summary = {
        "candidate_count": candidate_count,
        "judged_candidates": len(completed),
        "pending_batches": len(batches),
        "pending_items": pending_items,
        "candidate_by_label": dict(sorted(by_label.items())),
    }
    print(json.dumps({"entity_plan": summary}, ensure_ascii=False), flush=True)
    if args.dry_run or not batches:
        dump_json(summary_path("entity_online"), summary)
        return summary

    client, model, base_url = client_from_args(args)
    example_index = None if args.no_label_examples else ent_aug.build_example_index(TRIALS_FOR_EXAMPLES)
    write_lock = threading.Lock()

    def audit(job):
        record_index, batch_index, batch = job
        record = base[record_index]
        examples = []
        if example_index is not None and args.examples_per_label > 0:
            examples = ent_aug.select_label_examples(
                example_index,
                {c["label"] for c in batch},
                -1,
                record_index,
                args.examples_per_label,
            )
        prompt = ent_aug.build_prompt(record["text"], record.get("entities", []) or [], batch, examples)
        raw = client.chat(prompt, args.max_tokens)
        parsed = ent_aug.parse_json_lenient(raw)
        if not isinstance(parsed, dict) or not isinstance(parsed.get("decisions"), list):
            parsed = {"decisions": [], "parse_error": True}
        out = {
            "kind": "entity",
            "record_index": record_index,
            "batch_index": batch_index,
            "prompt_hash": ent_aug.prompt_hash(prompt, model),
            "model": model,
            "base_url": base_url,
            "examples_per_label": 0 if args.no_label_examples else args.examples_per_label,
            "candidate_ids": [c["candidate_id"] for c in batch],
            "decisions": parsed.get("decisions", []),
        }
        if parsed.get("parse_error"):
            out["parse_error"] = True
        if args.keep_raw:
            out["raw"] = raw
        return out

    completed_batches = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(audit, job): job for job in batches}
        for fut in as_completed(futures):
            job = futures[fut]
            try:
                row = fut.result()
            except Exception as exc:
                failed += 1
                record_index, batch_index, batch = job
                print(
                    json.dumps(
                        {
                            "kind": "entity",
                            "record_index": record_index,
                            "batch_index": batch_index,
                            "candidate_ids": [c["candidate_id"] for c in batch],
                            "error": repr(exc),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                continue
            with write_lock:
                append_jsonl(entity_judgment_path(), row)
            completed_batches += 1
            if completed_batches % 10 == 0 or completed_batches + failed == len(batches):
                print(
                    f"entity progress completed={completed_batches} failed={failed}/{len(batches)}",
                    flush=True,
                )
    summary["completed_batches"] = completed_batches
    summary["failed_batches"] = failed
    summary["judged_candidates_after"] = len(done_ids(entity_judgment_path()))
    dump_json(summary_path("entity_online"), summary)
    if failed:
        raise SystemExit(f"entity online audit has failed batches: {failed}")
    return summary


def make_entity_augmented_base(args) -> dict:
    if not entity_candidate_path().exists():
        prepare_entity_candidates()
    base = ftu.load_records(BASE)
    pool = ftu.load_records(POOL)
    entity_rows = load_json(entity_candidate_path())
    entity_decisions = collect_decisions(entity_judgment_path())
    if ENTITY_DEDUP == "span_only":
        add_entities = add_entities_span_only
    elif ENTITY_DEDUP == "label_span_only":
        add_entities = add_entities_label_span_only
    else:
        add_entities = ftu.add_entities
    base_plus_entities, ent_stats, ent_by_label, ent_applied = add_entities(
        base,
        entity_rows,
        entity_decisions,
        args.entity_keep_threshold,
        args.entity_fix_threshold,
        "drop",
    )
    dump_json(entity_augmented_path(), base_plus_entities)
    dump_json(WORK_OUT / "applied_entity_decisions_preview.json", ent_applied)
    relation_rows, relation_by_label, skipped = ftu.make_relation_candidates(base_plus_entities, pool, RELATION_LABELS)
    dump_json(relation_candidate_path(), relation_rows)
    summary = {
        "entity_dedup": ENTITY_DEDUP,
        "mention_repeat_labels": sorted(MENTION_REPEAT_LABELS) if ENTITY_DEDUP == "label_span_only" else [],
        "entity_augmented_stats": ftu.count_records(base_plus_entities),
        "entity_apply": dict(sorted(ent_stats.items())),
        "entity_added_by_label": dict(sorted(ent_by_label.items())),
        "relation_candidates": len(relation_rows),
        "relation_candidate_by_label": dict(sorted(relation_by_label.items())),
        "relation_skipped_missing_endpoint": dict(sorted(skipped.items())),
    }
    dump_json(summary_path("entity_apply_relation_prepare"), summary)
    print(json.dumps({"entity_apply_relation_prepare": summary}, ensure_ascii=False), flush=True)
    return summary


def judge_relations(args) -> dict:
    if not relation_candidate_path().exists():
        make_entity_augmented_base(args)
    rows = load_json(relation_candidate_path())
    base_plus_entities = load_json(entity_augmented_path())
    completed = done_ids(relation_judgment_path())
    jobs = group_relation_candidates(rows, base_plus_entities, completed)
    batches = split_batches(jobs, args.relation_batch_items, args.max_batches)
    by_label = Counter(row["label"] for row in rows)
    summary = {
        "candidate_count": len(rows),
        "judged_candidates": len(completed),
        "pending_batches": len(batches),
        "pending_items": sum(len(batch) for _, _, batch in batches),
        "candidate_by_label": dict(sorted(by_label.items())),
    }
    print(json.dumps({"relation_plan": summary}, ensure_ascii=False), flush=True)
    if args.dry_run or not batches:
        dump_json(summary_path("relation_online"), summary)
        return summary

    client, model, base_url = client_from_args(args, system=rel_aug.SYSTEM)
    example_index = None if args.no_label_examples else rel_aug.build_relation_example_index(TRIALS_FOR_EXAMPLES)
    write_lock = threading.Lock()

    def audit(job):
        record_index, batch_index, batch = job
        record = base_plus_entities[record_index]
        examples = []
        if example_index is not None and args.examples_per_label > 0:
            examples = rel_aug.select_relation_examples(
                example_index,
                {c["label"] for c in batch},
                -1,
                record_index,
                args.examples_per_label,
            )
        prompt = rel_aug.build_prompt(
            record["text"],
            record.get("entities", []) or [],
            record.get("relations", []) or [],
            batch,
            examples,
        )
        raw = client.chat(prompt, args.max_tokens)
        parsed = ent_aug.parse_json_lenient(raw)
        if not isinstance(parsed, dict) or not isinstance(parsed.get("decisions"), list):
            parsed = {"decisions": [], "parse_error": True}
        out = {
            "kind": "relation",
            "record_index": record_index,
            "batch_index": batch_index,
            "prompt_hash": ent_aug.prompt_hash(prompt, model),
            "model": model,
            "base_url": base_url,
            "examples_per_label": 0 if args.no_label_examples else args.examples_per_label,
            "candidate_ids": [c["candidate_id"] for c in batch],
            "decisions": parsed.get("decisions", []),
        }
        if parsed.get("parse_error"):
            out["parse_error"] = True
        if args.keep_raw:
            out["raw"] = raw
        return out

    completed_batches = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(audit, job): job for job in batches}
        for fut in as_completed(futures):
            job = futures[fut]
            try:
                row = fut.result()
            except Exception as exc:
                failed += 1
                record_index, batch_index, batch = job
                print(
                    json.dumps(
                        {
                            "kind": "relation",
                            "record_index": record_index,
                            "batch_index": batch_index,
                            "candidate_ids": [c["candidate_id"] for c in batch],
                            "error": repr(exc),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                continue
            with write_lock:
                append_jsonl(relation_judgment_path(), row)
            completed_batches += 1
            if completed_batches % 10 == 0 or completed_batches + failed == len(batches):
                print(
                    f"relation progress completed={completed_batches} failed={failed}/{len(batches)}",
                    flush=True,
                )
    summary["completed_batches"] = completed_batches
    summary["failed_batches"] = failed
    summary["judged_candidates_after"] = len(done_ids(relation_judgment_path()))
    dump_json(summary_path("relation_online"), summary)
    if failed:
        raise SystemExit(f"relation online audit has failed batches: {failed}")
    return summary


def apply_final(args) -> dict:
    DELIVER_OUT.mkdir(parents=True, exist_ok=True)
    output_json = DELIVER_OUT / f"{OUTPUT_PREFIX}.json"
    output_zip = DELIVER_OUT / f"{OUTPUT_PREFIX}_submit.zip"
    output_summary = DELIVER_OUT / f"{OUTPUT_PREFIX}_summary.json"
    apply_args = argparse.Namespace(
        base=str(BASE),
        pool=str(POOL),
        out_dir=str(WORK_OUT),
        entity_labels=ENTITY_LABELS,
        relation_labels=RELATION_LABELS,
        output_relation_labels=RELATION_LABELS,
        entity_candidates=str(entity_candidate_path()),
        relation_candidates=str(relation_candidate_path()),
        entity_judgments=str(entity_judgment_path()),
        relation_judgments=str(relation_judgment_path()),
        entity_keep_threshold=args.entity_keep_threshold,
        entity_fix_threshold=args.entity_fix_threshold,
        relation_keep_threshold=args.relation_keep_threshold,
        relation_fix_threshold=args.relation_fix_threshold,
        missing_entity_action="drop",
        missing_relation_action="drop",
        apply_relation_fix=False,
        solution_src=str(SOLUTION_SRC),
        output_json=str(output_json),
        output_zip=str(output_zip),
        summary=str(output_summary),
    )
    original_add_entities = ftu.add_entities
    try:
        if ENTITY_DEDUP == "span_only":
            ftu.add_entities = add_entities_span_only
        elif ENTITY_DEDUP == "label_span_only":
            ftu.add_entities = add_entities_label_span_only
        ftu.apply(apply_args)
    finally:
        ftu.add_entities = original_add_entities
    return load_json(output_summary)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["all", "prepare", "entity", "relations", "apply"], default="all")
    parser.add_argument("--base", type=Path, default=None)
    parser.add_argument("--pool", type=Path, default=None)
    parser.add_argument("--work_out", type=Path, default=None)
    parser.add_argument("--deliver_out", type=Path, default=None)
    parser.add_argument("--solution_src", type=Path, default=None)
    parser.add_argument("--work_name", default="", help="Subdirectory under the base work out for this run.")
    parser.add_argument("--run_name", default="", help="Subdirectory under work out for relation candidates/judgments.")
    parser.add_argument("--output_prefix", default="test_b_filter_then_union_augmented")
    parser.add_argument("--entity_dedup", choices=["default", "span_only", "label_span_only"], default="default")
    parser.add_argument("--mention_repeat_labels", nargs="+", default=["TRT", "VAR", "CROP", "GENE"])
    parser.add_argument("--relation_labels", nargs="+", default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--entity_batch_items", type=int, default=20)
    parser.add_argument("--relation_batch_items", type=int, default=20)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--base_url", default=None)
    parser.add_argument("--api_key_env", default=None)
    parser.add_argument("--timeout", type=int, default=int(os.getenv("OPUS_TIMEOUT", "180")))
    parser.add_argument("--retries", type=int, default=int(os.getenv("OPUS_RETRIES", "3")))
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--examples_per_label", type=int, default=3)
    parser.add_argument("--no_label_examples", action="store_true")
    parser.add_argument("--entity_keep_threshold", type=float, default=0.92)
    parser.add_argument("--entity_fix_threshold", type=float, default=0.92)
    parser.add_argument("--relation_keep_threshold", type=float, default=0.92)
    parser.add_argument("--relation_fix_threshold", type=float, default=0.92)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--keep_raw", action="store_true")
    return parser.parse_args()


def main() -> None:
    global BASE, POOL, SOLUTION_SRC, DELIVER_OUT
    global RELATION_LABELS, RUN_NAME, OUTPUT_PREFIX, WORK_OUT, BASE_WORK_OUT, ENTITY_DEDUP, MENTION_REPEAT_LABELS
    args = parse_args()
    if args.base:
        BASE = args.base
    if args.pool:
        POOL = args.pool
    if args.work_out:
        WORK_OUT = args.work_out
        BASE_WORK_OUT = args.work_out
    if args.deliver_out:
        DELIVER_OUT = args.deliver_out
    if args.solution_src:
        SOLUTION_SRC = args.solution_src
    if args.relation_labels:
        RELATION_LABELS = args.relation_labels
    if args.work_name.strip():
        WORK_OUT = BASE_WORK_OUT / args.work_name.strip()
    RUN_NAME = args.run_name.strip()
    OUTPUT_PREFIX = args.output_prefix.strip() or OUTPUT_PREFIX
    ENTITY_DEDUP = args.entity_dedup
    MENTION_REPEAT_LABELS = {str(x) for x in args.mention_repeat_labels}
    if args.stage in {"all", "prepare"}:
        print(json.dumps({"prepare_entity": prepare_entity_candidates()}, ensure_ascii=False), flush=True)
    if args.stage in {"all", "entity"}:
        judge_entities(args)
    if args.stage in {"all", "relations"}:
        make_entity_augmented_base(args)
        judge_relations(args)
    if args.stage in {"all", "apply"}:
        summary = apply_final(args)
        print(json.dumps({"final_summary": summary}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
