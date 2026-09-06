from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path


ROOT = Path("C:/Users/Zzl410410/Documents/Codex/2026-06-28/c-users-zzl410410-documents-codex-2026-3")
RUN_OPS = ROOT / "work/test_b_filter_then_augment/run_ops_audit_sharded.py"
APPLY_STRATEGY = Path("C:/Users/Zzl410410/Documents/Codex/2026-06-26/019ef541-55e7-7af1-8f34-0927a521fe82/work/apply_entity_pair_strategy.py")
OUT_DIR = ROOT / "outputs/test_b_filter_then_augment"
WORK_DIR = ROOT / "work/test_b_filter_then_augment/iterative_relation_ops_spanonly"


def solution_src() -> Path:
    for child in Path("K:/").iterdir():
        candidate = child / "CCL/solution/solution/src"
        if candidate.exists():
            return candidate
    return Path("K:/浩然/CCL/solution/solution/src")


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def dump_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def file_slug(path: Path) -> str:
    import re

    return re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem)


def ranges(n_records: int, shards: int) -> list[tuple[int, int]]:
    width = (n_records + shards - 1) // shards
    return [(start, min(n_records, start + width)) for start in range(0, n_records, width)]


def shard_dir(out_dir: Path, shard_index: int, start: int, end: int) -> Path:
    return out_dir / f"shard_{shard_index:02d}_{start:03d}_{end:03d}"


def judgment_path(pred: Path, out_dir: Path, shard_index: int, start: int, end: int) -> Path:
    return shard_dir(out_dir, shard_index, start, end) / "judgments" / f"{file_slug(pred)}.opus_item_judgments.jsonl"


def read_rows(path: Path) -> dict[tuple[int, int], dict]:
    rows = {}
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[(row.get("record_index"), row.get("batch_index"))] = row
    return rows


def run_shard_process(
    pred: Path,
    out_dir: Path,
    shards: int,
    shard_index: int,
    start: int,
    end: int,
    model_args: list[str],
    tag: str,
) -> subprocess.Popen:
    sd = shard_dir(out_dir, shard_index, start, end)
    sd.mkdir(parents=True, exist_ok=True)
    stdout = open(sd / f"{tag}.stdout.log", "a", encoding="utf-8")
    stderr = open(sd / f"{tag}.stderr.log", "a", encoding="utf-8")
    cmd = [
        sys.executable,
        str(RUN_OPS),
        "run-shard",
        "--pred",
        str(pred),
        "--out_dir",
        str(out_dir),
        "--shards",
        str(shards),
        "--shard_index",
        str(shard_index),
        "--start",
        str(start),
        "--end",
        str(end),
        "--batch_items",
        "999",
        "--timeout",
        "240",
        "--retries",
        "5",
        "--max_tokens",
        "4096",
        *model_args,
    ]
    return subprocess.Popen(cmd, stdout=stdout, stderr=stderr)


def wait_processes(processes: list[subprocess.Popen], poll_seconds: int) -> None:
    while True:
        running = [p for p in processes if p.poll() is None]
        if not running:
            return
        print(json.dumps({"event": "ops_wait", "running": len(running)}, ensure_ascii=False), flush=True)
        time.sleep(poll_seconds)


def collect_round_rows(pred: Path, out_dir: Path, shards: int) -> tuple[dict[tuple[int, int], dict], dict]:
    records = load_json(pred)
    rows = {}
    shard_stats = []
    for i, (start, end) in enumerate(ranges(len(records), shards)):
        path = judgment_path(pred, out_dir, i, start, end)
        shard_rows = read_rows(path)
        rows.update(shard_rows)
        shard_stats.append(
            {
                "shard": i,
                "range": [start, end],
                "expected": end - start,
                "rows": len({k for k in shard_rows if isinstance(k[0], int)}),
                "parse_errors": sum(1 for row in shard_rows.values() if row.get("parse_error")),
            }
        )
    parse_errors = sum(1 for row in rows.values() if row.get("parse_error"))
    return rows, {
        "records": len(records),
        "rows": len(rows),
        "record_rows": len({k[0] for k in rows if isinstance(k[0], int)}),
        "parse_errors": parse_errors,
        "shards": shard_stats,
    }


def run_ops_round(pred: Path, round_dir: Path, shards: int, poll_seconds: int) -> Path:
    records = load_json(pred)
    round_dir.mkdir(parents=True, exist_ok=True)
    rr = ranges(len(records), shards)
    processes = [
        run_shard_process(pred, round_dir, shards, i, start, end, [], "opus")
        for i, (start, end) in enumerate(rr)
    ]
    wait_processes(processes, poll_seconds)
    rows, stats = collect_round_rows(pred, round_dir, shards)
    print(json.dumps({"event": "ops_initial_done", **stats}, ensure_ascii=False), flush=True)

    incomplete = [
        (i, start, end)
        for i, (start, end) in enumerate(rr)
        if len({k for k in read_rows(judgment_path(pred, round_dir, i, start, end)) if isinstance(k[0], int)}) < (end - start)
    ]
    if incomplete:
        print(json.dumps({"event": "ops_gemini_fallback", "incomplete": incomplete}, ensure_ascii=False), flush=True)
        model_args = [
            "--model",
            "gemini-3.1-pro-preview",
            "--base_url",
            "https://zjapi.com",
            "--api_key_env",
            "GEMINI_API_KEY",
        ]
        processes = [
            run_shard_process(pred, round_dir, shards, i, start, end, model_args, "gemini_fallback")
            for i, start, end in incomplete
        ]
        wait_processes(processes, poll_seconds)
        rows, stats = collect_round_rows(pred, round_dir, shards)
        print(json.dumps({"event": "ops_fallback_done", **stats}, ensure_ascii=False), flush=True)

    rows, stats = collect_round_rows(pred, round_dir, shards)
    if stats["record_rows"] != stats["records"]:
        raise RuntimeError(f"OPS round incomplete: {stats}")

    parse_records = sorted({row["record_index"] for row in rows.values() if row.get("parse_error")})
    if parse_records:
        print(json.dumps({"event": "ops_parse_fix", "records": parse_records}, ensure_ascii=False), flush=True)
        fix_dir = round_dir / "parsefix"
        for fix_i, idx in enumerate(parse_records):
            model_args = [
                "--model",
                "gemini-3.1-pro-preview",
                "--base_url",
                "https://zjapi.com",
                "--api_key_env",
                "GEMINI_API_KEY",
            ]
            p = run_shard_process(pred, fix_dir, len(parse_records), fix_i, idx, idx + 1, model_args, "parsefix")
            wait_processes([p], poll_seconds)
        for fix_i, idx in enumerate(parse_records):
            rows.update(read_rows(judgment_path(pred, fix_dir, fix_i, idx, idx + 1)))
        if any(row.get("parse_error") for row in rows.values()):
            raise RuntimeError("parse_error remained after Gemini parsefix")

    merged = round_dir / "merged" / f"{file_slug(pred)}.opus_item_judgments_fixed.jsonl"
    merged.parent.mkdir(parents=True, exist_ok=True)
    with open(merged, "w", encoding="utf-8") as f:
        for _, row in sorted(rows.items(), key=lambda kv: (kv[0][0], kv[0][1])):
            row = dict(row)
            row["source_file"] = str(pred.resolve())
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "event": "ops_merged",
                "path": str(merged),
                "rows": len(rows),
                "records": len({k[0] for k in rows if isinstance(k[0], int)}),
                "parse_errors": sum(1 for row in rows.values() if row.get("parse_error")),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return merged


def write_zip(json_path: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(json_path, arcname="submit.json")


def apply_full_ops(pred: Path, judgments: Path, output_prefix: str) -> tuple[Path, Path]:
    output_json = OUT_DIR / f"{output_prefix}.json"
    output_zip = OUT_DIR / f"{output_prefix}_submit.zip"
    summary = OUT_DIR / f"{output_prefix}_summary.json"
    strategy = OUT_DIR / f"{output_prefix}_strategy.json"
    cmd = [
        sys.executable,
        str(APPLY_STRATEGY),
        "--pred",
        str(pred),
        "--judgments",
        str(judgments),
        "--solution_src",
        str(solution_src()),
        "--output_json",
        str(output_json),
        "--output_zip",
        str(output_zip),
        "--summary",
        str(summary),
        "--preset",
        "safe_anydrop_rel098",
        "--entity_policy",
        "drop_threshold",
        "--entity_drop_threshold",
        "0.75",
        "--apply_entity_fix",
        "true",
        "--entity_fix_threshold",
        "0.0",
        "--relation_policy",
        "drop_threshold",
        "--relation_drop_threshold",
        "0.95",
        "--apply_relation_fix",
        "false",
        "--protect_relation_endpoints",
        "false",
        "--noempty_fallback",
        "true",
        "--write_strategy_json",
        str(strategy),
    ]
    result = subprocess.run(cmd, text=True, encoding="utf-8", capture_output=True)
    if result.returncode:
        print(result.stdout, flush=True)
        print(result.stderr, flush=True)
        raise RuntimeError(f"full apply failed: {result.returncode}")
    print(result.stdout, flush=True)
    return output_json, summary


def normalize_action(action) -> str:
    action = str(action or "").strip().lower()
    return action if action in {"keep", "drop", "fix"} else "keep"


def confidence(decision: dict) -> float:
    try:
        return float(decision.get("confidence", 0) or 0)
    except Exception:
        return 0.0


def collect_decisions(rows: list[dict]) -> dict[int, dict[str, list[dict]]]:
    by_record: dict[int, dict[str, list[dict]]] = {}
    for row in rows:
        idx = row.get("record_index")
        if idx is None:
            continue
        record_decisions = by_record.setdefault(int(idx), {})
        for decision in row.get("decisions", []) or []:
            cid = str(decision.get("id", "")).strip()
            if cid:
                record_decisions.setdefault(cid, []).append(decision)
    return by_record


def relation_drop_decision(decisions: list[dict], threshold: float) -> dict | None:
    drops = [
        decision
        for decision in decisions
        if normalize_action(decision.get("action")) == "drop" and confidence(decision) >= threshold
    ]
    if not drops:
        return None
    return max(drops, key=confidence)


def apply_relation_only(
    pred: Path,
    judgments: Path,
    output_prefix: str,
    protected_relation_labels: set[str] | None = None,
) -> tuple[Path, Path]:
    output_json = OUT_DIR / f"{output_prefix}.json"
    output_zip = OUT_DIR / f"{output_prefix}_submit.zip"
    summary = OUT_DIR / f"{output_prefix}_summary.json"
    strategy = OUT_DIR / f"{output_prefix}_strategy.json"
    strategy_obj = {
        "entity_policy": "keep_all_strict",
        "entity_drop_threshold": 1.01,
        "apply_entity_fix": False,
        "entity_fix_threshold": 1.01,
        "relation_policy": "drop_threshold",
        "relation_drop_threshold": 0.95,
        "apply_relation_fix": False,
        "protect_relation_endpoints": False,
        "protected_relation_labels": sorted(protected_relation_labels or set()),
        "noempty_fallback": True,
        "postprocess": False,
    }
    pred_records = load_json(pred)
    decisions_by_record = collect_decisions(load_jsonl(judgments))
    out = []
    stats = Counter()
    changed_records = 0
    for idx, record in enumerate(pred_records):
        before = json.dumps(record, ensure_ascii=False, sort_keys=True)
        next_record = dict(record)
        next_record["entities"] = list(record.get("entities", []) or [])
        relations = []
        record_decisions = decisions_by_record.get(idx, {})
        for rel_idx, relation in enumerate(record.get("relations", []) or []):
            decision = relation_drop_decision(
                record_decisions.get(f"R{rel_idx}", []),
                strategy_obj["relation_drop_threshold"],
            )
            if decision and str(relation.get("label", "")) in (protected_relation_labels or set()):
                stats["protected_relation_keep"] += 1
                relations.append(relation)
                continue
            if decision:
                stats["relation_drop"] += 1
                continue
            relations.append(relation)
        next_record["relations"] = relations
        if strategy_obj["noempty_fallback"] and not next_record.get("entities") and not next_record.get("relations"):
            next_record = record
            stats["noempty_fallback"] += 1
        after = json.dumps(next_record, ensure_ascii=False, sort_keys=True)
        if before != after:
            changed_records += 1
        out.append(next_record)

    dump_json(output_json, out)
    write_zip(output_json, output_zip)
    before_stats = count_stats(pred_records)
    after_stats = count_stats(out)
    summary_obj = {
        "strategy": strategy_obj,
        "input": before_stats,
        "output": after_stats,
        "delta": {
            "entities": after_stats["entities"] - before_stats["entities"],
            "relations": after_stats["relations"] - before_stats["relations"],
            "empty": after_stats["empty"] - before_stats["empty"],
            "bad_relation_endpoints": after_stats["bad_relation_endpoints"] - before_stats["bad_relation_endpoints"],
            "entity_labels": label_delta(after_stats, before_stats, "entity_labels"),
            "relation_labels": label_delta(after_stats, before_stats, "relation_labels"),
        },
        "judged_records": len(decisions_by_record),
        "changed_records": changed_records,
        "applied": dict(stats),
        "output_json": str(output_json),
        "output_zip": str(output_zip),
    }
    dump_json(strategy, strategy_obj)
    dump_json(summary, summary_obj)
    print(json.dumps(summary_obj, ensure_ascii=False, indent=2), flush=True)
    return output_json, summary


def signature(records) -> str:
    return json.dumps(
        [
            {
                "entities": rec.get("entities", []) or [],
                "relations": rec.get("relations", []) or [],
            }
            for rec in records
        ],
        ensure_ascii=False,
        sort_keys=True,
    )


def count_stats(records) -> dict:
    ent, rel = Counter(), Counter()
    empty = 0
    bad_endpoints = 0
    for rec in records:
        ents = {(e.get("text"), e.get("label")) for e in rec.get("entities", []) or []}
        if not rec.get("entities") and not rec.get("relations"):
            empty += 1
        for e in rec.get("entities", []) or []:
            ent[str(e.get("label", ""))] += 1
        for r in rec.get("relations", []) or []:
            rel[str(r.get("label", ""))] += 1
            if (r.get("head"), r.get("head_type")) not in ents or (r.get("tail"), r.get("tail_type")) not in ents:
                bad_endpoints += 1
    return {
        "records": len(records),
        "entities": sum(ent.values()),
        "relations": sum(rel.values()),
        "empty": empty,
        "bad_relation_endpoints": bad_endpoints,
        "entity_labels": dict(sorted(ent.items())),
        "relation_labels": dict(sorted(rel.items())),
    }


def label_delta(after: dict, before: dict, key: str) -> dict:
    labels = set(after.get(key, {})) | set(before.get(key, {}))
    return {
        label: after.get(key, {}).get(label, 0) - before.get(key, {}).get(label, 0)
        for label in sorted(labels)
    }


def validate_zip(json_path: Path, zip_path: Path) -> dict:
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        data = json.loads(z.read("submit.json").decode("utf-8"))
    jdata = load_json(json_path)
    if data != jdata:
        raise RuntimeError("zip submit.json differs from output json")
    stats = count_stats(data)
    stats["zip_names"] = names
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_pred", type=Path, required=True)
    parser.add_argument("--prefix", default="test_b_filter_then_union_all_relations_spanonly_augmented_ops_ent075_rel095_iterrel")
    parser.add_argument("--work_dir", type=Path, default=WORK_DIR)
    parser.add_argument("--pre_full_ops", action="store_true")
    parser.add_argument("--pre_full_prefix")
    parser.add_argument("--shards", type=int, default=5)
    parser.add_argument("--max_rounds", type=int, default=20)
    parser.add_argument("--poll_seconds", type=int, default=30)
    parser.add_argument("--protect_relation_labels", nargs="*", default=[])
    args = parser.parse_args()

    work_dir = args.work_dir
    work_dir.mkdir(parents=True, exist_ok=True)
    current = args.start_pred
    protected_relation_labels = {
        str(label).strip()
        for label in args.protect_relation_labels
        if str(label).strip()
    }
    history = []
    if args.pre_full_ops:
        before = load_json(current)
        before_sig = signature(before)
        before_stats = count_stats(before)
        print(json.dumps({"event": "pre_full_ops_start", "pred": str(current), "stats": before_stats}, ensure_ascii=False), flush=True)
        judgments = run_ops_round(current, work_dir / "pre_full_ops_ent075_rel095", args.shards, args.poll_seconds)
        output_prefix = args.pre_full_prefix or f"{args.prefix}_pre_full_ent075_rel095"
        next_json, summary = apply_full_ops(current, judgments, output_prefix)
        after = load_json(next_json)
        after_sig = signature(after)
        after_stats = count_stats(after)
        summary_obj = load_json(summary)
        pre_report = {
            "stage": "pre_full_ops_ent075_rel095",
            "input": str(current),
            "output": str(next_json),
            "summary": str(summary),
            "changed": before_sig != after_sig,
            "before": before_stats,
            "after": after_stats,
            "delta_entities": after_stats["entities"] - before_stats["entities"],
            "delta_relations": after_stats["relations"] - before_stats["relations"],
            "applied": summary_obj.get("applied", {}),
            "changed_records": summary_obj.get("changed_records"),
        }
        history.append(pre_report)
        dump_json(work_dir / "iterative_relation_ops_history.json", history)
        print(json.dumps({"event": "pre_full_ops_done", **pre_report}, ensure_ascii=False), flush=True)
        current = next_json

    for round_idx in range(1, args.max_rounds + 1):
        before = load_json(current)
        before_sig = signature(before)
        before_stats = count_stats(before)
        print(json.dumps({"event": "round_start", "round": round_idx, "pred": str(current), "stats": before_stats}, ensure_ascii=False), flush=True)
        round_dir = work_dir / f"round_{round_idx:02d}"
        judgments = run_ops_round(current, round_dir, args.shards, args.poll_seconds)
        output_prefix = f"{args.prefix}_round{round_idx:02d}"
        next_json, summary = apply_relation_only(current, judgments, output_prefix, protected_relation_labels)
        after = load_json(next_json)
        after_sig = signature(after)
        after_stats = count_stats(after)
        summary_obj = load_json(summary)
        changed = before_sig != after_sig
        round_report = {
            "round": round_idx,
            "stage": "relation_only_ops_rel095",
            "input": str(current),
            "output": str(next_json),
            "summary": str(summary),
            "changed": changed,
            "before": before_stats,
            "after": after_stats,
            "delta_entities": after_stats["entities"] - before_stats["entities"],
            "delta_relations": after_stats["relations"] - before_stats["relations"],
            "applied": summary_obj.get("applied", {}),
            "changed_records": summary_obj.get("changed_records"),
        }
        history.append(round_report)
        dump_json(work_dir / "iterative_relation_ops_history.json", history)
        print(json.dumps({"event": "round_done", **round_report}, ensure_ascii=False), flush=True)
        current = next_json
        if not changed:
            break

    final_zip = OUT_DIR / f"{args.prefix}_final_submit.zip"
    final_json = OUT_DIR / f"{args.prefix}_final.json"
    final_summary = OUT_DIR / f"{args.prefix}_final_summary.json"
    data = load_json(current)
    dump_json(final_json, data)
    with zipfile.ZipFile(final_zip, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(final_json, arcname="submit.json")
    final_stats = validate_zip(final_json, final_zip)
    final_report = {
        "start_pred": str(args.start_pred),
        "final_json": str(final_json),
        "final_zip": str(final_zip),
        "rounds": len(history),
        "history": history,
        "validation": final_stats,
    }
    dump_json(final_summary, final_report)
    print(json.dumps({"event": "final", **final_report}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
