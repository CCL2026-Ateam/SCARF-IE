"""Organizer-facing runner for the online high-score pipeline.

The runner starts from official train/test zip files, calls the configured
OpenAI-compatible API, and writes a fresh submit.json / submit.zip.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import zipfile
from collections import Counter
from copy import deepcopy
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_ROOT = PACKAGE_ROOT / "src" / "pipeline_reference"
STAGE00 = PIPELINE_ROOT / "stage_00_external_project_src"
STAGE03 = PIPELINE_ROOT / "stage_03_record_level_filter_prune_code"
STAGE04 = PIPELINE_ROOT / "stage_04_testb_filter_then_augment_code"
STAGE05 = PIPELINE_ROOT / "stage_05_ops_relation_prune_code"

DEFAULT_MEMBERS = ("secondary_k4", "main_k4", "secondary_k4_conv", "main_k4_conv")
RELATION_LABELS = ("AFF", "CON", "HAS", "LOI", "OCI", "USE")
FINAL_STRICT3_ADD_LABELS = {"AFF", "CON", "LOI", "OCI"}


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def dump_summary(path: Path, records: list[dict], extra: dict | None = None) -> None:
    ent = Counter()
    rel = Counter()
    for record in records:
        for entity in record.get("entities", []) or []:
            ent[str(entity.get("label", ""))] += 1
        for relation in record.get("relations", []) or []:
            rel[str(relation.get("label", ""))] += 1
    summary = {
        "records": len(records),
        "entities": sum(ent.values()),
        "relations": sum(rel.values()),
        "entity_labels": dict(sorted(ent.items())),
        "relation_labels": dict(sorted(rel.items())),
    }
    if extra:
        summary.update(extra)
    dump_json(path, summary)


def extract_single_json(zip_path: Path, expected_name: str | None = None):
    with zipfile.ZipFile(zip_path) as zf:
        names = [name for name in zf.namelist() if not name.endswith("/")]
        if expected_name and expected_name in names:
            name = expected_name
        elif len(names) == 1:
            name = names[0]
        else:
            raise ValueError(f"Cannot choose JSON from {zip_path}: {names}")
        return json.loads(zf.read(name).decode("utf-8"))


def write_submit_zip(zip_path: Path, submit_json: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(submit_json, arcname="submit.json")


def ensure_file(path: Path, hint: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{hint} not found: {path}")


def prepare_data(args) -> dict[str, Path]:
    data_dir = args.work_dir / "data"
    dataset_dir = args.work_dir / "dataset"
    data_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    train_records = extract_single_json(args.train_zip, "train.json")
    test_records = extract_single_json(args.test_zip, "test_B.json")

    pool_records = train_records[:800]
    dev_records = train_records[800:1000]
    paths = {
        "pool": data_dir / "pool.json",
        "dev": data_dir / "dev.json",
        "original1000": data_dir / "original_1000_pool.json",
        "test_b": dataset_dir / "test_B.json",
    }
    dump_json(paths["pool"], pool_records)
    dump_json(paths["dev"], dev_records)
    dump_json(paths["original1000"], train_records)
    dump_json(paths["test_b"], test_records)
    dump_summary(data_dir / "prepare_summary.json", test_records, {
        "train_records": len(train_records),
        "pool_records": len(pool_records),
        "dev_records": len(dev_records),
        "train_zip": str(args.train_zip),
        "test_zip": str(args.test_zip),
    })
    return paths


def base_env(args, pool_path: Path | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "BASE_URL": args.base_url.rstrip("/"),
            "MODEL_MAIN": args.model_main,
            "MODEL_CHEAP": args.model_cheap,
            "MODEL_SECONDARY": args.model_secondary,
            "TEMPERATURE": str(args.temperature),
            "TIMEOUT": str(args.timeout),
            "MAX_RETRIES": str(args.max_retries),
            "CONCURRENCY": str(args.record_concurrency),
            "CACHE_DIR": str(args.work_dir / "cache"),
            "OPUS_BASE_URL": args.base_url.rstrip("/"),
            "OPUS_MODEL": args.model_main,
            "OPUS_TIMEOUT": str(args.timeout),
            "OPUS_RETRIES": str(args.max_retries),
        }
    )
    if args.api_key:
        env["API_KEY"] = args.api_key
        env["OPUS_API_KEY"] = args.api_key
    if args.api_key_env and os.getenv(args.api_key_env):
        env["API_KEY"] = os.environ[args.api_key_env]
        env["OPUS_API_KEY"] = os.environ[args.api_key_env]
    if pool_path:
        env["POOL_PATH"] = str(pool_path)
    return env


def run_cmd(cmd: list[str], env: dict[str, str], cwd: Path, timeout: int | None = None) -> None:
    printable = [str(item) for item in cmd]
    print("[RUN]", " ".join(printable), flush=True)
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True, timeout=timeout)


def smoke_api(args) -> None:
    env = base_env(args)
    for model_alias in ("main", "cheap", "secondary"):
        run_cmd([sys.executable, str(STAGE00 / "client.py"), model_alias], env=env, cwd=PIPELINE_ROOT)


def member_suffixes(args) -> list[str]:
    if not args.member:
        return list(DEFAULT_MEMBERS)
    suffixes = []
    for item in args.member:
        parts = item.split(":")
        if len(parts) != 3:
            raise ValueError(f"bad --member {item!r}; expected MODEL:CONVENTIONS:SUFFIX")
        suffixes.append(parts[2].strip())
    return suffixes


def run_online_ensemble(
    args,
    *,
    input_path: Path,
    pool_path: Path,
    tag: str,
    member_out_dir: Path,
    output_dir: Path,
    submit_name: str,
    cache_dir: Path,
    skip_run: bool | None = None,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    member_out_dir.mkdir(parents=True, exist_ok=True)
    env = base_env(args, pool_path=pool_path)
    submit_json = output_dir / f"{submit_name}.json"
    submit_zip = output_dir / f"{submit_name}.zip"
    ensemble_json = output_dir / f"{tag}_ensemble.json"
    cmd = [
        sys.executable,
        str(STAGE00 / "run_online_best_submit.py"),
        "--input",
        str(input_path),
        "--tag",
        tag,
        "--kshot",
        str(args.kshot),
        "--retriever",
        args.retriever,
        "--record_concurrency",
        str(args.record_concurrency),
        "--member_concurrency",
        str(args.member_concurrency),
        "--member_timeout",
        str(args.member_timeout),
        "--max_tokens",
        str(args.max_tokens),
        "--member_out_dir",
        str(member_out_dir),
        "--ensemble_out",
        str(ensemble_json),
        "--submit_out",
        str(submit_json),
        "--zip_out",
        str(submit_zip),
        "--pool",
        str(pool_path),
        "--cache_dir",
        str(cache_dir),
    ]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    for member in args.member:
        cmd += ["--member", member]
    if args.no_cache:
        cmd.append("--no_cache")
    should_skip = args.skip_existing if skip_run is None else skip_run
    if should_skip:
        cmd.append("--skip_run")
    run_cmd(cmd, env=env, cwd=PIPELINE_ROOT)
    return {
        "ensemble": ensemble_json,
        "json": submit_json,
        "zip": submit_zip,
        "submit_json": submit_json,
        "submit_zip": submit_zip,
    }


def run_base_ensemble(args, paths: dict[str, Path]) -> dict[str, Path]:
    return run_online_ensemble(
        args,
        input_path=paths["test_b"],
        pool_path=paths["pool"],
        tag=args.base_tag,
        member_out_dir=args.work_dir / "member_outputs",
        output_dir=args.output_dir,
        submit_name="base_submit",
        cache_dir=args.work_dir / "cache" / "base",
    )


def ekey(entity: dict) -> tuple[int, int, str]:
    return int(entity["start"]), int(entity["end"]), str(entity["label"])


def rkey(relation: dict) -> tuple[int, int, str, int, int, str, str]:
    return (
        int(relation["head_start"]),
        int(relation["head_end"]),
        str(relation["head_type"]),
        int(relation["tail_start"]),
        int(relation["tail_end"]),
        str(relation["tail_type"]),
        str(relation["label"]),
    )


def sort_record(record: dict) -> dict:
    out = dict(record)
    out["entities"] = sorted(
        out.get("entities", []) or [],
        key=lambda e: (int(e["start"]), int(e["end"]), str(e["label"])),
    )
    out["relations"] = sorted(
        out.get("relations", []) or [],
        key=lambda r: (
            int(r["head_start"]),
            int(r["head_end"]),
            int(r["tail_start"]),
            int(r["tail_end"]),
            str(r["label"]),
        ),
    )
    return out


def vote_records(pred_lists: list[list[dict]], ent_vote_min: int, rel_vote_min: int) -> list[dict]:
    if not pred_lists:
        raise ValueError("no prediction lists to vote")
    lengths = {len(records) for records in pred_lists}
    if len(lengths) != 1:
        raise ValueError(f"member output lengths differ: {sorted(lengths)}")
    out = []
    for i in range(len(pred_lists[0])):
        ent_votes = Counter()
        rel_votes = Counter()
        ent_obj = {}
        rel_obj = {}
        text = pred_lists[0][i]["text"]
        for records in pred_lists:
            rec = records[i]
            seen_entities = set()
            for entity in rec.get("entities", []) or []:
                key = ekey(entity)
                if key not in seen_entities:
                    seen_entities.add(key)
                    ent_votes[key] += 1
                    ent_obj[key] = {
                        "start": int(entity["start"]),
                        "end": int(entity["end"]),
                        "text": str(entity["text"]),
                        "label": str(entity["label"]),
                    }
            seen_relations = set()
            for relation in rec.get("relations", []) or []:
                key = rkey(relation)
                if key not in seen_relations:
                    seen_relations.add(key)
                    rel_votes[key] += 1
                    rel_obj[key] = {
                        "head": str(relation["head"]),
                        "head_start": int(relation["head_start"]),
                        "head_end": int(relation["head_end"]),
                        "head_type": str(relation["head_type"]),
                        "tail": str(relation["tail"]),
                        "tail_start": int(relation["tail_start"]),
                        "tail_end": int(relation["tail_end"]),
                        "tail_type": str(relation["tail_type"]),
                        "label": str(relation["label"]),
                    }
        entities = [ent_obj[key] for key, votes in ent_votes.items() if votes >= ent_vote_min]
        entity_keys = set(ekey(entity) for entity in entities)
        relations = []
        for key, votes in rel_votes.items():
            if votes < rel_vote_min:
                continue
            relation = rel_obj[key]
            if (
                (relation["head_start"], relation["head_end"], relation["head_type"]) in entity_keys
                and (relation["tail_start"], relation["tail_end"], relation["tail_type"]) in entity_keys
            ):
                relations.append(relation)
        out.append(sort_record({"text": text, "entities": entities, "relations": relations}))
    return out


def member_output_paths(args, tag: str, member_out_dir: Path) -> list[Path]:
    return [member_out_dir / f"{tag}_{suffix}.json" for suffix in member_suffixes(args)]


def build_union_pool(args) -> dict[str, Path]:
    member_dir = args.work_dir / "member_outputs"
    paths = member_output_paths(args, args.base_tag, member_dir)
    for path in paths:
        ensure_file(path, "base member output")
    records = vote_records([load_json(path) for path in paths], ent_vote_min=1, rel_vote_min=1)
    out_json = args.work_dir / "union_pool.json"
    out_zip = args.work_dir / "union_pool.zip"
    dump_json(out_json, records)
    write_submit_zip(out_zip, out_json)
    dump_summary(args.work_dir / "union_pool_summary.json", records, {
        "entity_vote_min": 1,
        "relation_vote_min": 1,
        "members": [str(path) for path in paths],
    })
    print(f"[OK] union pool -> {out_json}", flush=True)
    return {"json": out_json, "zip": out_zip}


def run_original1000_ensemble(args, paths: dict[str, Path]) -> dict[str, Path]:
    return run_online_ensemble(
        args,
        input_path=paths["test_b"],
        pool_path=paths["original1000"],
        tag=args.original1000_tag,
        member_out_dir=args.work_dir / "original1000_member_outputs",
        output_dir=args.work_dir / "original1000_outputs",
        submit_name="original1000_submit",
        cache_dir=args.work_dir / "cache" / "original1000",
    )


def build_original1000_strict3(args) -> dict[str, Path]:
    member_dir = args.work_dir / "original1000_member_outputs"
    paths = member_output_paths(args, args.original1000_tag, member_dir)
    for path in paths:
        ensure_file(path, "original1000 member output")
    strict3 = vote_records([load_json(path) for path in paths], ent_vote_min=1, rel_vote_min=3)
    raw_json = args.work_dir / "original1000_strict3.json"
    raw_zip = args.work_dir / "original1000_strict3.zip"
    filtered_json = args.work_dir / "original1000_strict3_relfilter_conservative.json"
    filtered_zip = args.work_dir / "original1000_strict3_relfilter_conservative.zip"
    dump_json(raw_json, strict3)
    write_submit_zip(raw_zip, raw_json)
    dump_summary(args.work_dir / "original1000_strict3_summary.json", strict3, {
        "entity_vote_min": 1,
        "relation_vote_min": 3,
        "members": [str(path) for path in paths],
    })
    run_cmd(
        [
            sys.executable,
            str(STAGE00 / "relation_sanity_filter.py"),
            "--input",
            str(raw_zip),
            "--output",
            str(filtered_json),
            "--zip_out",
            str(filtered_zip),
            "--summary",
            str(args.work_dir / "original1000_strict3_relfilter_conservative_summary.json"),
            "--mode",
            "conservative",
        ],
        env=base_env(args),
        cwd=PIPELINE_ROOT,
    )
    return {"json": filtered_json, "zip": filtered_zip, "raw_json": raw_json, "raw_zip": raw_zip}


def run_augmentation(args, base_json: Path | None = None, pool_json: Path | None = None) -> dict[str, Path]:
    base_json = base_json or args.output_dir / "base_submit.json"
    pool_json = pool_json or args.work_dir / "union_pool.json"
    ensure_file(base_json, "base submit json")
    ensure_file(pool_json, "union pool json")
    deliver_dir = args.work_dir / "augment_outputs"
    work_out = args.work_dir / "augment_work"
    output_prefix = "augmented"
    cmd = [
        sys.executable,
        str(STAGE04 / "run_filter_then_union_testb.py"),
        "--stage",
        "all",
        "--base",
        str(base_json),
        "--pool",
        str(pool_json),
        "--work_out",
        str(work_out),
        "--deliver_out",
        str(deliver_dir),
        "--solution_src",
        str(STAGE00),
        "--output_prefix",
        output_prefix,
        "--entity_dedup",
        "span_only",
        "--relation_labels",
        *RELATION_LABELS,
        "--workers",
        str(args.augment_workers),
        "--entity_batch_items",
        str(args.augment_entity_batch_items),
        "--relation_batch_items",
        str(args.augment_relation_batch_items),
        "--model",
        args.model_main,
        "--base_url",
        args.base_url.rstrip("/"),
        "--api_key_env",
        "API_KEY",
        "--timeout",
        str(args.timeout),
        "--retries",
        str(args.max_retries),
        "--max_tokens",
        str(args.max_tokens),
        "--no_label_examples",
    ]
    if args.augment_max_batches:
        cmd += ["--max_batches", str(args.augment_max_batches)]
    if args.keep_raw:
        cmd.append("--keep_raw")
    run_cmd(cmd, env=base_env(args), cwd=PIPELINE_ROOT)
    out_json = deliver_dir / f"{output_prefix}.json"
    out_zip = deliver_dir / f"{output_prefix}_submit.zip"
    ensure_file(out_json, "augmented output")
    return {"json": out_json, "zip": out_zip, "summary": deliver_dir / f"{output_prefix}_summary.json"}


def record_ranges(n_records: int, shards: int) -> list[tuple[int, int]]:
    width = (n_records + shards - 1) // shards
    return [(start, min(n_records, start + width)) for start in range(0, n_records, width)]


def run_ops_apply(
    args,
    *,
    pred_json: Path,
    name: str,
    relation_only: bool = False,
) -> dict[str, Path]:
    ensure_file(pred_json, "OPS input prediction")
    records = load_json(pred_json)
    out_dir = args.work_dir / "ops_audit" / name
    env = base_env(args)
    ranges = record_ranges(len(records), args.ops_shards)
    for shard_index, (start, end) in enumerate(ranges):
        run_cmd(
            [
                sys.executable,
                str(STAGE05 / "run_ops_audit_sharded.py"),
                "run-shard",
                "--pred",
                str(pred_json),
                "--out_dir",
                str(out_dir),
                "--shards",
                str(args.ops_shards),
                "--shard_index",
                str(shard_index),
                "--start",
                str(start),
                "--end",
                str(end),
                "--model",
                args.model_main,
                "--base_url",
                args.base_url.rstrip("/"),
                "--api_key_env",
                "OPUS_API_KEY",
                "--timeout",
                str(args.timeout),
                "--retries",
                str(args.max_retries),
                "--max_tokens",
                str(args.max_tokens),
                "--batch_items",
                str(args.ops_batch_items),
            ],
            env=env,
            cwd=PIPELINE_ROOT,
        )
    merged = out_dir / f"{name}_merged.jsonl"
    run_cmd(
        [
            sys.executable,
            str(STAGE05 / "run_ops_audit_sharded.py"),
            "merge",
            "--pred",
            str(pred_json),
            "--out_dir",
            str(out_dir),
            "--shards",
            str(args.ops_shards),
            "--merged",
            str(merged),
        ],
        env=env,
        cwd=PIPELINE_ROOT,
    )
    output_json = args.work_dir / f"{name}.json"
    output_zip = args.work_dir / f"{name}.zip"
    summary = args.work_dir / f"{name}_summary.json"
    strategy = args.work_dir / f"{name}_strategy.json"
    apply_cmd = [
        sys.executable,
        str(STAGE03 / "apply_entity_pair_strategy.py"),
        "--pred",
        str(pred_json),
        "--judgments",
        str(merged),
        "--solution_src",
        str(STAGE00),
        "--output_json",
        str(output_json),
        "--output_zip",
        str(output_zip),
        "--summary",
        str(summary),
        "--preset",
        "safe_anydrop_rel098",
        "--relation_drop_threshold",
        str(args.ops_relation_drop_threshold),
        "--apply_relation_fix",
        "false",
        "--noempty_fallback",
        "true",
        "--write_strategy_json",
        str(strategy),
    ]
    if relation_only:
        apply_cmd += [
            "--entity_policy",
            "keep_all",
            "--apply_entity_fix",
            "false",
            "--relation_policy",
            "drop_threshold",
            "--protect_relation_endpoints",
            "true",
        ]
    else:
        apply_cmd += [
            "--entity_policy",
            "drop_threshold",
            "--entity_drop_threshold",
            str(args.ops_entity_drop_threshold),
            "--apply_entity_fix",
            "true",
            "--entity_fix_threshold",
            "0.0",
            "--relation_policy",
            "drop_threshold",
            "--protect_relation_endpoints",
            "false",
        ]
    run_cmd(apply_cmd, env=env, cwd=PIPELINE_ROOT)
    return {"json": output_json, "zip": output_zip, "summary": summary, "judgments": merged, "strategy": strategy}


def run_relation_prune_rounds(args, start_json: Path) -> dict[str, Path]:
    current = start_json
    latest = {"json": current, "zip": current.with_suffix(".zip")}
    for round_index in range(1, args.relation_prune_rounds + 1):
        latest = run_ops_apply(
            args,
            pred_json=current,
            name=f"relation_prune_round{round_index:02d}",
            relation_only=True,
        )
        current = latest["json"]
    return latest


def relation_endpoint_key(relation: dict, side: str) -> tuple[int, int, str]:
    return int(relation[f"{side}_start"]), int(relation[f"{side}_end"]), str(relation[f"{side}_type"])


def relation_has_existing_endpoints(relation: dict, entity_keys: set[tuple[int, int, str]]) -> bool:
    return relation_endpoint_key(relation, "head") in entity_keys and relation_endpoint_key(relation, "tail") in entity_keys


def final_synthesis(
    args,
    *,
    entity_source_json: Path,
    relation_source_json: Path,
    strict3_json: Path,
    has_source_json: Path | None = None,
) -> dict[str, Path]:
    ensure_file(entity_source_json, "final entity source")
    ensure_file(relation_source_json, "final relation source")
    ensure_file(strict3_json, "strict3 relation source")
    entities_records = load_json(entity_source_json)
    relation_records = load_json(relation_source_json)
    strict3_records = load_json(strict3_json)
    has_records = load_json(has_source_json) if has_source_json and has_source_json.exists() else entities_records
    lengths = {len(entities_records), len(relation_records), len(strict3_records), len(has_records)}
    if len(lengths) != 1:
        raise ValueError(f"final synthesis input lengths differ: {sorted(lengths)}")

    out = []
    stats = Counter()
    for entity_rec, relation_rec, strict_rec, has_rec in zip(
        entities_records, relation_records, strict3_records, has_records
    ):
        row = {"text": entity_rec["text"], "entities": deepcopy(entity_rec.get("entities", []) or []), "relations": []}
        entity_keys = {ekey(entity) for entity in row["entities"]}
        rel_map: dict[tuple, dict] = {}

        for relation in relation_rec.get("relations", []) or []:
            if str(relation.get("label")) == "USE":
                stats["drop_use"] += 1
                continue
            if relation_has_existing_endpoints(relation, entity_keys):
                rel_map[rkey(relation)] = deepcopy(relation)

        for relation in has_rec.get("relations", []) or []:
            if str(relation.get("label")) != "HAS":
                continue
            if relation_has_existing_endpoints(relation, entity_keys):
                key = rkey(relation)
                if key not in rel_map:
                    stats["add_protected_has"] += 1
                rel_map[key] = deepcopy(relation)

        for relation in strict_rec.get("relations", []) or []:
            label = str(relation.get("label"))
            if label not in FINAL_STRICT3_ADD_LABELS:
                continue
            if relation_has_existing_endpoints(relation, entity_keys):
                key = rkey(relation)
                if key not in rel_map:
                    stats["add_strict3_relation"] += 1
                rel_map[key] = deepcopy(relation)

        row["relations"] = list(rel_map.values())
        out.append(sort_record(row))

    out_json = args.output_dir / "submit.json"
    out_zip = args.output_dir / "submit.zip"
    summary = args.output_dir / "submit_summary.json"
    dump_json(out_json, out)
    write_submit_zip(out_zip, out_json)
    dump_summary(
        summary,
        out,
        {
            "entity_source_json": str(entity_source_json),
            "relation_source_json": str(relation_source_json),
            "has_source_json": str(has_source_json) if has_source_json else None,
            "strict3_json": str(strict3_json),
            "rules": {
                "drop_relation_label": "USE",
                "preserve_has_from": "has_source_json",
                "add_original1000_strict3_labels": sorted(FINAL_STRICT3_ADD_LABELS),
                "require_relation_endpoints_in_final_entities": True,
            },
            "changes": dict(sorted(stats.items())),
        },
    )
    print(f"[OK] final submit -> {out_zip}", flush=True)
    return {"json": out_json, "zip": out_zip, "summary": summary}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the CCL2026-MGBIE high-score pipeline from official data.")
    parser.add_argument("--train-zip", type=Path, default=PACKAGE_ROOT.parent / "train.zip")
    parser.add_argument("--test-zip", type=Path, default=PACKAGE_ROOT.parent / "test_B.zip")
    parser.add_argument("--work-dir", type=Path, default=PACKAGE_ROOT / "work" / "from_scratch")
    parser.add_argument("--output-dir", type=Path, default=PACKAGE_ROOT / "outputs" / "from_scratch")
    parser.add_argument(
        "--stage",
        choices=[
            "prepare",
            "smoke-api",
            "base-ensemble",
            "union-pool",
            "original1000-ensemble",
            "original1000-strict3",
            "augment",
            "ops2",
            "relation-prune",
            "final",
            "full",
        ],
        default="full",
    )
    parser.add_argument("--api-key", default=None, help="API key; prefer --api-key-env to avoid shell history leaks")
    parser.add_argument("--api-key-env", default="API_KEY")
    parser.add_argument("--base-url", default="https://zjapi.com")
    parser.add_argument("--model-main", default="gpt-5.4")
    parser.add_argument("--model-cheap", default="gpt-5.4-mini")
    parser.add_argument("--model-secondary", default="gpt-5.5-openai-compact")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--base-tag", default="test_b_4member")
    parser.add_argument("--original1000-tag", default="test_b_original1000_4member")
    parser.add_argument("--kshot", type=int, default=4)
    parser.add_argument("--retriever", choices=["lexical", "semantic"], default="lexical")
    parser.add_argument("--record-concurrency", type=int, default=16)
    parser.add_argument("--member-concurrency", type=int, default=4)
    parser.add_argument("--member-timeout", type=int, default=3600)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--member", action="append", default=[], help="optional MODEL:CONVENTIONS:SUFFIX override")
    parser.add_argument("--limit", type=int, default=None, help="smoke-test on first N records")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--augment-workers", type=int, default=4)
    parser.add_argument("--augment-entity-batch-items", type=int, default=20)
    parser.add_argument("--augment-relation-batch-items", type=int, default=20)
    parser.add_argument("--augment-max-batches", type=int, default=None)
    parser.add_argument("--ops-shards", type=int, default=8)
    parser.add_argument("--ops-batch-items", type=int, default=999)
    parser.add_argument("--ops-entity-drop-threshold", type=float, default=0.75)
    parser.add_argument("--ops-relation-drop-threshold", type=float, default=0.95)
    parser.add_argument("--relation-prune-rounds", type=int, default=12)
    parser.add_argument("--keep-raw", action="store_true")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    args.work_dir = args.work_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.train_zip = args.train_zip.resolve()
    args.test_zip = args.test_zip.resolve()

    if args.stage == "smoke-api":
        smoke_api(args)
        return

    paths = prepare_data(args)
    print(
        "[OK] prepared data: "
        f"pool={paths['pool']} dev={paths['dev']} original1000={paths['original1000']} test_b={paths['test_b']}",
        flush=True,
    )
    if args.stage == "prepare":
        return

    base_paths = {"json": args.output_dir / "base_submit.json", "zip": args.output_dir / "base_submit.zip"}
    union_paths = {"json": args.work_dir / "union_pool.json", "zip": args.work_dir / "union_pool.zip"}
    original_paths = {"json": args.work_dir / "original1000_outputs" / "original1000_submit.json"}
    strict3_paths = {"json": args.work_dir / "original1000_strict3_relfilter_conservative.json"}
    augment_paths = {"json": args.work_dir / "augment_outputs" / "augmented.json"}
    ops2_paths = {"json": args.work_dir / "ops2.json", "zip": args.work_dir / "ops2.zip"}

    if args.stage in {"base-ensemble", "full"}:
        base_paths = run_base_ensemble(args, paths)
        if args.stage == "base-ensemble":
            return

    if args.stage in {"union-pool", "full"}:
        union_paths = build_union_pool(args)
        if args.stage == "union-pool":
            return

    if args.stage in {"original1000-ensemble", "full"}:
        original_paths = run_original1000_ensemble(args, paths)
        if args.stage == "original1000-ensemble":
            return

    if args.stage in {"original1000-strict3", "full"}:
        strict3_paths = build_original1000_strict3(args)
        if args.stage == "original1000-strict3":
            return

    if args.stage in {"augment", "full"}:
        augment_paths = run_augmentation(args, base_json=base_paths["json"], pool_json=union_paths["json"])
        if args.stage == "augment":
            return

    if args.stage in {"ops2", "full"}:
        ops2_paths = run_ops_apply(args, pred_json=augment_paths["json"], name="ops2", relation_only=False)
        if args.stage == "ops2":
            return

    if args.relation_prune_rounds > 0:
        relation_prune_paths = {
            "json": args.work_dir / f"relation_prune_round{args.relation_prune_rounds:02d}.json",
            "zip": args.work_dir / f"relation_prune_round{args.relation_prune_rounds:02d}.zip",
        }
    else:
        relation_prune_paths = ops2_paths
    if args.stage in {"relation-prune", "full"}:
        relation_prune_paths = run_relation_prune_rounds(args, ops2_paths["json"])
        if args.stage == "relation-prune":
            return

    if args.stage in {"final", "full"}:
        final_synthesis(
            args,
            entity_source_json=relation_prune_paths["json"],
            relation_source_json=ops2_paths["json"],
            has_source_json=relation_prune_paths["json"],
            strict3_json=strict3_paths["json"],
        )
        return


if __name__ == "__main__":
    main()
