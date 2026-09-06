"""Run record-level online filter validation on saved 10x Test-B folds.

Default mode follows the requested one-to-one setup:
five selected validation folds are paired with five candidate routes. The script
first removes relation type triples not seen in that fold's training pool, then
optionally runs the online Opus auditor, applies the current best record-level
filter strategy, and reports score / TP / FP / FN before and after filtering.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[2]
PROJECT = Path(__file__).resolve().parent
PROJECT_OUT = PROJECT / "outputs" / "record_level_filter_5x5"
S100_T10 = PROJECT / "outputs" / "s100_t10"
SOLUTION = Path(os.getenv("SOLUTION_ROOT", "K:/\u6d69\u7136/CCL/solution/solution"))
SOLUTION_SRC = SOLUTION / "src"
FILTER_WORKSPACE = Path(
    "C:/Users/Zzl410410/Documents/Codex/2026-06-26/"
    "019ef541-55e7-7af1-8f34-0927a521fe82"
)
APPLY_STRATEGY = FILTER_WORKSPACE / "work" / "apply_entity_pair_strategy.py"
JUDGE_SCRIPT = WORKSPACE / "work" / "test_b_original1000_4member" / "src" / "judge_json_items_opus.py"

sys.path.insert(0, str(SOLUTION_SRC))
from common import dump_json, evaluate, load_json  # noqa: E402
from ensemble import ensemble  # noqa: E402
from relation_sanity_filter import entity_span_map, fill_types, type_pair_rank  # noqa: E402


MEMBER_NAMES = ["main_k4", "main_k4_conv", "secondary_k4", "secondary_k4_conv"]
BEST_ENTITY_THRESHOLDS = {"ABS": 2, "BIS": 2, "CHR": 2, "CROP": 2, "CROSS": 4, "GST": 2, "MRK": 4, "QTL": 2}
BEST_RELATION_THRESHOLDS = {"AFF": 3, "CON": 1, "HAS": 3, "LOI": 2, "OCI": 2, "USE": 5}
SEARCHED_ENTITY_THRESHOLDS = {
    "ABS": 2,
    "BIS": 2,
    "CHR": 2,
    "CROP": 2,
    "CROSS": 4,
    "GST": 3,
    "MRK": 2,
    "QTL": 4,
    "BM": 3,
    "GENE": 2,
    "TRT": 3,
    "VAR": 3,
}
SEARCHED_RELATION_THRESHOLDS = {"AFF": 2, "CON": 2, "HAS": 2, "LOI": 2, "OCI": 1, "USE": 4}

DEFAULT_TRIALS = [0, 1, 4, 6, 7]
DEFAULT_ROUTES = [
    "majority",
    "best",
    "union",
    "strict3",
    "highscore_plus_light_drop_blocked_loi",
    "highscore_plus_light_add_has_aff1",
]


def rel_key(rel):
    return (
        int(rel["head_start"]),
        int(rel["head_end"]),
        str(rel["head_type"]),
        int(rel["tail_start"]),
        int(rel["tail_end"]),
        str(rel["tail_type"]),
        str(rel["label"]),
    )


def endpoint_key(rel, side):
    return (int(rel[f"{side}_start"]), int(rel[f"{side}_end"]), str(rel[f"{side}_type"]))


def entity_keys(record):
    return {(int(e["start"]), int(e["end"]), str(e["label"])) for e in record.get("entities", []) or []}


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


def fill_empty(rows, fallback_rows):
    out = []
    for i, row in enumerate(rows):
        out.append(row if row.get("entities") or row.get("relations") else fallback_rows[i])
    return out


def build_core_routes(members):
    best = ensemble(
        members,
        vote_min=3,
        rel_vote_min=2,
        ent_label_thresholds=BEST_ENTITY_THRESHOLDS,
        rel_label_thresholds=BEST_RELATION_THRESHOLDS,
    )
    best_noempty = fill_empty(best, members[2])
    majority = ensemble(members, vote_min=2, rel_vote_min=2)
    strict3 = ensemble(members, vote_min=3, rel_vote_min=3)
    union = ensemble(members, vote_min=1, rel_vote_min=1)
    searched = ensemble(
        members,
        vote_min=3,
        rel_vote_min=2,
        ent_label_thresholds=SEARCHED_ENTITY_THRESHOLDS,
        rel_label_thresholds=SEARCHED_RELATION_THRESHOLDS,
    )
    return {
        "majority": majority,
        "best": best,
        "strict3": strict3,
        "union": union,
        "best_noempty": best_noempty,
        "searched_none": searched,
    }


def drop_blocked_loi(record):
    spans = entity_span_map(record)
    out = dict(record)
    out["entities"] = list(record.get("entities", []) or [])
    kept = []
    for rel in record.get("relations", []) or []:
        filled = fill_types(rel, spans)
        blocked = type_pair_rank(str(filled["label"]), str(filled["head_type"]), str(filled["tail_type"])) == "blocked"
        if str(rel.get("label")) == "LOI" and blocked:
            continue
        kept.append(rel)
    out["relations"] = kept
    out["relations"] = sorted_relations(out["relations"])
    return out


def add_agreed_relations(base_record, sources, labels):
    labels = set(labels)
    base_keys = {rel_key(r) for r in base_record.get("relations", []) or []}
    ents = entity_keys(base_record)
    source_maps = [
        {rel_key(r): r for r in source.get("relations", []) or [] if str(r.get("label")) in labels}
        for source in sources
    ]
    common = set(source_maps[0]) if source_maps else set()
    for source_map in source_maps[1:]:
        common &= set(source_map)

    additions = []
    seen = set(base_keys)
    by_label = Counter()
    for key in sorted(common):
        if key in seen:
            continue
        rel = source_maps[0][key]
        label = str(rel["label"])
        if by_label[label] >= 1 or len(additions) >= 1:
            continue
        if endpoint_key(rel, "head") not in ents or endpoint_key(rel, "tail") not in ents:
            continue
        additions.append(rel)
        seen.add(key)
        by_label[label] += 1

    out = dict(base_record)
    out["entities"] = list(base_record.get("entities", []) or [])
    out["relations"] = sorted_relations(list(base_record.get("relations", []) or []) + additions)
    return out


def build_route(route, members):
    core = build_core_routes(members)
    if route in core:
        return core[route]
    if route == "highscore_plus_light_drop_blocked_loi":
        return [drop_blocked_loi(row) for row in core["best_noempty"]]
    if route == "highscore_plus_light_add_has_aff1":
        return [
            add_agreed_relations(base, [maj, sea], ["HAS", "AFF"])
            for base, maj, sea in zip(core["best_noempty"], core["majority"], core["searched_none"])
        ]
    raise ValueError(f"unknown route: {route}")


def load_members(trial):
    method_dir = S100_T10 / f"trial_{trial:02d}" / "methods" / "augmented_clean"
    return [load_json(method_dir / f"{name}.json") for name in MEMBER_NAMES]


def relation_type_triples(pool_records):
    triples = set()
    for record in pool_records:
        for rel in record.get("relations", []) or []:
            label = rel.get("label")
            head_type = rel.get("head_type")
            tail_type = rel.get("tail_type")
            if label and head_type and tail_type:
                triples.add((str(label), str(head_type), str(tail_type)))
    return triples


def prune_unseen_relation_types(records, seen_triples):
    out = []
    dropped = Counter()
    for record in records:
        row = dict(record)
        row["entities"] = list(record.get("entities", []) or [])
        kept = []
        for rel in record.get("relations", []) or []:
            triple = (str(rel.get("label")), str(rel.get("head_type")), str(rel.get("tail_type")))
            if triple in seen_triples:
                kept.append(rel)
            else:
                dropped["relations"] += 1
                dropped["by_triple:" + "|".join(triple)] += 1
        row["relations"] = kept
        out.append(row)
    return out, dropped


def metrics_row(gold, pred):
    m = evaluate(gold, pred)
    return {
        "total_score": m["total_score"],
        "score_ner": m["score_ner"],
        "score_re": m["score_re"],
        "entity": m["entity"],
        "relation": m["relation"],
        "entities": sum(len(r.get("entities", []) or []) for r in pred),
        "relations": sum(len(r.get("relations", []) or []) for r in pred),
        "empty": sum(1 for r in pred if not r.get("entities") and not r.get("relations")),
    }


def expected_judgment_rows(records):
    return sum(1 for r in records if r.get("entities") or r.get("relations"))


def count_judgment_rows(path):
    if not path.exists():
        return 0
    seen = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            seen.add((row.get("record_index"), row.get("batch_index")))
    return len(seen)


def pair_output_dir(trial, route):
    return PROJECT_OUT / "pairs" / f"trial_{trial:02d}__{route}"


def prepare_pair(trial, route):
    out_dir = pair_output_dir(trial, route)
    out_dir.mkdir(parents=True, exist_ok=True)
    gold = load_json(S100_T10 / f"trial_{trial:02d}" / "gold.json")
    pool = load_json(S100_T10 / f"trial_{trial:02d}" / "methods" / "augmented_clean" / "pool.json")
    members = load_members(trial)
    base = build_route(route, members)
    pruned, dropped = prune_unseen_relation_types(base, relation_type_triples(pool))
    dump_json(out_dir / "base_pred.json", base)
    dump_json(out_dir / "prune_unseen_pred.json", pruned)
    summary = {
        "trial": trial,
        "route": route,
        "records": len(gold),
        "base": metrics_row(gold, base),
        "prune_unseen": metrics_row(gold, pruned),
        "prune_unseen_dropped": dict(dropped),
        "expected_judgment_rows": expected_judgment_rows(pruned),
    }
    dump_json(out_dir / "prepare_summary.json", summary)
    return out_dir, summary


def judgment_path(out_dir):
    return out_dir / "judgments" / "prune_unseen_pred.opus_item_judgments.jsonl"


def run_online_judge(out_dir, args, worker_index):
    jpath = judgment_path(out_dir)
    records = load_json(out_dir / "prune_unseen_pred.json")
    expected = expected_judgment_rows(records)
    if count_judgment_rows(jpath) >= expected:
        return {"status": "skipped_complete", "judgment_rows": expected, "path": str(jpath)}

    api_envs = [x for x in args.api_key_envs if x]
    api_env = api_envs[worker_index % len(api_envs)] if api_envs else "OPUS_API_KEY"
    env = os.environ.copy()
    if not env.get(api_env):
        fallback = env.get("OPUS_API_KEY") or env.get("API_KEY")
        if fallback:
            env[api_env] = fallback
    cmd = [
        sys.executable,
        str(JUDGE_SCRIPT),
        "--inputs",
        str(out_dir / "prune_unseen_pred.json"),
        "--output_dir",
        str(out_dir / "judgments"),
        "--model",
        args.model,
        "--base_url",
        args.base_url,
        "--api_key_env",
        api_env,
        "--timeout",
        str(args.timeout),
        "--retries",
        str(args.retries),
        "--max_tokens",
        str(args.max_tokens),
        "--batch_items",
        str(args.batch_items),
    ]
    if args.judge_max_records is not None:
        cmd += ["--max_records", str(args.judge_max_records)]
    log = out_dir / "online_judge.stdout.log"
    err = out_dir / "online_judge.stderr.log"
    with open(log, "a", encoding="utf-8") as lf, open(err, "a", encoding="utf-8") as ef:
        subprocess.run(cmd, check=True, cwd=str(WORKSPACE), env=env, stdout=lf, stderr=ef)
    return {"status": "ran", "judgment_rows": count_judgment_rows(jpath), "path": str(jpath)}


def apply_record_filter(out_dir, allow_partial=False):
    jpath = judgment_path(out_dir)
    if not jpath.exists():
        return None
    records = load_json(out_dir / "prune_unseen_pred.json")
    expected = expected_judgment_rows(records)
    if not allow_partial and count_judgment_rows(jpath) < expected:
        return None
    output_json = out_dir / "filtered_pred.json"
    output_zip = out_dir / "filtered_pred.zip"
    summary_json = out_dir / "filter_summary.json"
    strategy_json = out_dir / "filter_strategy.json"
    cmd = [
        sys.executable,
        str(APPLY_STRATEGY),
        "--pred",
        str(out_dir / "prune_unseen_pred.json"),
        "--judgments",
        str(jpath),
        "--solution_src",
        str(SOLUTION_SRC),
        "--output_json",
        str(output_json),
        "--output_zip",
        str(output_zip),
        "--summary",
        str(summary_json),
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
        str(strategy_json),
    ]
    subprocess.run(cmd, check=True, cwd=str(FILTER_WORKSPACE), capture_output=True, text=True)
    return output_json


def build_pairs(args):
    trials = args.trials
    routes = args.routes
    if args.pairing == "one_to_one":
        if len(trials) != len(routes):
            raise SystemExit("one_to_one pairing requires equal number of trials and routes")
        return list(zip(trials, routes))
    return [(trial, route) for trial in trials for route in routes]


def load_pair_result(trial, route):
    out_dir = pair_output_dir(trial, route)
    gold = load_json(S100_T10 / f"trial_{trial:02d}" / "gold.json")
    result = load_json(out_dir / "prepare_summary.json")
    jpath = judgment_path(out_dir)
    result["judgment"] = {
        "path": str(jpath),
        "rows": count_judgment_rows(jpath),
        "expected_rows": result["expected_judgment_rows"],
        "complete": count_judgment_rows(jpath) >= result["expected_judgment_rows"],
    }
    filtered = out_dir / "filtered_pred.json"
    if filtered.exists():
        result["filtered"] = metrics_row(gold, load_json(filtered))
        result["delta_prune_to_filtered"] = metric_delta(result["filtered"], result["prune_unseen"])
        result["delta_base_to_filtered"] = metric_delta(result["filtered"], result["base"])
    return result


def metric_delta(after, before):
    out = {
        "total_score": after["total_score"] - before["total_score"],
        "score_ner": after["score_ner"] - before["score_ner"],
        "score_re": after["score_re"] - before["score_re"],
        "entities": after["entities"] - before["entities"],
        "relations": after["relations"] - before["relations"],
    }
    for task in ["entity", "relation"]:
        out[task] = {
            key: after[task][key] - before[task][key]
            for key in ["tp", "fp", "fn", "precision", "recall", "f1"]
        }
    return out


def aggregate_results(results):
    stages = ["base", "prune_unseen", "filtered"]
    aggregate = {}
    for stage in stages:
        rows = [r[stage] for r in results if stage in r]
        if not rows:
            continue
        aggregate[stage] = {
            "n": len(rows),
            "mean_total": statistics.mean(r["total_score"] for r in rows),
            "mean_ner": statistics.mean(r["score_ner"] for r in rows),
            "mean_re": statistics.mean(r["score_re"] for r in rows),
            "sum_entity_tp": sum(r["entity"]["tp"] for r in rows),
            "sum_entity_fp": sum(r["entity"]["fp"] for r in rows),
            "sum_entity_fn": sum(r["entity"]["fn"] for r in rows),
            "sum_relation_tp": sum(r["relation"]["tp"] for r in rows),
            "sum_relation_fp": sum(r["relation"]["fp"] for r in rows),
            "sum_relation_fn": sum(r["relation"]["fn"] for r in rows),
        }
    return aggregate


def write_report(results, aggregate, args):
    lines = [
        "# Record-Level Online Filter 5x5 Experiment",
        "",
        f"Run time: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Pairing: `{args.pairing}`",
        "",
        "Pipeline: route prediction -> prune unseen relation type triples from fold pool -> online record-level audit -> ent>=0.75 / rel>=0.95 filter.",
        "",
        "## Pair Results",
        "",
        "| trial | route | stage | total | NER | RE | entity TP/FP/FN | relation TP/FP/FN | ents | rels |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        for stage in ["base", "prune_unseen", "filtered"]:
            if stage not in result:
                continue
            row = result[stage]
            lines.append(
                f"| {result['trial']} | `{result['route']}` | `{stage}` | "
                f"{row['total_score']:.6f} | {row['score_ner']:.6f} | {row['score_re']:.6f} | "
                f"{row['entity']['tp']}/{row['entity']['fp']}/{row['entity']['fn']} | "
                f"{row['relation']['tp']}/{row['relation']['fp']}/{row['relation']['fn']} | "
                f"{row['entities']} | {row['relations']} |"
            )
    lines += ["", "## Aggregate", ""]
    lines.append("| stage | n | total | NER | RE | entity TP/FP/FN | relation TP/FP/FN |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for stage, row in aggregate.items():
        lines.append(
            f"| `{stage}` | {row['n']} | {row['mean_total']:.6f} | "
            f"{row['mean_ner']:.6f} | {row['mean_re']:.6f} | "
            f"{row['sum_entity_tp']}/{row['sum_entity_fp']}/{row['sum_entity_fn']} | "
            f"{row['sum_relation_tp']}/{row['sum_relation_fp']}/{row['sum_relation_fn']} |"
        )
    lines += ["", "## Judgment Status", ""]
    lines.append("| trial | route | rows | expected | complete |")
    lines.append("|---:|---|---:|---:|---|")
    for result in results:
        j = result["judgment"]
        lines.append(
            f"| {result['trial']} | `{result['route']}` | {j['rows']} | "
            f"{j['expected_rows']} | {j['complete']} |"
        )
    (PROJECT_OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, nargs="+", default=DEFAULT_TRIALS)
    parser.add_argument("--routes", nargs="+", default=DEFAULT_ROUTES)
    parser.add_argument("--pairing", choices=["one_to_one", "full_grid"], default="one_to_one")
    parser.add_argument("--online", action="store_true", help="run online Opus judgments")
    parser.add_argument("--apply_filter", action="store_true", help="apply record-level strategy when judgments are present")
    parser.add_argument("--allow_partial_filter", action="store_true", help="apply strategy even if judgments are incomplete")
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--base_url", default=os.getenv("OPUS_BASE_URL", "https://zjapi.com"))
    parser.add_argument("--model", default=os.getenv("OPUS_MODEL", "claude-opus-4-8"))
    parser.add_argument("--api_key_envs", nargs="+", default=["OPUS_API_KEY", "OPUS_API_KEY_ALT", "API_KEY"])
    parser.add_argument("--timeout", type=int, default=int(os.getenv("OPUS_TIMEOUT", "240")))
    parser.add_argument("--retries", type=int, default=int(os.getenv("OPUS_RETRIES", "5")))
    parser.add_argument("--max_tokens", type=int, default=int(os.getenv("OPUS_MAX_TOKENS", "4096")))
    parser.add_argument("--batch_items", type=int, default=999)
    parser.add_argument("--judge_max_records", type=int, default=None, help="limit records sent to the online judge")
    args = parser.parse_args()

    PROJECT_OUT.mkdir(parents=True, exist_ok=True)
    pairs = build_pairs(args)
    dump_json(PROJECT_OUT / "experiment_config.json", {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "trials": args.trials,
        "routes": args.routes,
        "pairing": args.pairing,
        "pairs": pairs,
        "filter_notes": str(FILTER_WORKSPACE / "outputs" / "record_level_highscore_prune" / "FILTER_EXPERIMENT_NOTES.md"),
        "strategy": {
            "entity_policy": "drop_threshold",
            "entity_drop_threshold": 0.75,
            "apply_entity_fix": True,
            "relation_policy": "drop_threshold",
            "relation_drop_threshold": 0.95,
            "apply_relation_fix": False,
            "noempty_fallback": True,
        },
    })

    for trial, route in pairs:
        prepare_pair(trial, route)

    if args.online:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {
                ex.submit(run_online_judge, pair_output_dir(trial, route), args, i): (trial, route)
                for i, (trial, route) in enumerate(pairs)
            }
            for fut in as_completed(futures):
                trial, route = futures[fut]
                try:
                    status = fut.result()
                    print(json.dumps({"trial": trial, "route": route, **status}, ensure_ascii=False), flush=True)
                except Exception as exc:
                    print(json.dumps({"trial": trial, "route": route, "error": repr(exc)}, ensure_ascii=False), flush=True)

    if args.apply_filter:
        for trial, route in pairs:
            out = apply_record_filter(pair_output_dir(trial, route), allow_partial=args.allow_partial_filter)
            if out:
                print(json.dumps({"trial": trial, "route": route, "filtered": str(out)}, ensure_ascii=False), flush=True)

    results = [load_pair_result(trial, route) for trial, route in pairs]
    aggregate = aggregate_results(results)
    dump_json(PROJECT_OUT / "results.json", {"pairs": results, "aggregate": aggregate})
    write_report(results, aggregate, args)
    shutil.copy2(PROJECT_OUT / "results.json", SOLUTION / "outputs" / "record_level_filter_5x5_results.json")
    shutil.copy2(PROJECT_OUT / "REPORT.md", SOLUTION / "outputs" / "record_level_filter_5x5_REPORT.md")
    print(f"[OK] report -> {PROJECT_OUT / 'REPORT.md'}")


if __name__ == "__main__":
    main()
