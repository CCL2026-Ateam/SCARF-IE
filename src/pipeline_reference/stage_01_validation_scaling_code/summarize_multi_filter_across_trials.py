"""Summarize multi-filter replay results across saved Test-B trials."""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


PROJECT = Path(__file__).resolve().parent
OUT_ROOT = PROJECT / "outputs"

ROUTES = [
    "majority",
    "best",
    "union",
    "strict3",
    "highscore_plus_light_drop_blocked_loi",
    "highscore_plus_light_add_has_aff1",
]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def mean(values):
    return statistics.mean(values) if values else 0.0


def stdev(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0


def result_path(trial, route):
    return OUT_ROOT / f"record_level_multi_filter_trial_{trial:02d}__{route}" / "multi_filter_results.json"


def load_rows(trials):
    rows = []
    for trial in trials:
        for route in ROUTES:
            path = result_path(trial, route)
            if not path.exists():
                raise FileNotFoundError(path)
            result = load_json(path)
            base = next(r for r in result["main_results"] if r["name"] == "target_base_pruned")
            merged = next(r for r in result["main_results"] if r["name"] == "merged_all_filters_once")
            default_cascade = next(r for r in result["main_results"] if r["name"] == "cascade_default_order")
            best = result["main_results"][0]
            best_cascade = result["cascade_permutations"][0] if result.get("cascade_permutations") else None
            best_single = max(
                [r for r in result["main_results"] if r["name"].startswith("single_filter__")],
                key=lambda r: r["metric"]["total"],
            )
            rows.append({
                "trial": trial,
                "target_route": route,
                "base_total": base["metric"]["total"],
                "best_main_name": best["name"],
                "best_main_total": best["metric"]["total"],
                "best_main_gain": best["delta_vs_base"]["total"],
                "best_single_name": best_single["name"],
                "best_single_total": best_single["metric"]["total"],
                "best_single_gain": best_single["delta_vs_base"]["total"],
                "merged_gain": merged["delta_vs_base"]["total"],
                "default_cascade_gain": default_cascade["delta_vs_base"]["total"],
                "best_cascade_sample_gain": (
                    best_cascade["delta_vs_base"]["total"] if best_cascade else 0.0
                ),
            })
    return rows


def summarize_by_target(rows):
    by_target = defaultdict(list)
    for row in rows:
        by_target[row["target_route"]].append(row)

    summaries = []
    for route in ROUTES:
        group = by_target[route]
        best_names = Counter(row["best_main_name"] for row in group)
        best_single_names = Counter(row["best_single_name"] for row in group)
        summaries.append({
            "target_route": route,
            "n": len(group),
            "base_total_mean": mean([r["base_total"] for r in group]),
            "best_main_gain_mean": mean([r["best_main_gain"] for r in group]),
            "best_main_gain_stdev": stdev([r["best_main_gain"] for r in group]),
            "best_single_gain_mean": mean([r["best_single_gain"] for r in group]),
            "merged_gain_mean": mean([r["merged_gain"] for r in group]),
            "default_cascade_gain_mean": mean([r["default_cascade_gain"] for r in group]),
            "best_cascade_sample_gain_mean": mean([r["best_cascade_sample_gain"] for r in group]),
            "best_main_wins": dict(best_names.most_common()),
            "best_single_wins": dict(best_single_names.most_common()),
        })
    return summaries


def render_report(out_path, trials, rows, summaries):
    lines = [
        "# Multi-Filter Replay Across Trials",
        "",
        f"Trials: {', '.join(f'trial_{t:02d}' for t in trials)}.",
        "All runs are offline replays over saved record-level judgments; no model calls were made.",
        "",
        "## Mean Gain By Target Route",
        "",
        "| target route | n | base mean | best-main gain | best-single gain | merged-all gain | default-cascade gain | sampled best-cascade gain | most frequent best main |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for summary in summaries:
        top_name, top_count = next(iter(summary["best_main_wins"].items()))
        lines.append(
            f"| `{summary['target_route']}` | {summary['n']} | {summary['base_total_mean']:.6f} | "
            f"{summary['best_main_gain_mean']:+.6f} | "
            f"{summary['best_single_gain_mean']:+.6f} | "
            f"{summary['merged_gain_mean']:+.6f} | "
            f"{summary['default_cascade_gain_mean']:+.6f} | "
            f"{summary['best_cascade_sample_gain_mean']:+.6f} | "
            f"`{top_name}` ({top_count}/{summary['n']}) |"
        )

    strict3_rows = [r for r in rows if r["target_route"] == "strict3"]
    lines += [
        "",
        "## Strict3 Detail",
        "",
        "| trial | base | best main | best total | gain | merged gain | default-cascade gain | sampled best-cascade gain |",
        "|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(strict3_rows, key=lambda r: r["trial"]):
        lines.append(
            f"| {row['trial']} | {row['base_total']:.6f} | `{row['best_main_name']}` | "
            f"{row['best_main_total']:.6f} | {row['best_main_gain']:+.6f} | "
            f"{row['merged_gain']:+.6f} | {row['default_cascade_gain']:+.6f} | "
            f"{row['best_cascade_sample_gain']:+.6f} |"
        )

    better_than_merged = sum(1 for r in rows if r["best_single_gain"] > r["merged_gain"])
    better_than_default_cascade = sum(1 for r in rows if r["best_single_gain"] > r["default_cascade_gain"])
    lines += [
        "",
        "## Overall Check",
        "",
        f"Best single-filter gain beats merged-all gain in {better_than_merged}/{len(rows)} target-route cases.",
        f"Best single-filter gain beats default cascade gain in {better_than_default_cascade}/{len(rows)} target-route cases.",
        "",
        "The cross-trial result supports the single-trial observation: using the strongest matched filter source usually gives more gain than repeatedly applying multiple filters. Cascading/merging often removes additional false positives, but the extra deletions tend to hurt recall enough that total score drops versus the best single-filter choice.",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", default="0,1,4,6,7")
    args = parser.parse_args()
    trials = [int(x) for x in args.trials.split(",") if x.strip()]

    rows = load_rows(trials)
    summaries = summarize_by_target(rows)
    out_dir = OUT_ROOT / "record_level_multi_filter_across_trials"
    dump_json(out_dir / "summary.json", {
        "trials": trials,
        "rows": rows,
        "summaries": summaries,
    })
    render_report(out_dir / "SUMMARY.md", trials, rows, summaries)
    print(json.dumps({
        "summary": str(out_dir / "SUMMARY.md"),
        "rows": len(rows),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
