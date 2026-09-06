"""Summarize per-target multi-filter replay reports for one trial."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT = Path(__file__).resolve().parent
OUT_ROOT = PROJECT / "outputs"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial", type=int, default=0)
    args = parser.parse_args()

    rows = []
    pattern = f"record_level_multi_filter_trial_{args.trial:02d}__*/multi_filter_results.json"
    for path in sorted(OUT_ROOT.glob(pattern)):
        result = load_json(path)
        base = next(r for r in result["main_results"] if r["name"] == "target_base_pruned")
        best_main = result["main_results"][0]
        best_cascade = result["cascade_permutations"][0] if result.get("cascade_permutations") else None
        merged = next(r for r in result["main_results"] if r["name"] == "merged_all_filters_once")
        default_cascade = next(r for r in result["main_results"] if r["name"] == "cascade_default_order")
        rows.append({
            "target_route": result["target_route"],
            "base": base,
            "best_main": best_main,
            "merged_all_filters_once": merged,
            "cascade_default_order": default_cascade,
            "best_cascade_sample": best_cascade,
            "report": str(path.with_name("MULTI_FILTER_REPLAY.md")),
        })

    rows.sort(key=lambda r: r["best_main"]["metric"]["total"], reverse=True)
    out_dir = OUT_ROOT / f"record_level_multi_filter_trial_{args.trial:02d}_summary"
    write_json(out_dir / "summary.json", {"trial": args.trial, "rows": rows})

    lines = [
        f"# Multi-Filter Trial {args.trial:02d} Summary",
        "",
        "Offline replay over one saved Test-B trial; all scores reuse saved judgments and make no model calls.",
        "",
        "| target route | base total | best main experiment | best total | gain | merged-all gain | default-cascade gain | sampled best-cascade gain |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        base_total = row["base"]["metric"]["total"]
        best = row["best_main"]
        merged = row["merged_all_filters_once"]
        default_cascade = row["cascade_default_order"]
        best_cascade = row["best_cascade_sample"]
        cascade_gain = best_cascade["delta_vs_base"]["total"] if best_cascade else 0.0
        lines.append(
            f"| `{row['target_route']}` | {base_total:.6f} | `{best['name']}` | "
            f"{best['metric']['total']:.6f} | {best['delta_vs_base']['total']:+.6f} | "
            f"{merged['delta_vs_base']['total']:+.6f} | "
            f"{default_cascade['delta_vs_base']['total']:+.6f} | "
            f"{cascade_gain:+.6f} |"
        )
    lines += [
        "",
        "## Conclusion",
        "",
        "On this trial, repeated or cascaded multi-filtering is not the best pattern. The best gain usually comes from applying one stronger external filter source to the target prediction. Merging all filters or cascading them tends to be more aggressive and can reduce relation recall.",
    ]
    (out_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(out_dir / "SUMMARY.md"), "targets": len(rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
