import argparse
import hashlib
import json
import random
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
for _root in (ROOT, Path(r"K:\\浩然\\CCL\\solution\\solution")):
    if (_root / "data").exists() and (_root / "outputs").exists():
        ROOT = _root
        break
DATA = ROOT / "data"
OUT = ROOT / "outputs"
PROJECT = Path(__file__).resolve().parent
PROJECT_OUT = PROJECT / "outputs"
PY = sys.executable

MEMBERS = [
    ("secondary", False, "secondary_k4"),
    ("main", False, "main_k4"),
    ("secondary", True, "secondary_k4_conv"),
    ("main", True, "main_k4_conv"),
]

METHODS = {
    "old800": "original 800-record pool",
    "original1000": "original 800 pool + dev minus current validation sample",
    "augmented_clean": "original1000 + cleaned Test-A pseudo labels minus current validation sample",
}


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def strip_gold(records):
    return [{"text": r["text"]} for r in records]


def build_samples(n_records, trials, sample_size, seed, partition=False):
    rng = random.Random(seed)
    if partition:
        if trials * sample_size != n_records:
            raise ValueError("--partition_dev requires trials * sample_size == number of dev records")
        indices = list(range(n_records))
        rng.shuffle(indices)
        return [
            sorted(indices[i * sample_size:(i + 1) * sample_size])
            for i in range(trials)
        ]
    samples = []
    for trial in range(trials):
        samples.append(sorted(rng.sample(range(n_records), sample_size)))
    return samples


def make_pool(method, pool800, dev, valid_indices, augmented_clean=None):
    if method == "old800":
        return list(pool800)
    if method == "augmented_clean":
        if augmented_clean is None:
            raise ValueError("augmented_clean data is required for augmented_clean method")
        valid_texts = {dev[i]["text"] for i in valid_indices}
        return [r for r in augmented_clean if r["text"] not in valid_texts]
    valid = set(valid_indices)
    return list(pool800) + [r for i, r in enumerate(dev) if i not in valid]


def run(cmd, cwd=ROOT, stdout=None, stderr=None, timeout=None):
    print("[RUN]", " ".join(str(x) for x in cmd), flush=True)
    return subprocess.run(cmd, cwd=cwd, check=True, stdout=stdout, stderr=stderr, timeout=timeout)


def run_member(input_path, trial_dir, method, model, conventions, suffix, record_concurrency, member_timeout_sec):
    out_dir = trial_dir / "methods" / method
    output = out_dir / f"{suffix}.json"
    log = out_dir / f"{suffix}.log"
    err = out_dir / f"{suffix}.err.log"
    if output.exists():
        try:
            data = load_json(output)
            if len(data) == len(load_json(input_path)):
                print(f"[SKIP] existing member {trial_dir.name} {method} {suffix}", flush=True)
                return output
        except Exception:
            pass
    cmd = [
        PY,
        "src/run.py",
        "--input",
        str(input_path),
        "--output",
        str(output),
        "--model",
        model,
        "--kshot",
        "4",
        "--retriever",
        "lexical",
        "--concurrency",
        str(record_concurrency),
        "--max_tokens",
        "4096",
        "--no_cache",
    ]
    if conventions:
        cmd.append("--conventions")
    with open(log, "w", encoding="utf-8") as lf, open(err, "w", encoding="utf-8") as ef:
        run(cmd, stdout=lf, stderr=ef, timeout=member_timeout_sec)
    return output


def evaluate_method(trial_dir, method, member_paths):
    code = f"""
import json, pathlib, sys
root=pathlib.Path(r'{ROOT}')
sys.path.insert(0, str(root/'src'))
from common import load_json, dump_json, evaluate
from ensemble import ensemble
from relation_sanity_filter import filter_record
BEST_ENTITY_THRESHOLDS={{'ABS':2,'BIS':2,'CHR':2,'CROP':2,'CROSS':4,'GST':2,'MRK':4,'QTL':2}}
BEST_RELATION_THRESHOLDS={{'AFF':3,'CON':1,'HAS':3,'LOI':2,'OCI':2,'USE':5}}
gold=load_json(r'{trial_dir / "gold.json"}')
members=[load_json(p) for p in {json.dumps([str(p) for p in member_paths], ensure_ascii=False)}]
variants={{
  'best': ensemble(members, vote_min=3, rel_vote_min=2, rel_label_thresholds=BEST_RELATION_THRESHOLDS, ent_label_thresholds=BEST_ENTITY_THRESHOLDS),
  'strict3': ensemble(members, vote_min=3, rel_vote_min=3),
  'majority': ensemble(members, vote_min=2, rel_vote_min=2),
  'union': ensemble(members, vote_min=1, rel_vote_min=1),
}}
base=pathlib.Path(r'{trial_dir}')/'methods'/r'{method}'
out={{}}
for name,pred in variants.items():
    dump_json(base/f'{{name}}_ensemble.json', pred)
    out[name]={{}}
    for mode in ['none','conservative','balanced']:
        if mode == 'none':
            records = pred
            dropped = 0
        else:
            import collections
            stats={{'dropped_by_reason':collections.Counter(),'dropped_by_label':collections.Counter()}}
            records=[filter_record(r, mode, stats) for r in pred]
            dump_json(base/f'{{name}}_relfilter_{{mode}}.json', records)
            dropped=sum(len(r.get('relations',[])) for r in pred)-sum(len(r.get('relations',[])) for r in records)
        m=evaluate(gold, records)
        out[name][mode]={{
            'total_score':m['total_score'],
            'score_ner':m['score_ner'],
            'score_re':m['score_re'],
            'entity_precision':m['entity']['precision'],
            'entity_recall':m['entity']['recall'],
            'entity_f1':m['entity']['f1'],
            'relation_precision':m['relation']['precision'],
            'relation_recall':m['relation']['recall'],
            'relation_f1':m['relation']['f1'],
            'entities':sum(len(r.get('entities',[])) for r in records),
            'relations':sum(len(r.get('relations',[])) for r in records),
            'empty_count':sum(1 for r in records if not r.get('entities') and not r.get('relations')),
            'dropped_relations':dropped,
        }}
print(json.dumps(out, ensure_ascii=False))
"""
    proc = subprocess.run([PY, "-c", code], cwd=ROOT, check=True, capture_output=True, text=True)
    metrics = json.loads(proc.stdout)
    dump_json(trial_dir / "methods" / method / "metrics.json", metrics)
    return metrics


def summarize(master, methods=None):
    import statistics

    methods = list(methods or METHODS)
    variants = ["best", "strict3", "majority", "union"]
    modes = ["none", "conservative", "balanced"]
    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "trials": len(master["trials"]),
        "sample_size": master["sample_size"],
        "seed": master["seed"],
        "methods": {m: METHODS[m] for m in methods},
        "aggregate": {},
        "trials_detail": master["trials"],
    }
    for variant in variants:
        summary["aggregate"][variant] = {}
        for method in methods:
            summary["aggregate"][variant][method] = {}
            for mode in modes:
                rows = [
                    t["methods"][method]["metrics"][variant][mode]
                    for t in master["trials"]
                    if method in t.get("methods", {}) and "metrics" in t["methods"][method]
                ]
                if not rows:
                    continue
                summary["aggregate"][variant][method][mode] = {
                    "n": len(rows),
                    "mean_total": statistics.mean(r["total_score"] for r in rows),
                    "std_total": statistics.pstdev(r["total_score"] for r in rows),
                    "fold_total": [r["total_score"] for r in rows],
                    "mean_ner": statistics.mean(r["score_ner"] for r in rows),
                    "mean_re": statistics.mean(r["score_re"] for r in rows),
                    "mean_relation_precision": statistics.mean(r["relation_precision"] for r in rows),
                    "mean_relation_recall": statistics.mean(r["relation_recall"] for r in rows),
                    "mean_relations": statistics.mean(r["relations"] for r in rows),
                    "mean_dropped_relations": statistics.mean(r["dropped_relations"] for r in rows),
                }
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--sample_size", type=int, default=24)
    ap.add_argument("--seed", type=int, default=20260625)
    ap.add_argument("--member_concurrency", type=int, default=4)
    ap.add_argument("--record_concurrency", type=int, default=3)
    ap.add_argument("--start_trial", type=int, default=0)
    ap.add_argument("--max_trials", type=int, default=None)
    ap.add_argument("--member_timeout", type=int, default=14400, help="per-member timeout in seconds")
    ap.add_argument("--project_out", default=None, help="output directory for this validation run")
    ap.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    ap.add_argument("--no_copy_to_solution", action="store_true")
    ap.add_argument("--solution_prefix", default="testB_10x_validation")
    ap.add_argument("--partition_dev", action="store_true", help="partition dev without overlap; requires trials*sample_size == len(dev)")
    args = ap.parse_args()

    project_out = Path(args.project_out).resolve() if args.project_out else PROJECT_OUT.resolve()
    selected_methods = list(args.methods)

    project_out.mkdir(parents=True, exist_ok=True)
    pool_path = DATA / "pool.json"
    backup_path = DATA / "pool.before_10x_validation.json"
    shutil.copy2(pool_path, backup_path)

    pool800 = load_json(DATA / "pool.before_testB_original1000.json")
    dev = load_json(DATA / "dev.json")
    augmented_clean = load_json(OUT / "pseudo_train_augmented_clean.json")
    samples = build_samples(len(dev), args.trials, args.sample_size, args.seed, args.partition_dev)
    dump_json(project_out / "sample_sets.json", {
        "seed": args.seed,
        "trials": args.trials,
        "sample_size": args.sample_size,
        "methods": selected_methods,
        "partition_dev": args.partition_dev,
        "samples": samples,
    })

    master = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": args.seed,
        "sample_size": args.sample_size,
        "requested_trials": args.trials,
        "pool_backup": str(backup_path),
        "pool_backup_sha256": sha256(backup_path),
        "trials": [],
    }
    existing_master = project_out / "process_master.partial.json"
    if existing_master.exists():
        try:
            master = load_json(existing_master)
        except Exception:
            pass

    end_trial = args.trials if args.max_trials is None else min(args.trials, args.start_trial + args.max_trials)
    try:
        for trial in range(args.start_trial, end_trial):
            indices = samples[trial]
            trial_dir = project_out / f"trial_{trial:02d}"
            gold = [dev[i] for i in indices]
            dump_json(trial_dir / "gold.json", gold)
            dump_json(trial_dir / "input.json", strip_gold(gold))
            dump_json(trial_dir / "sample_manifest.json", {
                "trial": trial,
                "global_dev_indices": indices,
                "sample_size": len(indices),
            })

            trial_info = {
                "trial": trial,
                "indices": indices,
                "trial_dir": str(trial_dir),
                "methods": {},
            }
            prior = {t["trial"]: t for t in master.get("trials", [])}
            if trial in prior:
                trial_info = prior[trial]

            for method in selected_methods:
                pool = make_pool(method, pool800, dev, indices, augmented_clean)
                pool_file = trial_dir / "methods" / method / "pool.json"
                dump_json(pool_file, pool)
                shutil.copy2(pool_file, pool_path)
                print(f"[POOL] trial={trial} method={method} records={len(pool)}", flush=True)

                input_path = trial_dir / "input.json"
                member_paths = []
                with ThreadPoolExecutor(max_workers=args.member_concurrency) as ex:
                    futures = {
                        ex.submit(
                            run_member,
                            input_path,
                            trial_dir,
                            method,
                            model,
                            conventions,
                            suffix,
                            args.record_concurrency,
                            args.member_timeout,
                        ): suffix
                        for model, conventions, suffix in MEMBERS
                    }
                    for fut in as_completed(futures):
                        suffix = futures[fut]
                        member_paths.append(fut.result())
                        print(f"[DONE] trial={trial} method={method} member={suffix}", flush=True)
                member_paths = sorted(member_paths, key=lambda p: p.name)
                metrics = evaluate_method(trial_dir, method, member_paths)
                trial_info["methods"][method] = {
                    "pool_records": len(pool),
                    "pool_file": str(pool_file),
                    "member_files": [str(p) for p in member_paths],
                    "metrics": metrics,
                }

                # Update master after every method.
                master_trials = {t["trial"]: t for t in master.get("trials", [])}
                master_trials[trial] = trial_info
                master["trials"] = [master_trials[k] for k in sorted(master_trials)]
                dump_json(project_out / "process_master.partial.json", master)
                dump_json(project_out / "aggregate.partial.json", summarize(master, selected_methods))
    finally:
        shutil.copy2(backup_path, pool_path)
        master["restored_pool_sha256"] = sha256(pool_path)
        master["restored_pool_records"] = len(load_json(pool_path))
        dump_json(project_out / "process_master.json", master)
        aggregate = summarize(master, selected_methods)
        dump_json(project_out / "aggregate.json", aggregate)
        if not args.no_copy_to_solution:
            shutil.copy2(project_out / "sample_sets.json", OUT / f"{args.solution_prefix}_sample_sets.json")
            shutil.copy2(project_out / "process_master.json", OUT / f"{args.solution_prefix}_process_master.json")
            shutil.copy2(project_out / "aggregate.json", OUT / f"{args.solution_prefix}_aggregate.json")
            print(f"[OK] copied outputs to {OUT} with prefix {args.solution_prefix}")
        print(f"[OK] sample sets -> {project_out / 'sample_sets.json'}")
        print(f"[OK] process master -> {project_out / 'process_master.json'}")
        print(f"[OK] aggregate -> {project_out / 'aggregate.json'}")


if __name__ == "__main__":
    main()
