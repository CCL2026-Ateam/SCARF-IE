import argparse
import json
import os
import requests
import shutil
import subprocess
import sys
import time
from pathlib import Path

from dotenv import dotenv_values


WORKSPACE = Path(__file__).resolve().parents[2]
SOLUTION_ROOT = Path(r"K:\浩然\CCL\solution\solution")
ENV_PATH = SOLUTION_ROOT / ".env"
TENX = WORKSPACE / "work" / "test_b_internal_validation_10x" / "run_10x_validation.py"
RUNNER = WORKSPACE / "work" / "test_b_validation_scaling" / "run_resumable_member.py"
PROJECT_OUT = WORKSPACE / "work" / "test_b_validation_scaling" / "outputs" / "s100_t10"
PY = sys.executable

sys.path.insert(0, str(TENX.parent))
import run_10x_validation as tenx  # noqa: E402


MEMBERS = [
    ("main", False, "main_k4"),
    ("main", True, "main_k4_conv"),
    ("secondary", False, "secondary_k4"),
    ("secondary", True, "secondary_k4_conv"),
]

SECONDARY_HEALTH_TTL = int(os.environ.get("SECONDARY_HEALTH_TTL", "300"))
SECONDARY_HEALTH_WAIT = int(os.environ.get("SECONDARY_HEALTH_WAIT", "90"))


def env_value(name, default=""):
    if name in os.environ and str(os.environ[name]).strip():
        return str(os.environ[name]).strip()
    vals = dotenv_values(ENV_PATH)
    return str(vals.get(name, default) or "").strip()


def env_with_solution_defaults():
    env = os.environ.copy()
    vals = dotenv_values(ENV_PATH)
    for name in ["API_KEY", "BASE_URL", "MODEL_MAIN", "MODEL_SECONDARY", "TIMEOUT", "MAX_RETRIES"]:
        if not str(env.get(name, "")).strip() and vals.get(name):
            env[name] = str(vals[name]).strip()
    return env


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def state_counts(output, n_records):
    output = Path(output)
    state_path = output.with_name(output.stem + "_state.json")
    if not state_path.exists():
        if output.exists():
            try:
                if len(load_json(output)) == n_records:
                    return n_records, [], []
            except Exception:
                pass
        return 0, [], list(range(n_records))
    state = load_json(state_path)
    statuses = state.get("statuses", [])
    ok = sum(1 for s in statuses if s == "ok")
    failed = [i for i, s in enumerate(statuses) if s == "failed"]
    pending = [i for i, s in enumerate(statuses) if s in (None, "pending")]
    return ok, failed, pending


def choose_start(output, n_records, pending_first=True):
    ok, failed, pending = state_counts(output, n_records)
    todo = (pending or failed) if pending_first else (failed or pending)
    return ok, todo[0] if todo else None


def secondary_health(base_url):
    api_key = env_value("API_KEY")
    model = env_value("MODEL_SECONDARY", "gpt-5.5-openai-compact")
    if not api_key:
        return False, "missing API_KEY"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": 'Reply only JSON: {"ok": true}'}],
        "max_tokens": 50,
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    try:
        resp = requests.post(
            f"{base_url.rstrip('/')}/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=(20, 60),
        )
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    if not resp.ok:
        return False, f"HTTP {resp.status_code}: {resp.text[:180]}"
    return True, "ok"


def prepare_trial(trial, project_out):
    data = SOLUTION_ROOT / "data"
    out = SOLUTION_ROOT / "outputs"
    pool800 = load_json(data / "pool.before_testB_original1000.json")
    dev = load_json(data / "dev.json")
    augmented_clean = load_json(out / "pseudo_train_augmented_clean.json")
    samples = tenx.build_samples(len(dev), 10, 100, 20260625, False)
    sample_file = project_out / "sample_sets.json"
    if not sample_file.exists():
        dump_json(sample_file, {
            "seed": 20260625,
            "trials": 10,
            "sample_size": 100,
            "methods": ["augmented_clean"],
            "partition_dev": False,
            "samples": samples,
        })
    indices = samples[trial]
    trial_dir = project_out / f"trial_{trial:02d}"
    gold = [dev[i] for i in indices]
    dump_json(trial_dir / "gold.json", gold)
    dump_json(trial_dir / "input.json", tenx.strip_gold(gold))
    dump_json(trial_dir / "sample_manifest.json", {
        "trial": trial,
        "global_dev_indices": indices,
        "sample_size": len(indices),
    })
    pool = tenx.make_pool("augmented_clean", pool800, dev, indices, augmented_clean)
    pool_file = trial_dir / "methods" / "augmented_clean" / "pool.json"
    dump_json(pool_file, pool)
    if os.environ.get("COPY_POOL_TO_SOLUTION", "0") == "1":
        shutil.copy2(pool_file, data / "pool.json")
    return trial_dir, len(gold), len(pool), pool_file


def run_member(trial_dir, n_records, pool_file, model, conventions, suffix, max_cycles):
    out_dir = trial_dir / "methods" / "augmented_clean"
    output = out_dir / f"{suffix}.json"
    chunk = 8 if model == "main" else 1
    bases = ["https://zjapi.com"] if model == "main" else ["https://cdn.zjapi.com", "https://zjapi.com"]
    next_secondary_health_at = 0

    for cycle in range(1, max_cycles + 1):
        ok, start = choose_start(output, n_records, pending_first=True)
        if ok == n_records and output.exists():
            print(f"[MEMBER-DONE] {trial_dir.name} {suffix}", flush=True)
            return output
        base_url = bases[(cycle - 1) % len(bases)]
        if model == "secondary" and time.time() >= next_secondary_health_at:
            healthy, detail = secondary_health(base_url)
            print(f"[HEALTH] {trial_dir.name} {suffix} base={base_url} healthy={healthy} detail={detail}", flush=True)
            if not healthy:
                time.sleep(SECONDARY_HEALTH_WAIT)
                next_secondary_health_at = 0
                continue
            next_secondary_health_at = time.time() + SECONDARY_HEALTH_TTL
        stamp = time.strftime("%Y%m%d_%H%M%S")
        log = out_dir / f"{suffix}_resumable_{start}_{start + chunk}_{stamp}.log"
        err = out_dir / f"{suffix}_resumable_{start}_{start + chunk}_{stamp}.err.log"
        cmd = [
            PY,
            str(RUNNER),
            "--input",
            str(trial_dir / "input.json"),
            "--output",
            str(output),
            "--model",
            model,
            "--kshot",
            "4",
            "--max_tokens",
            "4096",
            "--pool",
            str(pool_file),
            "--start",
            str(start),
            "--limit",
            str(chunk),
            "--stop_after_failures",
            "1",
            "--sleep_after_failure",
            "20",
        ]
        if conventions:
            cmd.append("--conventions")
        env = env_with_solution_defaults()
        env["BASE_URL"] = base_url
        env["MAX_RETRIES"] = env.get("MAX_RETRIES", "3")
        print(f"[RUN] {trial_dir.name} {suffix} ok={ok}/{n_records} range={start}:{start + chunk} base={base_url}", flush=True)
        with log.open("w", encoding="utf-8") as lf, err.open("w", encoding="utf-8") as ef:
            rc = subprocess.run(cmd, cwd=WORKSPACE, stdout=lf, stderr=ef, env=env).returncode
        new_ok, failed, pending = state_counts(output, n_records)
        print(f"[STATE] {suffix} rc={rc} ok={new_ok}/{n_records} failed={failed[:10]} pending={len(pending)}", flush=True)
        if new_ok == n_records and output.exists():
            return output
        if rc != 0:
            if model == "secondary":
                next_secondary_health_at = 0
            time.sleep(120 if model == "secondary" else 30)
    raise RuntimeError(f"{suffix} did not complete after {max_cycles} cycles")


def evaluate_trial(trial, project_out):
    cmd = [
        PY,
        str(TENX),
        "--trials",
        "10",
        "--sample_size",
        "100",
        "--seed",
        "20260625",
        "--start_trial",
        str(trial),
        "--max_trials",
        "1",
        "--methods",
        "augmented_clean",
        "--project_out",
        str(project_out),
        "--member_concurrency",
        "1",
        "--record_concurrency",
        "1",
        "--member_timeout",
        "60",
        "--no_copy_to_solution",
    ]
    subprocess.run(cmd, cwd=WORKSPACE, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial", type=int, required=True)
    ap.add_argument("--project_out", default=str(PROJECT_OUT))
    ap.add_argument("--max_cycles", type=int, default=220)
    ap.add_argument("--members", nargs="*", default=None, help="Optional member suffixes to run.")
    ap.add_argument("--skip_evaluate", action="store_true")
    args = ap.parse_args()

    project_out = Path(args.project_out)
    trial_dir, n_records, pool_records, pool_file = prepare_trial(args.trial, project_out)
    print(f"[TRIAL] {trial_dir.name} records={n_records} pool_records={pool_records}", flush=True)
    selected = set(args.members or [suffix for _, _, suffix in MEMBERS])
    for model, conventions, suffix in MEMBERS:
        if suffix not in selected:
            continue
        run_member(trial_dir, n_records, pool_file, model, conventions, suffix, args.max_cycles)
    if not args.skip_evaluate:
        evaluate_trial(args.trial, project_out)
        print(f"[DONE] evaluated {trial_dir.name}", flush=True)
    else:
        print(f"[DONE] skipped evaluation {trial_dir.name}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
