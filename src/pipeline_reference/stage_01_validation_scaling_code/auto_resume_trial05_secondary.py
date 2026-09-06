import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import requests
from dotenv import dotenv_values


ROOT = Path(__file__).resolve().parents[2]
SOLUTION_ROOT = Path(r"K:\浩然\CCL\solution\solution")
ENV_PATH = SOLUTION_ROOT / ".env"
DEFAULT_TRIAL_DIR = ROOT / "work" / "test_b_validation_scaling" / "outputs" / "s100_t10" / "trial_05"
RUNNER = ROOT / "work" / "test_b_validation_scaling" / "run_resumable_member.py"


def env_value(name, default=""):
    if name in os.environ and os.environ[name].strip():
        return os.environ[name].strip()
    vals = dotenv_values(ENV_PATH)
    return str(vals.get(name, default) or "").strip()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def target_status(output):
    output = Path(output)
    state_path = output.with_name(output.stem + "_state.json")
    if not state_path.exists():
        return 0, [], list(range(100))
    state = load_json(state_path)
    statuses = state.get("statuses", [])
    ok = sum(1 for s in statuses if s == "ok")
    failed = [i for i, s in enumerate(statuses) if s == "failed"]
    pending = [i for i, s in enumerate(statuses) if s in (None, "pending")]
    return ok, failed, pending


def probe(base_url, api_key, model):
    payload = {
        "model": model,
        "temperature": 0,
        "messages": [{"role": "user", "content": 'Reply with JSON only: {"ok":true}'}],
        "max_tokens": 30,
        "response_format": {"type": "json_object"},
    }
    try:
        resp = requests.post(
            base_url.rstrip("/") + "/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=(20, 120),
        )
        text = resp.text[:180].replace("\n", " ")
        return resp.status_code, text
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def choose_range(output, chunk, pending_first=False):
    ok, failed, pending = target_status(output)
    todo = pending or failed if pending_first else failed or pending
    if not todo:
        return ok, None, None
    start = todo[0]
    limit = 1
    for idx in todo[1:]:
        if idx == start + limit and limit < chunk:
            limit += 1
        else:
            break
    return ok, start, limit


def run_chunk(input_path, output, conventions, start, limit, log_dir, child_env):
    stamp = time.strftime("%Y%m%d_%H%M%S")
    name = Path(output).stem
    log_path = log_dir / f"{name}_auto_{start}_{start + limit}_{stamp}.log"
    err_path = log_dir / f"{name}_auto_{start}_{start + limit}_{stamp}.err.log"
    args = [
        sys.executable,
        str(RUNNER),
        "--input",
        str(input_path),
        "--output",
        str(output),
        "--model",
        "secondary",
        "--kshot",
        "4",
        "--max_tokens",
        "4096",
        "--start",
        str(start),
        "--limit",
        str(limit),
        "--stop_after_failures",
        "1",
        "--sleep_after_failure",
        "20",
    ]
    if conventions:
        args.append("--conventions")
    print(f"[RUN] {name} range={start}:{start + limit} log={log_path.name}", flush=True)
    with log_path.open("w", encoding="utf-8") as out, err_path.open("w", encoding="utf-8") as err:
        proc = subprocess.run(args, cwd=ROOT, stdout=out, stderr=err, env=child_env)
    print(f"[RUN-DONE] {name} rc={proc.returncode}", flush=True)
    return proc.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial_dir", default=str(DEFAULT_TRIAL_DIR))
    ap.add_argument("--chunk", type=int, default=2)
    ap.add_argument("--cooldown", type=int, default=180)
    ap.add_argument("--max_cycles", type=int, default=200)
    ap.add_argument("--base_url", action="append", default=None)
    ap.add_argument("--skip_probe", action="store_true")
    ap.add_argument("--pending_first", action="store_true")
    ap.add_argument("--one_target_per_cycle", action="store_true")
    args = ap.parse_args()

    trial_dir = Path(args.trial_dir)
    methods = trial_dir / "methods" / "augmented_clean"
    input_path = trial_dir / "input.json"
    targets = [
        (methods / "secondary_k4_conv.json", True),
        (methods / "secondary_k4.json", False),
    ]

    api_key = env_value("API_KEY")
    base_urls = args.base_url or [env_value("BASE_URL", "https://cdn.zjapi.com")]
    base_urls = [u.rstrip("/") for u in base_urls if u and u.strip()]
    model = env_value("MODEL_SECONDARY", "gpt-5.5-openai-compact")
    if not api_key or not base_urls:
        raise SystemExit("missing API_KEY or BASE_URL")

    print(f"[AUTO] bases={base_urls} model={model} chunk={args.chunk} cooldown={args.cooldown}", flush=True)
    for cycle in range(1, args.max_cycles + 1):
        statuses = []
        for output, _ in targets:
            ok, failed, pending = target_status(output)
            statuses.append(f"{Path(output).stem}: ok={ok} failed={len(failed)} pending={len(pending)}")
        print(f"[CYCLE {cycle}] " + " | ".join(statuses), flush=True)
        if all(target_status(output)[0] == 100 for output, _ in targets):
            print("[DONE] all secondary targets complete", flush=True)
            return 0

        selected_base = None
        if args.skip_probe:
            selected_base = base_urls[(cycle - 1) % len(base_urls)]
            print(f"[DIRECT] base={selected_base}", flush=True)
        else:
            for base_url in base_urls:
                code, text = probe(base_url, api_key, model)
                print(f"[PROBE] base={base_url} status={code} body={text}", flush=True)
                if code == 200:
                    selected_base = base_url
                    break
        if selected_base is None:
            time.sleep(args.cooldown)
            continue

        child_env = os.environ.copy()
        child_env["API_KEY"] = api_key
        child_env["BASE_URL"] = selected_base
        child_env["MODEL_SECONDARY"] = model
        progressed = False
        if args.one_target_per_cycle:
            ordered_targets = targets[(cycle - 1) % len(targets):] + targets[:(cycle - 1) % len(targets)]
        else:
            ordered_targets = targets
        for output, conventions in ordered_targets:
            ok, start, limit = choose_range(output, args.chunk, pending_first=args.pending_first)
            if start is None:
                continue
            rc = run_chunk(input_path, output, conventions, start, limit, methods, child_env)
            new_ok, _, _ = target_status(output)
            progressed = progressed or new_ok > ok
            if args.one_target_per_cycle or rc != 0:
                break

        if not progressed:
            time.sleep(args.cooldown)

    print("[STOP] max_cycles reached", flush=True)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
