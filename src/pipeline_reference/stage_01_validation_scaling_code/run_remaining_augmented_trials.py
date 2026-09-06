import argparse
import subprocess
import sys
import time
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[2]
RUNNER = WORKSPACE / "work" / "test_b_validation_scaling" / "run_resumable_augmented_trial.py"
PROJECT_OUT = WORKSPACE / "work" / "test_b_validation_scaling" / "outputs" / "s100_t10"
PY = sys.executable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start_trial", type=int, default=7)
    ap.add_argument("--end_trial", type=int, default=9)
    ap.add_argument("--project_out", default=str(PROJECT_OUT))
    ap.add_argument("--max_cycles", type=int, default=420)
    ap.add_argument("--retry_wait", type=int, default=900)
    args = ap.parse_args()

    project_out = Path(args.project_out)
    for trial in range(args.start_trial, args.end_trial + 1):
        attempt = 1
        while True:
            print(f"[SUPERVISOR] trial_{trial:02d} attempt={attempt}", flush=True)
            cmd = [
                PY,
                str(RUNNER),
                "--trial",
                str(trial),
                "--project_out",
                str(project_out),
                "--max_cycles",
                str(args.max_cycles),
            ]
            rc = subprocess.run(cmd, cwd=WORKSPACE).returncode
            print(f"[SUPERVISOR] trial_{trial:02d} rc={rc}", flush=True)
            if rc == 0:
                break
            attempt += 1
            time.sleep(args.retry_wait)
    print("[SUPERVISOR] all requested trials complete", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
