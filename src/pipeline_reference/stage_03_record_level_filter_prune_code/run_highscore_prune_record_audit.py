import contextlib
import os
import runpy
import sys
from pathlib import Path


ROOT = Path(r"C:\Users\Zzl410410\Documents\Codex\2026-06-26\019ef541-55e7-7af1-8f34-0927a521fe82")
BASE_JUDGE = Path(
    r"C:\Users\Zzl410410\Documents\Codex\2026-06-24\019ef53b-8c1b-7a92-ad59-c7c456388b6f"
    r"\work\test_b_original1000_4member\src\judge_json_items_opus.py"
)
PRED = (
    ROOT
    / "work"
    / "highscore_prune_record_audit"
    / "highscore_plus_light_add_has_aff1_prune_unseen_relation_types_input_pred.json"
)
LOG_DIR = ROOT / "work" / "highscore_prune_record_audit"
OUT_DIR = LOG_DIR / "judgments_record_level"


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stdout_path = LOG_DIR / "run_record_audit.stdout.log"
    stderr_path = LOG_DIR / "run_record_audit.stderr.log"
    base_url = os.getenv("OPUS_BASE_URL", "https://zjapi.com")
    max_tokens = os.getenv("OPUS_MAX_TOKENS", "4096")
    timeout = os.getenv("OPUS_TIMEOUT", "240")
    retries = os.getenv("OPUS_RETRIES", "5")
    argv = [
        str(BASE_JUDGE),
        "--inputs",
        str(PRED),
        "--output_dir",
        str(OUT_DIR),
        "--model",
        "claude-opus-4-8",
        "--base_url",
        base_url,
        "--timeout",
        timeout,
        "--retries",
        retries,
        "--max_tokens",
        max_tokens,
        "--batch_items",
        "999",
    ]
    with open(stdout_path, "a", encoding="utf-8") as out, open(stderr_path, "a", encoding="utf-8") as err:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            print("[RUN] " + " ".join(argv), flush=True)
            old_argv = sys.argv
            try:
                sys.argv = argv
                runpy.run_path(str(BASE_JUDGE), run_name="__main__")
            finally:
                sys.argv = old_argv


if __name__ == "__main__":
    main()
