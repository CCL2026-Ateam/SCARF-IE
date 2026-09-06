import json
import os
import runpy
import time
from pathlib import Path

import requests


ROOT = Path(r"C:\Users\Zzl410410\Documents\Codex\2026-06-26\019ef541-55e7-7af1-8f34-0927a521fe82")
LOG_DIR = ROOT / "work" / "highscore_prune_record_audit"
RUNNER = ROOT / "work" / "run_highscore_prune_record_audit.py"


def log(message):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    line = json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), **message}, ensure_ascii=False)
    with open(LOG_DIR / "wait_and_run_record_audit.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def probe():
    api_key = os.getenv("OPUS_API_KEY") or os.getenv("API_KEY")
    if not api_key:
        return {"ok": False, "error": "missing OPUS_API_KEY or API_KEY"}
    base_url = os.getenv("OPUS_BASE_URL", "https://zjapi.com").rstrip("/")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": os.getenv("OPUS_MODEL", "claude-opus-4-8"),
        "temperature": 0,
        "messages": [{"role": "user", "content": 'Return JSON: {"ok": true}'}],
        "max_tokens": 100,
    }
    try:
        resp = requests.post(
            f"{base_url}/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=(20, 120),
        )
        return {
            "ok": resp.status_code == 200,
            "status": resp.status_code,
            "body_head": resp.text[:120],
        }
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def main():
    sleep_seconds = int(os.getenv("OPUS_WAIT_SLEEP", "120"))
    max_attempts = int(os.getenv("OPUS_WAIT_ATTEMPTS", "720"))
    for attempt in range(1, max_attempts + 1):
        result = probe()
        log({"event": "probe", "attempt": attempt, **result})
        if result.get("ok"):
            log({"event": "start_runner", "runner": str(RUNNER)})
            runpy.run_path(str(RUNNER), run_name="__main__")
            log({"event": "runner_finished"})
            return
        time.sleep(sleep_seconds)
    log({"event": "gave_up", "attempts": max_attempts})


if __name__ == "__main__":
    main()
