"""OpenAI-compatible chat client with retries."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv


_ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(_ENV_PATH)

API_KEY = os.getenv("API_KEY", "").strip()
BASE_URL = os.getenv("BASE_URL", "").strip().rstrip("/")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "5"))
TIMEOUT = int(os.getenv("TIMEOUT", "300"))

MODEL_MAIN = os.getenv("MODEL_MAIN", "gpt-5.4").strip()
MODEL_CHEAP = os.getenv("MODEL_CHEAP", "gpt-5.4-mini").strip()
MODEL_SECONDARY = os.getenv("MODEL_SECONDARY", "gpt-5.5-openai-compact").strip()
LAST_RESPONSE_MODEL = None


class APIError(Exception):
    pass


def endpoint() -> str:
    return f"{BASE_URL}/v1/chat/completions"


def check_config() -> Optional[str]:
    """Return an error message if config looks unset, else None."""
    if not API_KEY or API_KEY.startswith("replace_with"):
        return "API_KEY is not configured"
    if not BASE_URL or BASE_URL.startswith("replace_with"):
        return "BASE_URL is not configured"
    return None


def chat(
    prompt: str,
    model: str,
    temperature: float | None = None,
    use_json: bool = True,
    max_tokens: int = 4096,
    system: str | None = None,
    retries: int | None = None,
) -> str:
    """Run one chat completion and return the assistant text content."""
    global LAST_RESPONSE_MODEL
    if temperature is None:
        temperature = TEMPERATURE
    if retries is None:
        retries = MAX_RETRIES

    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "temperature": temperature,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    if use_json:
        payload["response_format"] = {"type": "json_object"}

    last_err = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(endpoint(), headers=headers, json=payload, timeout=(20, TIMEOUT))
            if resp.status_code in (429, 500, 502, 503, 504):
                wait = min(30.0, 2.0 * (2 ** (attempt - 1)))
                retry_after = resp.headers.get("Retry-After")
                if retry_after:
                    try:
                        wait = max(wait, float(retry_after))
                    except ValueError:
                        pass
                last_err = f"HTTP {resp.status_code}: {resp.text[:300]}"
                time.sleep(wait)
                continue

            resp.raise_for_status()
            data = resp.json()
            LAST_RESPONSE_MODEL = data.get("model")
            choices = data.get("choices", [])
            if not choices:
                raise APIError(f"no choices: {json.dumps(data, ensure_ascii=False)[:300]}")

            msg = choices[0].get("message", {})
            content = msg.get("content")
            if isinstance(content, list):
                content = "".join(
                    part.get("text", "")
                    for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
                )
            if not content:
                content = msg.get("reasoning_content") or ""
            if not content:
                raise APIError(f"empty content: {json.dumps(choices[0], ensure_ascii=False)[:300]}")
            return content

        except requests.HTTPError as exc:
            code = getattr(exc.response, "status_code", None)
            try:
                body = exc.response.text[:300]
            except Exception:
                body = ""
            last_err = f"HTTPError {code}: {body}"
            if code == 400 and use_json and "response_format" in payload:
                payload.pop("response_format", None)
                use_json = False
                continue
            time.sleep(min(10.0, 1.0 * (2 ** (attempt - 1))))
        except Exception as exc:
            last_err = f"{type(exc).__name__}: {exc}"
            time.sleep(min(10.0, 1.0 * (2 ** (attempt - 1))))

    raise APIError(f"failed after {retries} retries. last_err={last_err}")


if __name__ == "__main__":
    err = check_config()
    if err:
        print("[CONFIG]", err)
        raise SystemExit(1)

    import sys

    model_arg = sys.argv[1] if len(sys.argv) > 1 else "cheap"
    model = {
        "main": MODEL_MAIN,
        "cheap": MODEL_CHEAP,
        "secondary": MODEL_SECONDARY,
    }.get(model_arg, model_arg)
    print(f"Testing model={model} @ {endpoint()}")
    out = chat('Reply with JSON: {"ok": true, "model": "<your model name>"}', model=model, max_tokens=100)
    if LAST_RESPONSE_MODEL:
        print("RESPONSE_MODEL:", LAST_RESPONSE_MODEL)
    print("RESPONSE:", out[:500])
