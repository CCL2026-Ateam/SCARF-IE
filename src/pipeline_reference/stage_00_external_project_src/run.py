"""Main extraction pipeline driver."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import client  # noqa: E402
from align import postprocess  # noqa: E402
from common import dump_json, evaluate, load_json, print_metrics  # noqa: E402
from prompts import SYSTEM_PROMPT, build_prompt  # noqa: E402
from retriever import ExemplarRetriever, SemanticExemplarRetriever  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
POOL_PATH = Path(client.os.getenv("POOL_PATH", str(ROOT / "data" / "pool.json")))
CACHE_DIR = Path(client.os.getenv("CACHE_DIR", str(ROOT / "cache")))
EMBED_CACHE_DIR = Path(client.os.getenv("EMBED_CACHE_DIR", str(CACHE_DIR / "srag_embeddings")))


def parse_json_lenient(s: str):
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s).strip()
    try:
        return json.loads(s)
    except Exception:
        pass

    start = s.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        c = s[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(s[start : i + 1])
                except Exception:
                    return None
    return None


def cache_key(model: str, prompt: str) -> str:
    return hashlib.sha256((model + "\x00" + prompt).encode("utf-8")).hexdigest()


def safe_dir(model: str) -> str:
    return re.sub(r'[<>:"/\\|?*]', "_", model)


def get_cached(model: str, prompt: str):
    f = CACHE_DIR / safe_dir(model) / (cache_key(model, prompt) + ".json")
    if f.exists():
        try:
            return json.loads(f.read_text(encoding="utf-8"))["raw"]
        except Exception:
            return None
    return None


def set_cached(model: str, prompt: str, raw: str) -> None:
    d = CACHE_DIR / safe_dir(model)
    d.mkdir(parents=True, exist_ok=True)
    f = d / (cache_key(model, prompt) + ".json")
    f.write_text(json.dumps({"raw": raw}, ensure_ascii=False), encoding="utf-8")


def resolve_model(name: str) -> str:
    return {
        "main": client.MODEL_MAIN,
        "cheap": client.MODEL_CHEAP,
        "secondary": client.MODEL_SECONDARY,
    }.get(name, name)


def process_record(rec, model, retriever, kshot, use_cache, max_tokens, conventions=False, temperature=None):
    text = rec["text"]
    exemplars = retriever.retrieve(text, k=kshot) if (retriever and kshot > 0) else None
    prompt = build_prompt(text, exemplars, conventions=conventions)
    cache_model = model if temperature in (None, 0.0) else f"{model}@t{temperature}"
    raw = get_cached(cache_model, prompt) if use_cache else None
    if raw is None:
        raw = client.chat(
            prompt,
            model=model,
            system=SYSTEM_PROMPT,
            use_json=True,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if use_cache:
            set_cached(cache_model, prompt, raw)
    parsed = parse_json_lenient(raw)
    if parsed is None:
        parsed = {"entities": [], "relations": []}
    return postprocess(parsed, text)


def build_retriever(args):
    if args.kshot <= 0:
        return None

    pool_path = Path(args.pool) if args.pool else POOL_PATH
    if not pool_path.exists():
        print(f"[WARN] few-shot pool not found: {pool_path}; running without exemplars")
        return None

    pool = load_json(pool_path)
    if args.retriever == "semantic":
        if not args.retriever_model:
            raise SystemExit("--retriever semantic requires --retriever_model")
        cache_path = args.retriever_cache
        if cache_path is None:
            safe_model = re.sub(r'[<>:"/\\|?*]', "_", args.retriever_model)
            cache_path = EMBED_CACHE_DIR / f"{safe_model}.json"
        return SemanticExemplarRetriever(pool, args.retriever_model, cache_path=str(cache_path))

    return ExemplarRetriever(pool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default="main", help="main|cheap|secondary|<raw model name>")
    ap.add_argument("--kshot", type=int, default=4)
    ap.add_argument("--pool", default=None, help="few-shot pool JSON path; overrides POOL_PATH env")
    ap.add_argument("--cache_dir", default=None, help="LLM response cache dir; overrides CACHE_DIR env")
    ap.add_argument("--retriever", choices=["lexical", "semantic"], default="lexical")
    ap.add_argument("--retriever_model", default=None)
    ap.add_argument("--retriever_cache", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--concurrency", type=int, default=int(client.os.getenv("CONCURRENCY", "16")))
    ap.add_argument("--max_tokens", type=int, default=4096)
    ap.add_argument("--no_cache", action="store_true")
    ap.add_argument("--conventions", action="store_true")
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--allow_record_failures", action="store_true")
    args = ap.parse_args()

    err = client.check_config()
    if err:
        print("[CONFIG ERROR]", err)
        print("Please configure API_KEY and BASE_URL, either in environment variables or pipeline_reference/.env.")
        raise SystemExit(1)

    global CACHE_DIR, EMBED_CACHE_DIR
    if args.cache_dir:
        CACHE_DIR = Path(args.cache_dir)
        EMBED_CACHE_DIR = CACHE_DIR / "srag_embeddings"

    model = resolve_model(args.model)
    records = load_json(args.input)
    if args.limit:
        records = records[: args.limit]

    retriever = build_retriever(args)

    print(
        f"[INFO] model={model} kshot={args.kshot} records={len(records)} "
        f"retriever={args.retriever} concurrency={args.concurrency} "
        f"cache={'off' if args.no_cache else 'on'}"
    )

    results = [None] * len(records)
    errors = []
    use_cache = not args.no_cache
    n_done = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {
            ex.submit(
                process_record,
                rec,
                model,
                retriever,
                args.kshot,
                use_cache,
                args.max_tokens,
                args.conventions,
                args.temperature,
            ): i
            for i, rec in enumerate(records)
        }
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                results[i] = fut.result()
            except Exception as exc:
                print(f"[WARN] record {i} failed: {exc}")
                errors.append((i, str(exc)))
                results[i] = {"text": records[i]["text"], "entities": [], "relations": []}
            n_done += 1
            if n_done % 20 == 0 or n_done == len(records):
                print(f"  progress {n_done}/{len(records)}", flush=True)

    if errors and not args.allow_record_failures:
        preview = "; ".join(f"{idx}: {msg[:160]}" for idx, msg in errors[:5])
        raise SystemExit(f"{len(errors)} record(s) failed; refusing to write submission. First errors: {preview}")

    dump_json(args.output, results)
    print(f"[OK] wrote {args.output}")

    if args.eval:
        metrics = evaluate(records, results)
        print_metrics(metrics, title=f"DEV EVAL model={model} kshot={args.kshot}")
        dump_json(Path(args.output).with_name(Path(args.output).stem + "_metrics.json"), metrics)

    if args.submit:
        clean = []
        for r in results:
            clean.append(
                {
                    "text": r["text"],
                    "entities": [
                        {"start": e["start"], "end": e["end"], "text": e["text"], "label": e["label"]}
                        for e in r["entities"]
                    ],
                    "relations": r["relations"],
                }
            )
        sub_path = Path(args.output).with_name("submit.json")
        dump_json(sub_path, clean)
        print(f"[OK] wrote submission json {sub_path}")


if __name__ == "__main__":
    main()
