import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path

import requests


ENTITY_LABELS = ["CROP", "VAR", "TRT", "GST", "GENE", "QTL", "MRK", "CHR", "BM", "CROSS", "ABS", "BIS"]
RELATION_LABELS = ["CON", "USE", "HAS", "AFF", "OCI", "LOI"]

SYSTEM = (
    "You are a strict information-extraction auditor for cereal/millet breeding "
    "literature. You judge existing entity and relation candidates only. Return "
    "strict JSON only."
)

GUIDE = """Entity labels:
- CROP crop species/class/genus
- VAR cultivar/variety/accession/line/genotype/mutant/transgenic material/germplasm
- TRT phenotypic/agronomic/quality/resistance/physiological trait or measured indicator
- GST growth/developmental/measurement stage
- GENE gene/candidate gene/gene-family member/gene symbol
- QTL QTL/locus/mapping interval/MQTL/resistance locus
- MRK molecular marker
- CHR chromosome/linkage group/chromosome segment
- BM breeding/screening/detection/omics/mapping method
- CROSS parent/cross combination
- ABS abiotic stress/treatment/non-living adverse condition
- BIS biotic stress/pest/pathogen/disease

Relation labels:
- CON contains/belongs-to/is-a/alias/abbreviation/synonym/apposition
- USE uses/employs/applies/via/by/through method/technique/material
- HAS object has/shows/exhibits/possesses a trait
- AFF affects/regulates/promotes/suppresses/induces/responds/sensitive/tolerant/resistant
- OCI occurs-in/measured-at/investigated-at/treated-at a growth stage
- LOI located/mapped/linked/associated/candidate/co-localized molecular positioning

Audit rules:
1. Judge each candidate independently against the current TEXT.
2. Entity text must be a contiguous substring and label must match the meaning.
3. Relation must have direct textual evidence; do not infer from co-occurrence.
4. Relation head/tail must be grounded in the text and label direction must be correct.
5. If a candidate is nearly correct but label/span/relation label is wrong, action="fix" and provide corrected.
6. If unsupported, generic, hallucinated, or only weakly implied, action="drop".
"""


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def append_jsonl(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def read_jsonl_keys(path):
    done = set()
    path = Path(path)
    if not path.exists():
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            done.add((obj.get("record_index"), obj.get("batch_index")))
    return done


def parse_json_lenient(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
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
                    return json.loads(text[start : i + 1])
                except Exception:
                    return None
    return None


def file_slug(path):
    name = Path(path).stem
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def candidate_id(kind, idx):
    return ("E" if kind == "entity" else "R") + str(idx)


def build_candidates(record):
    out = []
    for i, e in enumerate(record.get("entities", []) or []):
        out.append({
            "id": candidate_id("entity", i),
            "kind": "entity",
            "candidate": {
                "text": e.get("text", ""),
                "start": e.get("start"),
                "end": e.get("end"),
                "label": e.get("label", ""),
            },
        })
    for i, r in enumerate(record.get("relations", []) or []):
        out.append({
            "id": candidate_id("relation", i),
            "kind": "relation",
            "candidate": {
                "head": r.get("head", ""),
                "head_type": r.get("head_type", ""),
                "tail": r.get("tail", ""),
                "tail_type": r.get("tail_type", ""),
                "label": r.get("label", ""),
            },
        })
    return out


def chunked(items, size):
    for i in range(0, len(items), size):
        yield i // size, items[i : i + size]


def build_prompt(text, candidates):
    return f"""{GUIDE}

TEXT:
{text}

CANDIDATES:
{json.dumps(candidates, ensure_ascii=False)}

Return this JSON shape:
{{
  "decisions": [
    {{
      "id": "E0",
      "kind": "entity",
      "action": "keep|drop|fix",
      "confidence": 0.0,
      "reason": "short reason",
      "corrected": null
    }}
  ]
}}

For fixed entities, corrected must be {{"text": "...", "label": "...", "start": <int|null>, "end": <int|null>}}.
For fixed relations, corrected must be {{"head": "...", "head_type": "...", "tail": "...", "tail_type": "...", "label": "..."}}.
"""


class OpusClient:
    def __init__(self, api_key, base_url, model, timeout, retries):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.retries = retries

    def chat(self, prompt, max_tokens):
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
        last_err = None
        for attempt in range(1, self.retries + 1):
            try:
                resp = requests.post(
                    f"{self.base_url}/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=(20, self.timeout),
                )
                if resp.status_code in (429, 500, 502, 503, 504):
                    last_err = f"HTTP {resp.status_code}: {resp.text[:300]}"
                    time.sleep(min(60, 2 ** attempt))
                    continue
                if resp.status_code == 400 and "response_format" in payload:
                    payload.pop("response_format", None)
                    continue
                resp.raise_for_status()
                data = resp.json()
                msg = data["choices"][0]["message"]
                content = msg.get("content") or msg.get("reasoning_content") or ""
                if isinstance(content, list):
                    content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
                if content:
                    return content
                last_err = "empty content"
            except Exception as exc:
                last_err = f"{type(exc).__name__}: {exc}"
                time.sleep(min(30, 2 ** (attempt - 1)))
        raise RuntimeError(f"Opus judge failed after {self.retries} retries: {last_err}")


def prompt_hash(prompt, model):
    return hashlib.sha256((model + "\0" + prompt).encode("utf-8")).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--model", default=os.getenv("OPUS_MODEL", "claude-opus-4-8"))
    ap.add_argument("--base_url", default=os.getenv("OPUS_BASE_URL", os.getenv("BASE_URL", "https://zjapi.com")))
    ap.add_argument("--api_key_env", default="OPUS_API_KEY")
    ap.add_argument("--timeout", type=int, default=int(os.getenv("OPUS_TIMEOUT", "180")))
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--max_tokens", type=int, default=4096)
    ap.add_argument("--batch_items", type=int, default=40)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--max_records", type=int, default=None)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--keep_raw", action="store_true")
    args = ap.parse_args()

    api_key = os.getenv(args.api_key_env) or os.getenv("API_KEY")
    if not api_key:
        raise SystemExit(f"missing API key env: {args.api_key_env} or API_KEY")

    client = OpusClient(api_key, args.base_url, args.model, args.timeout, args.retries)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for input_path in args.inputs:
        records = load_json(input_path)
        stop = len(records) if args.end is None else min(args.end, len(records))
        idxs = list(range(max(0, args.start), stop))
        if args.max_records is not None:
            idxs = idxs[: args.max_records]
        slug = file_slug(input_path)
        out_path = out_dir / f"{slug}.opus_item_judgments.jsonl"
        done = read_jsonl_keys(out_path)
        planned_batches = 0
        planned_items = 0

        jobs = []
        for idx in idxs:
            cands = build_candidates(records[idx])
            if not cands:
                continue
            planned_items += len(cands)
            for batch_index, batch in chunked(cands, args.batch_items):
                if (idx, batch_index) in done:
                    continue
                planned_batches += 1
                jobs.append((idx, batch_index, batch))

        print(
            f"[PLAN] {input_path} records={len(idxs)} items={planned_items} "
            f"pending_batches={planned_batches} model={args.model}",
            flush=True,
        )
        if args.dry_run:
            continue

        completed = 0
        for idx, batch_index, batch in jobs:
            prompt = build_prompt(records[idx]["text"], batch)
            raw = client.chat(prompt, args.max_tokens)
            parsed = parse_json_lenient(raw)
            if not isinstance(parsed, dict) or not isinstance(parsed.get("decisions"), list):
                parsed = {"decisions": [], "parse_error": True}
            row = {
                "source_file": str(input_path),
                "record_index": idx,
                "batch_index": batch_index,
                "prompt_hash": prompt_hash(prompt, args.model),
                "model": args.model,
                "candidate_ids": [c["id"] for c in batch],
                "decisions": parsed.get("decisions", []),
            }
            if parsed.get("parse_error"):
                row["parse_error"] = True
            if args.keep_raw:
                row["raw"] = raw
            append_jsonl(out_path, row)
            completed += 1
            if completed % 10 == 0 or completed == len(jobs):
                print(f"  progress {completed}/{len(jobs)} -> {out_path}", flush=True)

        summary = summarize(out_path)
        dump_json(out_dir / f"{slug}.opus_item_judgments_summary.json", summary)
        print(f"[OK] wrote {out_path}", flush=True)


def summarize(path):
    summary = {
        "records_with_judgments": 0,
        "batches": 0,
        "decisions": 0,
        "by_kind": {},
        "by_action": {},
        "parse_errors": 0,
    }
    seen_records = set()
    path = Path(path)
    if not path.exists():
        return summary
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            summary["batches"] += 1
            if obj.get("parse_error"):
                summary["parse_errors"] += 1
            seen_records.add(obj.get("record_index"))
            for d in obj.get("decisions", []):
                summary["decisions"] += 1
                kind = str(d.get("kind", "unknown"))
                action = str(d.get("action", "unknown"))
                summary["by_kind"][kind] = summary["by_kind"].get(kind, 0) + 1
                summary["by_action"][action] = summary["by_action"].get(action, 0) + 1
    summary["records_with_judgments"] = len(seen_records)
    return summary


if __name__ == "__main__":
    main()
