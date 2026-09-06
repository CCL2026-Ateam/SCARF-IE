"""AI-audited union-only entity augmenter for strict3 predictions.

The script treats strict3 filtered predictions as the high-precision base and
uses union filtered predictions only as an entity recall pool. It can:

1. prepare union-only entity candidates;
2. ask an OpenAI-compatible chat/completions model to judge those candidates;
3. apply keep/fix decisions back onto strict3 predictions;
4. evaluate the augmented result on the saved validation folds.

Online calls are resumable through JSONL judgment files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import threading
import time
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from env_loader import default_env_paths, load_env_defaults


WORKSPACE = Path(__file__).resolve().parents[2]
PROJECT = Path(__file__).resolve().parent
load_env_defaults(*default_env_paths(PROJECT))
PAIR_ROOT = PROJECT / "outputs" / "record_level_filter_5x5" / "pairs"
S100_T10 = PROJECT / "outputs" / "s100_t10"
OUT_DIR = Path(os.getenv("ENTITY_AUG_OUT_DIR") or (PROJECT / "outputs" / "union_only_entity_ai_augmenter"))
SOLUTION_SRC = Path(os.getenv("SOLUTION_SRC") or (PROJECT.parent / "stage_00_external_project_src"))

DEFAULT_TRIALS = [0, 1, 4, 6, 7]

ENTITY_LABELS = ["CROP", "VAR", "TRT", "GST", "GENE", "QTL", "MRK", "CHR", "BM", "CROSS", "ABS", "BIS"]

SYSTEM = (
    "You are a strict information-extraction auditor for cereal/millet breeding "
    "literature. You judge whether candidate entities should be added to an "
    "existing high-precision extraction. Return strict JSON only."
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

Audit rules:
1. Judge each ADD_CANDIDATE independently against TEXT.
2. Keep only if the candidate span is a contiguous substring and the label matches the local meaning.
3. Existing strict entities are already accepted; use them only as context, not as a reason to reject a valid missing entity.
4. Drop generic words, unsupported spans, duplicate/overlapping variants that do not improve extraction, or labels inferred only from topic.
5. If the entity is real but span or label is slightly wrong, action="fix" and provide corrected.
6. Prefer precision over recall. A weak candidate should be dropped.
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


def load_jsonl(path):
    rows = []
    path = Path(path)
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def parse_json_lenient(text):
    text = str(text or "").strip()
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


def entity_key(entity):
    return int(entity["start"]), int(entity["end"]), str(entity["label"])


def entity_text_key(entity):
    return str(entity.get("text", "")), str(entity.get("label", ""))


def candidate_id(trial, record_index, entity):
    raw = f"{trial}\0{record_index}\0{entity['start']}\0{entity['end']}\0{entity['label']}\0{entity.get('text','')}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def normalize_action(action):
    action = str(action or "").strip().lower()
    return action if action in {"keep", "drop", "fix"} else "drop"


def confidence(decision):
    try:
        return float(decision.get("confidence", 0) or 0)
    except Exception:
        return 0.0


def valid_corrected_entity(obj, text):
    if not isinstance(obj, dict):
        return False
    label = str(obj.get("label", ""))
    if label not in ENTITY_LABELS:
        return False
    ent_text = str(obj.get("text", ""))
    if not ent_text:
        return False
    start = obj.get("start")
    end = obj.get("end")
    if isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(text):
        return text[start:end] == ent_text
    return ent_text in text


def resolve_corrected_entity(obj, text):
    ent_text = str(obj.get("text", ""))
    label = str(obj.get("label", ""))
    start = obj.get("start")
    end = obj.get("end")
    if not (isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(text) and text[start:end] == ent_text):
        found = text.find(ent_text)
        if found < 0:
            return None
        start = found
        end = found + len(ent_text)
    return {"start": start, "end": end, "text": ent_text, "label": label}


class ChatClient:
    def __init__(self, api_key, base_url, model, timeout, retries, system=None):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.retries = retries
        self.system = system or SYSTEM

    def chat(self, prompt, max_tokens):
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": self.system},
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
        raise RuntimeError(f"AI audit failed after {self.retries} retries: {last_err}")


def prompt_hash(prompt, model):
    return hashlib.sha256((model + "\0" + prompt).encode("utf-8")).hexdigest()


def chunked(items, size):
    for i in range(0, len(items), size):
        yield i // size, items[i : i + size]


def context_snippet(text, start, end, window=90):
    left = max(0, start - window)
    right = min(len(text), end + window)
    prefix = "..." if left > 0 else ""
    suffix = "..." if right < len(text) else ""
    return prefix + text[left:right] + suffix


def build_example_index(trials):
    examples = defaultdict(list)
    for trial in trials:
        gold = load_json(S100_T10 / f"trial_{trial:02d}" / "gold.json")
        for record_index, record in enumerate(gold):
            text = record.get("text", "")
            for entity in record.get("entities", []) or []:
                label = str(entity.get("label", ""))
                if label not in ENTITY_LABELS:
                    continue
                start = int(entity.get("start"))
                end = int(entity.get("end"))
                examples[label].append(
                    {
                        "trial": trial,
                        "record_index": record_index,
                        "text": entity.get("text", ""),
                        "start": start,
                        "end": end,
                        "label": label,
                        "context": context_snippet(text, start, end),
                    }
                )
    return examples


def select_label_examples(example_index, labels, current_trial, current_record_index, per_label):
    selected = []
    for label in sorted(labels):
        count = 0
        for example in example_index.get(label, []):
            if example["trial"] == current_trial and example["record_index"] == current_record_index:
                continue
            selected.append(
                {
                    "label": label,
                    "entity": example["text"],
                    "context": example["context"],
                }
            )
            count += 1
            if count >= per_label:
                break
    return selected


def build_prompt(text, strict_entities, candidates, label_examples=None):
    compact_strict = [
        {"text": e.get("text", ""), "start": e.get("start"), "end": e.get("end"), "label": e.get("label", "")}
        for e in strict_entities
    ]
    compact_candidates = [
        {
            "id": c["candidate_id"],
            "text": c["text"],
            "start": c["start"],
            "end": c["end"],
            "label": c["label"],
        }
        for c in candidates
    ]
    examples_block = ""
    if label_examples:
        examples_block = (
            "\nVALIDATION_GOLD_LABEL_EXAMPLES:\n"
            + json.dumps(label_examples, ensure_ascii=False)
            + "\n"
            + "Use these as label semantics examples only. They are not candidates for the current TEXT.\n"
        )
    return f"""{GUIDE}

TEXT:
{text}

EXISTING_STRICT_ENTITIES:
{json.dumps(compact_strict, ensure_ascii=False)}
{examples_block}

ADD_CANDIDATES:
{json.dumps(compact_candidates, ensure_ascii=False)}

Return this JSON shape:
{{
  "decisions": [
    {{
      "id": "<candidate id>",
      "action": "keep|drop|fix",
      "confidence": 0.0,
      "reason": "short reason",
      "corrected": null
    }}
  ]
}}

For fixed entities, corrected must be {{"text": "...", "label": "...", "start": <int|null>, "end": <int|null>}}.
"""


def pair_dir(trial, route):
    return PAIR_ROOT / f"trial_{trial:02d}__{route}"


def make_candidates(trials, entity_labels=None):
    label_allow = set(entity_labels or [])
    rows = []
    by_record = defaultdict(list)
    stats = Counter()
    for trial in trials:
        strict = load_json(pair_dir(trial, "strict3") / "filtered_pred.json")
        union = load_json(pair_dir(trial, "union") / "filtered_pred.json")
        for record_index, (strict_row, union_row) in enumerate(zip(strict, union)):
            strict_keys = {entity_key(e) for e in strict_row.get("entities", []) or []}
            seen = set(strict_keys)
            for entity in union_row.get("entities", []) or []:
                key = entity_key(entity)
                if key in seen:
                    continue
                seen.add(key)
                label = str(entity.get("label", ""))
                if label_allow and label not in label_allow:
                    continue
                row = {
                    "candidate_id": candidate_id(trial, record_index, entity),
                    "trial": trial,
                    "record_index": record_index,
                    "text": str(entity.get("text", "")),
                    "start": int(entity.get("start")),
                    "end": int(entity.get("end")),
                    "label": label,
                }
                rows.append(row)
                by_record[(trial, record_index)].append(row)
                stats[label] += 1
    return rows, by_record, stats


def write_candidate_files(trials, entity_labels):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows, by_record, stats = make_candidates(trials, entity_labels)
    dump_json(OUT_DIR / "union_only_entity_candidates.json", rows)
    dump_json(
        OUT_DIR / "candidate_summary.json",
        {
            "trials": trials,
            "entity_labels": entity_labels or "all",
            "candidates": len(rows),
            "records_with_candidates": len(by_record),
            "by_label": dict(sorted(stats.items())),
        },
    )
    return rows, by_record, stats


def judgment_path():
    return OUT_DIR / "union_only_entity_ai_judgments.jsonl"


def done_batches(path):
    done = set()
    for row in load_jsonl(path):
        done.add((int(row.get("trial")), int(row.get("record_index")), int(row.get("batch_index"))))
    return done


def run_online(args):
    rows, by_record, stats = write_candidate_files(args.trials, args.entity_labels)
    done = done_batches(judgment_path())
    example_index = None if args.no_label_examples else build_example_index(args.trials)
    jobs = []
    planned_items = 0
    for (trial, record_index), candidates in sorted(by_record.items()):
        if args.max_records is not None and len(jobs) >= args.max_records:
            break
        planned_items += len(candidates)
        for batch_index, batch in chunked(candidates, args.batch_items):
            if (trial, record_index, batch_index) not in done:
                jobs.append((trial, record_index, batch_index, batch))
    print(
        f"[PLAN] candidates={len(rows)} labels={dict(sorted(stats.items()))} "
        f"pending_batches={len(jobs)} planned_items={planned_items} model={args.model} "
        f"examples_per_label={0 if args.no_label_examples else args.examples_per_label}",
        flush=True,
    )
    if args.dry_run:
        return
    api_key = os.getenv(args.api_key_env) or os.getenv("API_KEY")
    if not api_key:
        raise SystemExit(f"missing API key env: {args.api_key_env} or API_KEY")
    client = ChatClient(api_key, args.base_url, args.model, args.timeout, args.retries)
    write_lock = threading.Lock()
    strict_by_trial = {
        trial: load_json(pair_dir(trial, "strict3") / "filtered_pred.json")
        for trial in sorted({trial for trial, _, _, _ in jobs})
    }

    def audit_job(job):
        trial, record_index, batch_index, batch = job
        strict_row = strict_by_trial[trial][record_index]
        label_examples = []
        if example_index is not None and args.examples_per_label > 0:
            label_examples = select_label_examples(
                example_index,
                {c["label"] for c in batch},
                trial,
                record_index,
                args.examples_per_label,
            )
        prompt = build_prompt(strict_row["text"], strict_row.get("entities", []) or [], batch, label_examples)
        raw = client.chat(prompt, args.max_tokens)
        parsed = parse_json_lenient(raw)
        if not isinstance(parsed, dict) or not isinstance(parsed.get("decisions"), list):
            parsed = {"decisions": [], "parse_error": True}
        out = {
            "trial": trial,
            "record_index": record_index,
            "batch_index": batch_index,
            "prompt_hash": prompt_hash(prompt, args.model),
            "model": args.model,
            "base_url": args.base_url,
            "examples_per_label": 0 if args.no_label_examples else args.examples_per_label,
            "candidate_ids": [c["candidate_id"] for c in batch],
            "decisions": parsed.get("decisions", []),
        }
        if parsed.get("parse_error"):
            out["parse_error"] = True
        if args.keep_raw:
            out["raw"] = raw
        return out

    completed = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(audit_job, job): job for job in jobs}
        for fut in as_completed(futures):
            job = futures[fut]
            try:
                out = fut.result()
            except Exception as exc:
                failed += 1
                trial, record_index, batch_index, _ = job
                print(
                    json.dumps(
                        {
                            "trial": trial,
                            "record_index": record_index,
                            "batch_index": batch_index,
                            "error": repr(exc),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                continue
            with write_lock:
                append_jsonl(judgment_path(), out)
            completed += 1
            if completed % 10 == 0 or completed + failed == len(jobs):
                print(
                    f"  progress completed={completed} failed={failed}/{len(jobs)} -> {judgment_path()}",
                    flush=True,
                )


def collect_decisions():
    latest = {}
    for row in load_jsonl(judgment_path()):
        for decision in row.get("decisions", []) or []:
            cid = str(decision.get("id", "")).strip()
            if cid:
                latest[cid] = decision
    out = defaultdict(list)
    for cid, decision in latest.items():
        out[cid].append(decision)
    return out


def choose_decision(decisions, keep_threshold, fix_threshold):
    if not decisions:
        return {"action": "drop", "confidence": 0.0, "reason": "missing judgment"}
    fixes = [d for d in decisions if normalize_action(d.get("action")) == "fix" and confidence(d) >= fix_threshold]
    keeps = [d for d in decisions if normalize_action(d.get("action")) == "keep" and confidence(d) >= keep_threshold]
    drops = [d for d in decisions if normalize_action(d.get("action")) == "drop"]
    if fixes:
        return max(fixes, key=confidence)
    if keeps:
        return max(keeps, key=confidence)
    if drops:
        return max(drops, key=confidence)
    return max(decisions, key=confidence)


def add_entities(base, additions):
    out = []
    changed = 0
    for record, ents in zip(base, additions):
        row = dict(record)
        existing = {entity_key(e) for e in row.get("entities", []) or []}
        text_seen = {entity_text_key(e) for e in row.get("entities", []) or []}
        merged = list(row.get("entities", []) or [])
        for entity in ents:
            key = entity_key(entity)
            tkey = entity_text_key(entity)
            if key in existing or tkey in text_seen:
                continue
            merged.append(entity)
            existing.add(key)
            text_seen.add(tkey)
            changed += 1
        row["entities"] = sorted(merged, key=lambda e: (int(e["start"]), int(e["end"]), str(e["label"])))
        out.append(row)
    return out, changed


def evaluate_if_available(trials, outputs_by_trial):
    sys.path.insert(0, str(SOLUTION_SRC))
    from common import evaluate  # noqa: E402

    rows = []
    for trial in trials:
        gold = load_json(S100_T10 / f"trial_{trial:02d}" / "gold.json")
        pred = outputs_by_trial[trial]
        m = evaluate(gold, pred)
        rows.append(
            {
                "trial": trial,
                "total": m["total_score"],
                "ner": m["score_ner"],
                "re": m["score_re"],
                "entity": m["entity"],
                "relation": m["relation"],
            }
        )
    return {
        "rows": rows,
        "mean_total": sum(r["total"] for r in rows) / len(rows),
        "mean_ner": sum(r["ner"] for r in rows) / len(rows),
        "mean_re": sum(r["re"] for r in rows) / len(rows),
        "entity_tp": sum(r["entity"]["tp"] for r in rows),
        "entity_fp": sum(r["entity"]["fp"] for r in rows),
        "entity_fn": sum(r["entity"]["fn"] for r in rows),
        "relation_tp": sum(r["relation"]["tp"] for r in rows),
        "relation_fp": sum(r["relation"]["fp"] for r in rows),
        "relation_fn": sum(r["relation"]["fn"] for r in rows),
    }


def write_zip(json_path, zip_path):
    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(json_path, "submit.json")


def apply_augments(args):
    rows, by_record, stats = write_candidate_files(args.trials, args.entity_labels)
    by_id = {row["candidate_id"]: row for row in rows}
    decisions = collect_decisions()
    outputs_by_trial = {}
    apply_stats = Counter()
    by_label = Counter()
    decision_rows = []

    for trial in args.trials:
        strict = load_json(pair_dir(trial, "strict3") / "filtered_pred.json")
        additions = [[] for _ in strict]
        for row in rows:
            if row["trial"] != trial:
                continue
            decision = choose_decision(decisions.get(row["candidate_id"], []), args.keep_threshold, args.fix_threshold)
            action = normalize_action(decision.get("action"))
            entity = None
            if action == "keep":
                entity = {
                    "start": row["start"],
                    "end": row["end"],
                    "text": row["text"],
                    "label": row["label"],
                }
            elif action == "fix" and valid_corrected_entity(decision.get("corrected"), strict[row["record_index"]]["text"]):
                entity = resolve_corrected_entity(decision["corrected"], strict[row["record_index"]]["text"])
            if entity:
                additions[row["record_index"]].append(entity)
                apply_stats["added"] += 1
                by_label[entity["label"]] += 1
            else:
                apply_stats["rejected"] += 1
            decision_rows.append(
                {
                    **row,
                    "action": action,
                    "confidence": confidence(decision),
                    "reason": decision.get("reason", ""),
                    "applied": bool(entity),
                    "applied_entity": entity,
                }
            )
        outputs_by_trial[trial], changed = add_entities(strict, additions)
        apply_stats["changed_entities"] += changed
        out_json = OUT_DIR / f"trial_{trial:02d}_strict3_union_entity_ai_augmented.json"
        out_zip = OUT_DIR / f"trial_{trial:02d}_strict3_union_entity_ai_augmented.zip"
        dump_json(out_json, outputs_by_trial[trial])
        write_zip(out_json, out_zip)

    dump_json(OUT_DIR / "applied_candidate_decisions.json", decision_rows)
    summary = {
        "trials": args.trials,
        "entity_labels": args.entity_labels or "all",
        "candidate_count": len(rows),
        "judged_candidates": sum(1 for row in rows if row["candidate_id"] in decisions),
        "apply_stats": dict(sorted(apply_stats.items())),
        "added_by_label": dict(sorted(by_label.items())),
    }
    try:
        summary["evaluation"] = evaluate_if_available(args.trials, outputs_by_trial)
    except Exception as exc:
        summary["evaluation_error"] = repr(exc)
    dump_json(OUT_DIR / "augment_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def summarize_judgments():
    action = Counter()
    labels = Counter()
    parse_errors = 0
    rows = load_jsonl(judgment_path())
    candidate_rows = {r["candidate_id"]: r for r in load_json(OUT_DIR / "union_only_entity_candidates.json")} if (OUT_DIR / "union_only_entity_candidates.json").exists() else {}
    latest = {}
    for row in rows:
        if row.get("parse_error"):
            parse_errors += 1
        for decision in row.get("decisions", []) or []:
            cid = str(decision.get("id", ""))
            if cid:
                latest[cid] = decision
    for cid, decision in latest.items():
        act = normalize_action(decision.get("action"))
        action[act] += 1
        if cid in candidate_rows:
            labels[candidate_rows[cid]["label"] + ":" + act] += 1
    summary = {
        "judgment_batches": len(rows),
        "unique_decision_ids": len(latest),
        "parse_errors": parse_errors,
        "decisions_by_action": dict(sorted(action.items())),
        "label_action_counts": dict(sorted(labels.items())),
    }
    dump_json(OUT_DIR / "judgment_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, nargs="+", default=DEFAULT_TRIALS)
    parser.add_argument("--entity_labels", nargs="+", default=None, help="Optional union-only entity labels to audit")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--online", action="store_true")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--model", default=os.getenv("OPUS_MODEL", "claude-opus-4-8"))
    parser.add_argument("--base_url", default=os.getenv("OPUS_BASE_URL", os.getenv("BASE_URL", "https://zjapi.com")))
    parser.add_argument("--api_key_env", default="OPUS_API_KEY")
    parser.add_argument("--timeout", type=int, default=int(os.getenv("OPUS_TIMEOUT", "180")))
    parser.add_argument("--retries", type=int, default=int(os.getenv("OPUS_RETRIES", "3")))
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--batch_items", type=int, default=20)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max_records", type=int, default=None, help="Limit pending record batches during online auditing")
    parser.add_argument("--examples_per_label", type=int, default=3, help="Gold validation examples included per candidate label")
    parser.add_argument("--no_label_examples", action="store_true", help="Disable validation label examples in the audit prompt")
    parser.add_argument("--keep_threshold", type=float, default=0.80)
    parser.add_argument("--fix_threshold", type=float, default=0.80)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--keep_raw", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not any([args.prepare, args.online, args.apply, args.summary]):
        args.prepare = True
    if args.prepare:
        rows, by_record, stats = write_candidate_files(args.trials, args.entity_labels)
        print(json.dumps({
            "out_dir": str(OUT_DIR),
            "candidates": len(rows),
            "records_with_candidates": len(by_record),
            "by_label": dict(sorted(stats.items())),
        }, ensure_ascii=False, indent=2))
    if args.online:
        run_online(args)
    if args.summary:
        summarize_judgments()
    if args.apply:
        apply_augments(args)


if __name__ == "__main__":
    main()
