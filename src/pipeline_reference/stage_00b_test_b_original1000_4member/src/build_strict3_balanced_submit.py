"""Build the Test-B original1000 + strict3 + balanced submission package.

This script assumes the 4-member strict3 ensemble has already been generated:

  outputs/testB_original1000_4member_strict3_submit.zip

It then applies the relation sanity filter in balanced mode and verifies that
the resulting zip contains a valid submit.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path


DEFAULT_SOLUTION = Path(r"K:\浩然\CCL\solution\solution")
INPUT_NAME = "testB_original1000_4member_strict3_submit.zip"
OUTPUT_JSON_NAME = "testB_original1000_4member_strict3_relfilter_balanced_submit.json"
OUTPUT_ZIP_NAME = "testB_original1000_4member_strict3_relfilter_balanced_submit.zip"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_submit_zip(path: Path) -> dict:
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        if names != ["submit.json"]:
            raise ValueError(f"Unexpected zip entries: {names!r}")
        data = json.loads(zf.read("submit.json").decode("utf-8"))

    return {
        "zip": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "records": len(data),
        "entities": sum(len(row.get("entities", [])) for row in data),
        "relations": sum(len(row.get("relations", [])) for row in data),
        "empty_records": [
            i for i, row in enumerate(data)
            if not row.get("entities") and not row.get("relations")
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solution", default=str(DEFAULT_SOLUTION))
    args = parser.parse_args()

    solution = Path(args.solution)
    input_zip = solution / "outputs" / INPUT_NAME
    output_json = solution / "outputs" / OUTPUT_JSON_NAME
    output_zip = solution / "outputs" / OUTPUT_ZIP_NAME

    if not input_zip.exists():
        raise FileNotFoundError(
            f"Missing strict3 input: {input_zip}. "
            "Generate original1000 4-member strict3 first."
        )

    cmd = [
        sys.executable,
        str(solution / "src" / "relation_sanity_filter.py"),
        "--input",
        str(input_zip),
        "--output",
        str(output_json),
        "--zip_out",
        str(output_zip),
        "--mode",
        "balanced",
    ]
    subprocess.run(cmd, cwd=solution, check=True)

    stats = verify_submit_zip(output_zip)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
