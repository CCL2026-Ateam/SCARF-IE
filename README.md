# SCARF-IE

### Schema-Constrained Arbitration and Relation Fusion for Information Extraction

Training-free information extraction system for the **CCL2026-MGBIE Track-A** no-finetuning track. SCARF-IE combines retrieval-conditioned proposal views, multi-member voting, schema-constrained candidate arbitration, full-graph auditing, and entity-closed relation fusion.

[![Task](https://img.shields.io/badge/task-information%20extraction-2f80ed)](#task) [![Track](https://img.shields.io/badge/CCL2026-MGBIE-6c5ce7)](#results) [![Training](https://img.shields.io/badge/training-free-27ae60)](#method-overview)

## GitHub repository description

> Training-free schema-constrained candidate arbitration and entity-closed relation fusion for CCL2026-MGBIE.

## Overview

Large language models can improve candidate coverage in domain information extraction, but diversified prompting also introduces span drift, type conflicts, unsupported relations, and invalid relation endpoints. SCARF-IE separates candidate proposal from final graph commitment:

1. retrieve task-relevant few-shot examples from official annotated data;
2. run multiple extraction members with complementary prompt conventions;
3. build a precision-oriented backbone and a recall-oriented candidate union;
4. restrict an LLM arbiter to indexed `keep`, `drop`, or schema-valid local `fix` actions;
5. audit the full graph and freeze the final entity set;
6. monotonically prune relations and fuse only entity-closed, schema-valid edges.

```text
Official data
     ↓
Retrieval-conditioned multi-member extraction
     ↓
Voting backbone + high-recall candidate union
     ↓
Schema-constrained keep / drop / fix arbitration
     ↓
Full-graph audit and relation-only pruning
     ↓
Entity-closed final relation fusion
     ↓
submit.json / submit.zip
```

## Results

The archived reference submission reports the following online result:

| Metric | Score |
|---|---:|
| Overall | **0.4967** |
| NER | **0.7358** |
| RE | **0.3373** |

The reference files are available under [`outputs/reference/`](outputs/reference/). The archived output is provided for comparison; a fresh reproduction should rerun the pipeline from the official data.

## Task

The system extracts entities and relations from grain and millet breeding literature.

### Entity types

`CROP` · `VAR` · `TRT` · `GST` · `GENE` · `QTL` · `MRK` · `CHR` · `BM` · `CROSS` · `ABS` · `BIS`

### Relation types

`CON` · `USE` · `HAS` · `AFF` · `OCI` · `LOI`

## Method overview

### Retrieval-conditioned proposal views

The pipeline constructs few-shot retrieval pools from official annotated training data. Different extraction members use complementary model and boundary-convention settings to improve candidate diversity and span consistency.

### Candidate arbitration

The high-recall union is not directly committed to the final graph. Instead, the arbiter makes local indexed decisions:

```text
keep  → retain the candidate
drop  → remove the candidate
fix   → apply a schema-valid local correction
```

This avoids unconstrained regeneration of complete records and makes the correction process auditable.

### Entity-closed relation fusion

After the full-graph audit, the entity set is frozen. Relation-only refinement can remove unsupported edges but cannot reopen entity extraction. A relation is added to the final result only when its endpoints exist in the final entity set and the relation passes the corresponding schema, voting, and arbitration checks.

## Repository structure

```text
.
├── config/
│   └── model.env.example
├── docs/
│   ├── method_overview.md
│   ├── reproduction_steps.md
│   ├── organizer_reproduction_from_scratch.md
│   └── external_data_api_rules.md
├── outputs/reference/
│   ├── submit_highscore_4967.json
│   ├── submit_highscore_4967.zip
│   └── submit_highscore_4967_summary.json
├── src/
│   ├── run_from_scratch.py
│   └── pipeline_reference/
├── requirements.txt
└── README.md
```

## Installation

Python 3.10 or later is recommended.

```bash
pip install -r requirements.txt
```

## Configuration

The pipeline calls an OpenAI-compatible chat-completions service. Provide the API key through an environment variable; do not commit credentials to GitHub.

Linux/macOS:

```bash
export API_KEY="<your_api_key>"
```

Windows PowerShell:

```powershell
$env:API_KEY="<your_api_key>"
```

Model and endpoint examples are provided in [`config/model.env.example`](config/model.env.example). An alternative service endpoint can be supplied with `--base-url`.

## Reproduction

Place the official competition files in the parent directory:

```text
../train.zip
../test_B.zip
```

Run an API smoke test:

```bash
python src/run_from_scratch.py --stage smoke-api --api-key-env API_KEY
```

Run the complete pipeline:

```bash
python src/run_from_scratch.py --stage full --api-key-env API_KEY
```

The generated files will be written to:

```text
outputs/from_scratch/submit.json
outputs/from_scratch/submit.zip
outputs/from_scratch/submit_summary.json
```

To specify custom data paths:

```bash
python src/run_from_scratch.py --stage full \
  --train-zip /path/to/train.zip \
  --test-zip /path/to/test_B.zip \
  --api-key-env API_KEY
```

For stage-by-stage commands and organizer-facing instructions, see [`docs/`](docs/).

## Reproducibility and security

- Official competition data and API credentials are not included in this repository.
- Do not commit real API keys, `.env` files, local caches, or generated `work/` files.
- Intermediate files and caches are excluded by [`.gitignore`](.gitignore).
- Online reproduction may vary slightly with model versions, service load, retry behavior, and structured-output stability.
- The archived score is a reference result, not a guarantee that every online rerun will produce byte-identical outputs.

## License and data use

Please follow the CCL2026-MGBIE competition rules and the licenses of all datasets, models, and services used for reproduction. Add the project-specific license before making the repository public if required by your institution or competition organizer.


