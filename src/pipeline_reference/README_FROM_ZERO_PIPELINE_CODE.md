# From-Scratch Pipeline Code Reference

This directory contains the historical code needed to understand and rerun the online pipeline from official data.

The organizer-facing reproduction path requires:

- official `pool/dev/test_B` data;
- `BASE_URL=https://zjapi.com`;
- API credentials;
- model names:
  - `MODEL_MAIN=gpt-5.4`
  - `MODEL_CHEAP=gpt-5.4-mini`
  - `MODEL_SECONDARY=gpt-5.5-openai-compact`

Some historical scripts still contain machine-specific absolute paths. When rerunning, replace those paths with the local official data and output directories.

Start with:

- `PIPELINE_STAGE_INDEX.md`
- `../../docs/organizer_reproduction_from_scratch.md`
- `../../docs/method_overview.md`

