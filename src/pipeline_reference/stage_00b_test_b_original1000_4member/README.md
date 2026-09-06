# Test-B Original-1000 Four-Member Run

Purpose: rerun Test-B with only the original 1000 labeled records, without the
past Test-A pseudo-label augmentation.

The original 1000 records are materialized as:

- `K:\浩然\CCL\solution\solution\outputs\original_1000_pool.json`
- source: `data\pool.json` (800) + `data\dev.json` (200)

All member runs are chunked and recorded under:

- `K:\浩然\CCL\solution\solution\outputs\testB_original1000_*`
- `work\test_b_original1000_4member\outputs\*.log`

Members:

- `main_k4`
- `main_k4_conv`
- `secondary_k4`
- `secondary_k4_conv`

The run temporarily sets `data\pool.json` to `original_1000_pool.json`, launches
the chunked workers, and restores the original 800-record pool after completion
or manual cleanup.
