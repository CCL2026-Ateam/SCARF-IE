# Record-Level Filter Experiment Notes

## Current Best

当前已验证最好提交：

`recordlevel_ent075_rel095_entityfix_norelfix_submit.zip`

线上分数：

| Submit time | Total | NER | RE |
|---|---:|---:|---:|
| 2026-06-27 01:50:31 | 0.4780 | 0.7233 | 0.3145 |

策略参数：

```text
entity_policy = drop_threshold
entity_drop_threshold = 0.75
apply_entity_fix = true
entity_fix_threshold = 0.0
relation_policy = drop_threshold
relation_drop_threshold = 0.95
apply_relation_fix = false
protect_relation_endpoints = false
noempty_fallback = true
```

输出统计：

```text
records = 600
entities = 4559
relations = 1791
empty = 0
bad_relation_endpoints = 0
```

相对原始 `highscore_plus_light_add_has_aff1_prune_unseen_relation_types_submit.zip`：

```text
entities: 4782 -> 4559  delta -223
relations: 1932 -> 1791 delta -141
```

## Key Files

原始待筛选预测：

`C:\Users\Zzl410410\Documents\Codex\2026-06-26\019ef541-55e7-7af1-8f34-0927a521fe82\work\highscore_prune_record_audit\highscore_plus_light_add_has_aff1_prune_unseen_relation_types_input_pred.json`

完整 record-level 审核结果：

`C:\Users\Zzl410410\Documents\Codex\2026-06-26\019ef541-55e7-7af1-8f34-0927a521fe82\work\highscore_prune_record_audit\merged_record_level\highscore_plus_light_add_has_aff1_prune_unseen_relation_types_input_pred.opus_item_judgments.jsonl`

筛选脚本：

`C:\Users\Zzl410410\Documents\Codex\2026-06-26\019ef541-55e7-7af1-8f34-0927a521fe82\work\apply_entity_pair_strategy.py`

审核分片/合并脚本：

`C:\Users\Zzl410410\Documents\Codex\2026-06-26\019ef541-55e7-7af1-8f34-0927a521fe82\work\highscore_record_audit_shards.py`

solution 源码目录：

`K:\浩然\CCL\solution\solution\src`

输出目录：

`C:\Users\Zzl410410\Documents\Codex\2026-06-26\019ef541-55e7-7af1-8f34-0927a521fe82\outputs\record_level_highscore_prune`

分数台账：

`C:\Users\Zzl410410\Documents\Codex\2026-06-26\019ef541-55e7-7af1-8f34-0927a521fe82\outputs\record_level_highscore_prune\SCOREBOARD.md`

## How The Filter Works

实体：

- `drop`: 当实体审核结果为 `drop` 且 confidence 达到阈值时删除。
- `fix`: 保留并应用实体修正，要求 `corrected` 是合法实体。
- 当前 best 使用 `entity_drop_threshold = 0.75`。

关系：

- 只做高置信 `drop`。
- 不做 relation fix，即 `apply_relation_fix = false`。
- 当前 best 使用 `relation_drop_threshold = 0.95`。

安全约束：

- `noempty_fallback = true`，避免把某条 record 删成空。
- 生成后必须检查：
  - `empty = 0`
  - `bad_relation_endpoints = 0`

## Reproduce Current Best

PowerShell:

```powershell
$pred='C:\Users\Zzl410410\Documents\Codex\2026-06-26\019ef541-55e7-7af1-8f34-0927a521fe82\work\highscore_prune_record_audit\highscore_plus_light_add_has_aff1_prune_unseen_relation_types_input_pred.json'
$jud='C:\Users\Zzl410410\Documents\Codex\2026-06-26\019ef541-55e7-7af1-8f34-0927a521fe82\work\highscore_prune_record_audit\merged_record_level\highscore_plus_light_add_has_aff1_prune_unseen_relation_types_input_pred.opus_item_judgments.jsonl'
$src='K:\浩然\CCL\solution\solution\src'
$outdir='C:\Users\Zzl410410\Documents\Codex\2026-06-26\019ef541-55e7-7af1-8f34-0927a521fe82\outputs\record_level_highscore_prune'
$name='recordlevel_ent075_rel095_entityfix_norelfix'

python work\apply_entity_pair_strategy.py `
  --pred $pred `
  --judgments $jud `
  --solution_src $src `
  --output_json (Join-Path $outdir ($name + '.json')) `
  --output_zip (Join-Path $outdir ($name + '_submit.zip')) `
  --summary (Join-Path $outdir ($name + '_summary.json')) `
  --preset safe_anydrop_rel098 `
  --entity_policy drop_threshold `
  --entity_drop_threshold 0.75 `
  --apply_entity_fix true `
  --entity_fix_threshold 0.0 `
  --relation_policy drop_threshold `
  --relation_drop_threshold 0.95 `
  --apply_relation_fix false `
  --protect_relation_endpoints false `
  --noempty_fallback true `
  --write_strategy_json (Join-Path $outdir ($name + '_strategy.json'))
```

## Candidate Next Experiments

Use current best as the anchor. Change only one dimension per submit.

Recommended order:

| Name idea | Entity drop | Relation drop | Why |
|---|---:|---:|---|
| `ent070_rel095` | 0.70 | 0.95 | Test if more entity pruning keeps NER/RE rising. |
| `ent080_rel095` | 0.80 | 0.95 | Safer neighbor around current best. |
| `ent075_rel096` | 0.75 | 0.96 | Slightly reduce relation deletion, may protect RE. |
| `ent075_rel094` | 0.75 | 0.94 | More aggressive relation pruning, riskier. |
| `ent070_rel096` | 0.70 | 0.96 | Separate stronger entity pruning from relation risk. |

Avoid for now:

- Enabling `relation_fix`: prior policy intentionally禁用，容易乱改关系类型/端点。
- Large jumps such as `relation_drop <= 0.90`: may over-delete validated RE.
- Changing CON/LOI/OCI/USE distribution manually without online evidence.

## Validation Command

After generating a zip, run this style of check:

```powershell
@'
import json, zipfile
from pathlib import Path
z=Path(r'PUT_SUBMIT_ZIP_HERE')
with zipfile.ZipFile(z) as f:
    data=json.loads(f.read('submit.json').decode('utf-8'))
ents=sum(len(r.get('entities') or []) for r in data)
rels=sum(len(r.get('relations') or []) for r in data)
empty=sum(1 for r in data if not r.get('entities') and not r.get('relations'))
bad=0
for r in data:
    es={(e.get('text'),e.get('start'),e.get('end'),e.get('label')) for e in (r.get('entities') or [])}
    for rel in r.get('relations') or []:
        h=(rel.get('head'),rel.get('head_start'),rel.get('head_end'),rel.get('head_type'))
        t=(rel.get('tail'),rel.get('tail_start'),rel.get('tail_end'),rel.get('tail_type'))
        if h not in es or t not in es:
            bad+=1
print({'records':len(data),'entities':ents,'relations':rels,'empty':empty,'bad_relation_endpoints':bad})
'@ | python -
```

## Audit Notes

Record-level audit was completed for all 600 records:

```text
merged_rows = 600
parse_errors = 0
empty_decision_rows = 0
manual_fallback_records = [206, 315, 401]
```

Manual fallback was only used because the Opus gateway repeatedly returned empty content for those short records. The fallback rows are marked with `manual_fallback: true` in the merged JSONL.

