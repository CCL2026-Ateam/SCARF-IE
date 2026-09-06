# Method Overview

## Models

最高分 pipeline 使用以下模型配置：

```env
BASE_URL=https://zjapi.com
MODEL_MAIN=gpt-5.4
MODEL_CHEAP=gpt-5.4-mini
MODEL_SECONDARY=gpt-5.5-openai-compact
TEMPERATURE=0.0
```

代码请求的 OpenAI-compatible endpoint 是 `https://zjapi.com/v1/chat/completions`。

主抽取 ensemble 包含四个成员：

- `main_k4`: `gpt-5.4`
- `main_k4_conv`: `gpt-5.4` + boundary conventions
- `secondary_k4`: `gpt-5.5-openai-compact`
- `secondary_k4_conv`: `gpt-5.5-openai-compact` + boundary conventions

`gpt-5.4-mini` 是 cheap/auxiliary 配置，主要用于辅助或连通性检查。

## Prompt

抽取 prompt 位于：

```text
src/pipeline_reference/stage_00_external_project_src/prompts.py
```

它要求模型作为严格的信息抽取系统，从谷物/小米育种文献文本中抽取：

- 实体：`CROP`、`VAR`、`TRT`、`GST`、`GENE`、`QTL`、`MRK`、`CHR`、`BM`、`CROSS`、`ABS`、`BIS`
- 关系：`CON`、`USE`、`HAS`、`AFF`、`OCI`、`LOI`

输出必须是 JSON，实体必须使用原文精确 span，关系必须连接已抽取实体。`_conv` 成员在 prompt 中加入边界约定，用于提高实体 span 一致性。

候选增强和 OPS 审核使用 keep/drop/fix 审核式 prompt，而不是自由生成完整记录。相关脚本：

- `stage_04_testb_filter_then_augment_code/run_filter_then_union_testb.py`
- `stage_05_ops_relation_prune_code/run_ops_audit_sharded.py`
- `stage_03_record_level_filter_prune_code/apply_entity_pair_strategy.py`

## Voting

投票逻辑位于：

```text
src/pipeline_reference/stage_00_external_project_src/ensemble.py
```

Key 定义：

- 实体 key：`(start, end, label)`
- 关系 key：`(head_start, head_end, head_type, tail_start, tail_end, tail_type, label)`

关系保留时要求两端实体也被保留。

线上校准后的四成员阈值位于 `run_online_best_submit.py`：

- 实体默认阈值：`3`
- 实体覆盖：`ABS=2`、`BIS=2`、`CHR=2`、`CROP=2`、`CROSS=4`、`GST=2`、`MRK=4`、`QTL=2`
- 关系默认阈值：`2`
- 关系覆盖：`AFF=3`、`CON=1`、`HAS=3`、`LOI=2`、`OCI=2`、`USE=5`

## Pipeline

1. 准备官方 `pool/dev/original1000/test_B`。
2. 使用 `pool` 作为 few-shot 检索池，运行四成员 Test-B 抽取，生成 base ensemble。
3. 对 base 四成员取 `vote >= 1` 并集，生成 union candidate pool。
4. 使用 `original1000` 作为 few-shot 检索池，再运行四成员抽取，并取 `relation vote >= 3` 作为 strict3 关系来源。
5. 使用 filter-then-union 对 union candidates 做 LLM 审核和补充。
6. 对增强结果做 OPS2 full audit，移除高置信错误实体/关系。
7. 做 relation-only OPS 剪枝，默认 12 轮，可通过 `--relation-prune-rounds` 调整。
8. 最终合成：保留剪枝后的实体，使用 OPS2 关系基础，删除 `USE`，补回 `HAS`，再从 strict3 中补入端点已存在的 `AFF/CON/LOI/OCI`。

## Final Rules

最终规则由 `src/run_from_scratch.py` 的 `final_synthesis()` 执行：

- entity source：relation-prune 输出。
- relation source：ops2 输出。
- HAS source：relation-prune 输出。
- strict3 source：original1000 strict3 conservative 输出。
- 删除所有 `USE`。
- 从 strict3 只补 `AFF/CON/LOI/OCI`。
- 所有关系都必须满足两端实体存在。

历史最高分参考文件的最终分布：

- 实体：4790
- 关系：2006
- `AFF=697`
- `CON=454`
- `HAS=267`
- `LOI=491`
- `OCI=97`
- `USE=0`
