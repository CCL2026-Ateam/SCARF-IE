# Organizer Reproduction From Scratch

本文说明举办方如何使用官方数据和 API 从头复现最高分方案。这里的“从头复现”指重新调用模型完成 Test-B 推理、投票、候选增强、OPS 审核剪枝和最终合成。

## 1. 环境

建议使用 Python 3.10 或更高版本。安装依赖：

```powershell
cd competition_submission_highscore_4967
pip install -r requirements.txt
```

本方案主要依赖标准库、`requests`、`tqdm`、`tenacity`、`python-dotenv`。不需要额外训练模型。

## 2. API 与模型

通过环境变量配置 API key：

```powershell
$env:API_KEY="<organizer_api_key>"
```

默认模型配置已经写在代码和 `config/model.env.example` 中：

```env
BASE_URL=https://zjapi.com
MODEL_MAIN=gpt-5.4
MODEL_CHEAP=gpt-5.4-mini
MODEL_SECONDARY=gpt-5.5-openai-compact
TEMPERATURE=0.0
```

实际 endpoint 是：

```text
https://zjapi.com/v1/chat/completions
```

历史脚本中保留了 `OPUS_MODEL`、`OPUS_API_KEY` 等变量名。复核时这些变量会由 `src/run_from_scratch.py` 自动映射到 `gpt-5.4` 和同一个 API key；变量名只是历史沿用，不表示使用其他厂商模型。

## 3. 官方数据

默认读取：

```text
../train.zip
../test_B.zip
```

也可以显式指定：

```powershell
python src\run_from_scratch.py --stage full `
  --train-zip D:\path\to\train.zip `
  --test-zip D:\path\to\test_B.zip `
  --api-key-env API_KEY
```

入口脚本会自动生成：

- `work/from_scratch/data/pool.json`：训练集前 800 条，作为 few-shot 检索池。
- `work/from_scratch/data/dev.json`：训练集后 200 条。
- `work/from_scratch/data/original_1000_pool.json`：官方 1000 条标注记录，用于 original1000 strict3。
- `work/from_scratch/dataset/test_B.json`：官方 Test-B 输入。

## 4. 完整复现命令

```powershell
cd competition_submission_highscore_4967
$env:API_KEY="<organizer_api_key>"
python src\run_from_scratch.py --stage full --api-key-env API_KEY
```

最终输出：

```text
outputs/from_scratch/submit.json
outputs/from_scratch/submit.zip
outputs/from_scratch/submit_summary.json
```

线上 LLM 服务可能因负载、模型后端版本或 JSON 格式稳定性导致少量差异。复核时应以相同模型名、相同中转站、相同 prompts、相同投票和后处理规则重新运行，并使用官方评分脚本评分。

## 5. Pipeline 阶段

### 5.1 base-ensemble

运行四个 Test-B 抽取成员：

- `main_k4`: `gpt-5.4`
- `main_k4_conv`: `gpt-5.4` + boundary conventions
- `secondary_k4`: `gpt-5.5-openai-compact`
- `secondary_k4_conv`: `gpt-5.5-openai-compact` + boundary conventions

Prompt 位于 `src/pipeline_reference/stage_00_external_project_src/prompts.py`。四成员结果按 `run_online_best_submit.py` 中的线上校准阈值投票。

### 5.2 union-pool

对 base 四成员做 `entity vote >= 1`、`relation vote >= 1` 的并集，生成 Test-B 候选池。该候选池供后续 filter-then-union 增强使用。

### 5.3 original1000-strict3

使用官方 1000 条标注记录作为 few-shot 检索池，再运行同样的四成员 Test-B 抽取。随后取 `relation vote >= 3` 的 strict3 关系来源，并做 conservative relation sanity filter。

### 5.4 augment

`stage_04_testb_filter_then_augment_code/run_filter_then_union_testb.py` 对 union 候选做 LLM 审核，保留通过阈值的实体和关系：

```text
entity_dedup=span_only
relation_labels=AFF CON HAS LOI OCI USE
entity_keep_threshold=0.92
entity_fix_threshold=0.92
relation_keep_threshold=0.92
relation_fix_threshold=0.92
model=gpt-5.4
```

### 5.5 ops2

对增强结果运行 OPS item audit，再用 `apply_entity_pair_strategy.py` 应用策略：

```text
entity_policy=drop_threshold
entity_drop_threshold=0.75
relation_policy=drop_threshold
relation_drop_threshold=0.95
apply_entity_fix=true
apply_relation_fix=false
```

### 5.6 relation-prune

默认再运行 12 轮 relation-only OPS 剪枝，固定实体，仅移除高置信 drop 的关系：

```text
entity_policy=keep_all
relation_policy=drop_threshold
relation_drop_threshold=0.95
protect_relation_endpoints=true
```

小规模调试时可以临时降低轮数，例如：

```powershell
python src\run_from_scratch.py --stage relation-prune --relation-prune-rounds 2 --api-key-env API_KEY
```

正式复核最高分路线应使用默认 `--relation-prune-rounds 12`。

### 5.7 final

最终合成规则：

1. 使用 relation-prune 输出中的实体集。
2. 使用 ops2 输出中的关系作为基础。
3. 删除所有 `USE` 关系。
4. 从 relation-prune 输出中保留/补回 `HAS`。
5. 从 original1000 strict3 conservative 来源中补入 `AFF/CON/LOI/OCI`，但仅当关系两端实体已在最终实体集中存在。
6. 写出 `outputs/from_scratch/submit.json` 和 `submit.zip`。

## 6. 常用断点命令

API 连通性：

```powershell
$env:API_KEY="<organizer_api_key>"
python src\run_from_scratch.py --stage smoke-api
```

只准备官方数据：

```powershell
python src\run_from_scratch.py --stage prepare
```

只跑基础四成员：

```powershell
$env:API_KEY="<organizer_api_key>"
python src\run_from_scratch.py --stage base-ensemble --api-key-env API_KEY
```

小规模调试：

```powershell
$env:API_KEY="<organizer_api_key>"
python src\run_from_scratch.py --stage base-ensemble --limit 1 `
  --record-concurrency 1 --member-concurrency 1 --api-key-env API_KEY
```
