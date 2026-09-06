# Reproduction Steps

复核应从官方数据和 API 重新运行，不依赖任何预生成中间文件。

## 1. 准备

```powershell
cd competition_submission_highscore_4967
pip install -r requirements.txt
$env:API_KEY="<organizer_api_key>"
```

确认官方数据位置：

```text
../train.zip
../test_B.zip
```

如果数据在其他路径，运行时加 `--train-zip` 和 `--test-zip`。

## 2. API smoke test

```powershell
python src\run_from_scratch.py --stage smoke-api --api-key-env API_KEY
```

该命令会分别检查：

- `gpt-5.4`
- `gpt-5.4-mini`
- `gpt-5.5-openai-compact`

## 3. 完整运行

```powershell
python src\run_from_scratch.py --stage full --api-key-env API_KEY
```

默认输出：

```text
outputs/from_scratch/submit.json
outputs/from_scratch/submit.zip
outputs/from_scratch/submit_summary.json
```

## 4. 分阶段运行

```powershell
python src\run_from_scratch.py --stage prepare
python src\run_from_scratch.py --stage base-ensemble --api-key-env API_KEY
python src\run_from_scratch.py --stage union-pool
python src\run_from_scratch.py --stage original1000-ensemble --api-key-env API_KEY
python src\run_from_scratch.py --stage original1000-strict3
python src\run_from_scratch.py --stage augment --api-key-env API_KEY
python src\run_from_scratch.py --stage ops2 --api-key-env API_KEY
python src\run_from_scratch.py --stage relation-prune --api-key-env API_KEY
python src\run_from_scratch.py --stage final
```

断点运行会复用 `work/from_scratch/` 和 `outputs/from_scratch/` 中已经生成的前序结果。若需要复用已经完成的四成员文件，可加 `--skip-existing`。

## 5. 关键默认参数

```text
BASE_URL=https://zjapi.com
MODEL_MAIN=gpt-5.4
MODEL_CHEAP=gpt-5.4-mini
MODEL_SECONDARY=gpt-5.5-openai-compact
temperature=0.0
kshot=4
retriever=lexical
record_concurrency=16
member_concurrency=4
max_tokens=4096
relation_prune_rounds=12
```

基础投票阈值：

- 实体默认 `3/4`；覆盖：`ABS=2`、`BIS=2`、`CHR=2`、`CROP=2`、`CROSS=4`、`GST=2`、`MRK=4`、`QTL=2`。
- 关系默认 `2/4`；覆盖：`AFF=3`、`CON=1`、`HAS=3`、`LOI=2`、`OCI=2`、`USE=5`。

## 6. 运行时间与稳定性

完整流程会多次调用 LLM API，耗时取决于中转站负载、并发限制和重试次数。若遇到超时，可以降低并发：

```powershell
python src\run_from_scratch.py --stage full --api-key-env API_KEY `
  --record-concurrency 4 --member-concurrency 2 --augment-workers 2 `
  --timeout 300 --max-retries 5
```
