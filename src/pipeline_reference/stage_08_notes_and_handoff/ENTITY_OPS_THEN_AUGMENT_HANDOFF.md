# 先实体 OPS -> 再增补 实验方案与代码总结

## 实验目的

验证顺序：

1. 从 `strict3 prune_unseen` 作为高精度底座开始；
2. 先做 OPS 二次筛选，但只筛实体；
3. 再做 AI union-only 增补：实体全标签 + 关系只保留 `CON/LOI`；
4. 对比反向顺序 `先增补 -> 再实体 OPS` 的指标。

最终五折验证集 `trial 0/1/4/6/7` 结论：

`先实体 OPS -> 再增补` 更高。

| 顺序 | total | NER | RE | entity TP/FP/FN | relation TP/FP/FN |
|---|---:|---:|---:|---:|---:|
| 先增补 -> 再实体 OPS | 0.474184 | 0.703798 | 0.321108 | 2445/1490/624 | 392/1002/677 |
| 先实体 OPS -> 再增补 | 0.477376 | 0.711823 | 0.321078 | 2454/1422/615 | 393/1009/676 |

差值：`先实体 OPS -> 再增补` 比 `先增补 -> 再实体 OPS` 高 `+0.003192 total`，NER 高 `+0.008025`，实体 FP 少 `68`，实体 TP 多 `9`。

## 关键实验设定

- 底座：`strict3 prune_unseen`
- 增补实体：全实体标签
  - `CROP VAR TRT GST GENE QTL MRK CHR BM CROSS ABS BIS`
- 增补关系：只输出 `CON/LOI`
- OPS 筛选：
  - `entity_drop_threshold = 0.75`
  - `relation_drop_threshold = 1.01`
  - 关系阈值设为 `1.01` 等价于不筛关系
  - `apply_entity_fix = true`
  - `apply_relation_fix = false`

## 主要代码文件

项目根目录：

`C:\Users\Zzl410410\Documents\Codex\2026-06-24\019ef53b-8c1b-7a92-ad59-c7c456388b6f`

核心脚本：

- `work\test_b_validation_scaling\order_swap_entityonly_experiment.py`
  - 生成并比较两个顺序：
    - `augment_then_filter_input`
    - `filter_then_augment`
  - `filter_then_augment` 就是“先实体 OPS -> 再增补”。
- `work\test_b_validation_scaling\run_holdout_ops_filter_sharded.py`
  - 分片并行 OPS 审核。
  - 用于跑 `先增补 -> 再实体 OPS` 的在线 AI 审核。
- `work\test_b_validation_scaling\run_holdout_ops_filter.py`
  - 定义 OPS variant 路径、应用筛选策略、汇总指标。
- `work\test_b_validation_scaling\union_only_entity_ai_augmenter.py`
  - union-only 实体 AI 审核增补器。
- `work\test_b_validation_scaling\union_only_relation_ai_augmenter.py`
  - relation 增补 CLI。
- `work\test_b_validation_scaling\relation_aug_core.py`
  - relation 增补核心逻辑。
- `work\test_b_validation_scaling\env_loader.py`
  - 新增的 `.env` 加载器，让 Gemini/Claude 通道自动读取环境变量。

## 环境变量

项目根目录和 solution 目录都有 `.env`。

脚本现在会自动加载：

- `GEMINI_API_KEY`
- `GEMINI_BASE_URL`
- `GEMINI_MODEL=gemini-3.1-pro-preview`
- `OPUS_API_KEY`
- `OPUS_BASE_URL`
- `OPUS_MODEL=claude-opus-4-8`

注意：不要在报告或日志里暴露 key。

## 复现实验命令

### 1. 生成两个顺序的中间预测

```powershell
cd C:\Users\Zzl410410\Documents\Codex\2026-06-24\019ef53b-8c1b-7a92-ad59-c7c456388b6f

python work\test_b_validation_scaling\order_swap_entityonly_experiment.py `
  --prepare `
  --trials 0 1 4 6 7
```

生成目录：

`work\test_b_validation_scaling\outputs\order_swap_entityonly_filter`

关键中间结果：

- `entity_filter_base\trial_XX_prune_unseen_entity_only_ops_filtered.json`
- `entity_filter_then_augment\trial_XX_entity_filter_then_entity_all_relation_con_loi_augmented.json`
- `augment_then_entity_filter_input\trial_XX_prune_unseen_entity_all_relation_con_loi_augmented.json`
- `prepare_summary.json`

其中：

- `entity_filter_then_augment` 是最终推荐顺序的输出。
- `augment_then_entity_filter_input` 是反向顺序的 OPS 输入。

### 2. 如果要公平比较反向顺序，再跑“先增补 -> 再实体 OPS”

```powershell
cd C:\Users\Zzl410410\Documents\Codex\2026-06-24\019ef53b-8c1b-7a92-ad59-c7c456388b6f

$env:OPS_FILTER_OUT_DIR = 'C:\Users\Zzl410410\Documents\Codex\2026-06-24\019ef53b-8c1b-7a92-ad59-c7c456388b6f\work\test_b_validation_scaling\outputs\order_swap_entityonly_filter\augment_then_entity_filter_ops'

python work\test_b_validation_scaling\run_holdout_ops_filter_sharded.py `
  --variants val_prune_unseen_augment_all_entities_partial_relations `
  --trials 0 1 4 6 7 `
  --workers 8 `
  --shard_size 5 `
  --shard_attempts 3 `
  --retry_sleep 15 `
  --timeout 180 `
  --retries 2 `
  --max_tokens 2048 `
  --batch_items 40 `
  --entity_drop_threshold 0.75 `
  --relation_drop_threshold 1.01
```

输出：

`work\test_b_validation_scaling\outputs\order_swap_entityonly_filter\augment_then_entity_filter_ops\ops_filter_sharded_summary.json`

## 已保存的最终报告

本次对话 outputs：

- `C:\Users\Zzl410410\Documents\Codex\2026-06-28\fnag\outputs\order_swap_entityonly_filter\ORDER_SWAP_ENTITYONLY_5FOLD.md`
- `C:\Users\Zzl410410\Documents\Codex\2026-06-28\fnag\outputs\order_swap_entityonly_filter\order_swap_entityonly_5fold_compare.json`
- `C:\Users\Zzl410410\Documents\Codex\2026-06-28\fnag\outputs\order_swap_entityonly_filter\ops_filter_sharded_summary.json`
- `C:\Users\Zzl410410\Documents\Codex\2026-06-28\fnag\outputs\order_swap_entityonly_filter\prepare_summary.json`

## 推荐下一步

如果要用于最终提交策略，优先使用：

`strict3 prune_unseen -> entity OPS only threshold 0.75 -> AI union-only entity all + relation CON/LOI augment`

原因：

- 五折全部胜出；
- overall total 最高；
- 主要收益来自实体 FP 明显降低，同时 TP 没有损失，反而略增；
- 关系不做 OPS 筛选更稳，之前实验表明关系 OPS 筛选会拉低 RE。
