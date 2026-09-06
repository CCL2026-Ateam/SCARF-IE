# External Data, API, Rules, and Retrieval Resources

## Official data

本方案只使用官方发布数据：

- 官方 `train.zip`
- 官方 `test_B.zip`

入口脚本会从 `train.zip` 切分出：

- `pool.json`：前 800 条，作为 few-shot 检索池。
- `dev.json`：后 200 条。
- `original_1000_pool.json`：全部 1000 条官方标注记录，用于 original1000 strict3。

未使用人工标注的私有测试数据。

## Retrieval resources

Few-shot 示例只从官方标注数据中检索。默认检索方式为 lexical retriever，用于为每条 Test-B 文本选择 prompt examples。

不需要外部知识库、网页抓取、论文库、ontology dump 或私有 test label 资源。

## API usage

在线 pipeline 调用 OpenAI-compatible chat/completions API：

```env
BASE_URL=https://zjapi.com
MODEL_MAIN=gpt-5.4
MODEL_CHEAP=gpt-5.4-mini
MODEL_SECONDARY=gpt-5.5-openai-compact
API_KEY=<organizer_api_key>
TEMPERATURE=0.0
```

实际请求 URL：

```text
https://zjapi.com/v1/chat/completions
```

API key 通过环境变量传入，不应写入仓库文件。

## Final rules

最终后处理规则：

- 保留 relation-prune 输出中的实体集。
- 使用 ops2 输出中的关系作为基础。
- 删除所有 `USE` 关系。
- 从 relation-prune 输出补回/保留 `HAS`。
- 从 original1000 strict3 conservative 输出补入 `AFF/CON/LOI/OCI`，但仅当两端实体已存在。
- 拒绝任何端点缺失的关系。

该规则保证最终输出没有无效关系端点。
