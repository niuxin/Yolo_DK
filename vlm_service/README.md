# VLM 结构化病害描述服务

VLM 服务接收图片、像素坐标框和类别，并返回描述、严重程度和建议。API 凭据只从
`VLM_API_KEY`（或 `--api-key-env` 指定的环境变量）读取。

```bash
VLM_PROVIDER_BASE_URL=https://provider.example/v1 \
VLM_MODEL=vision-model \
VLM_API_KEY=... \
python -m vlm_service.main --host 0.0.0.0 --port 8084
```

流媒体服务调用 `POST /v1/describe`，成功响应格式：

```json
{
  "disease_type": "crack",
  "summary": "构件表面可见线状裂缝。",
  "severity": "medium",
  "suggestion": "建议现场复核并持续观察。"
}
```

Provider 必须返回 JSON 对象，`severity` 只能是 `low`、`medium`、`high`；缺字段或非法值会作为错误返回，不会生成成功形状的默认结果。
