# 实时视频 AI 网关

`stream_service` 是业务系统调用的 AI 网关。它提供实时检测会话、业务结果回调，以及同一会话下的标注流发布。业务系统自己的 JWT 控制、查询和结果补数 API 不属于本服务。

## 检测会话

### 创建/幂等重试

`POST /api/v1/live-sessions` 接收业务生成的 `sessionId`。相同 `sessionId` 和相同参数重试会返回同一受理信息，不会重复启动；相同 ID 携带不同参数返回 HTTP 409。

```json
{
  "sessionId": "7c0a9b2e-7e9c-4ec3-90f9-a137d292849c",
  "jobId": "job-1001",
  "streamUrl": "rtmp://media.internal/live/drone-camera",
  "algorithmIds": ["crack", "spalling"],
  "callbackUrl": "https://business.example/manage/api/v1/ai-gateway/events"
}
```

成功响应为 HTTP 202：

```json
{
  "sessionId": "7c0a9b2e-7e9c-4ec3-90f9-a137d292849c",
  "gatewaySessionId": "gw-...",
  "status": "accepted",
  "acceptedAt": 1795161600000
}
```

`algorithmIds` 必须对应 `disease_service/config.yaml` 的模型 ID。模型由检测服务在首次调用时延迟加载。检测端按 `stream_service/config.yaml` 的 `stream.sample_fps` 抽帧；跨帧目标服务选出代表帧后，网关只对有命中的代表帧回调。

### 停止会话

`POST /api/v1/live-sessions/{sessionId}:stop`，请求体可省略：

```json
{"reason": "flight_landed"}
```

成功响应保持 `sessionId`，`status` 为 `stopping`。服务会先停止并断开会话关联的标注流，再请求检测 worker 停止。

## 业务结果回调

检测与描述共用创建请求中的 `callbackUrl`，路径必须为 `/manage/api/v1/ai-gateway/events`。每次回调携带 `X-App-Id`、Unix 秒 `X-Timestamp` 和 HMAC-SHA256 小写十六进制 `X-Signature`。重试复用完全相同的请求体和 `eventId`，仅更新时间戳并重新签名；遇到 429、5xx、连接失败时按指数退避，4xx 不重试。

密钥只从环境变量读取：

```bash
export LIVE_DETECTION_APP_ID="ai-gateway-prod"
export LIVE_DETECTION_APP_SECRET="<由部署环境安全注入>"
```

检测回调只包含已确认目标的最佳代表帧，图片以 `data:image/jpeg;base64,...` 填入 `frameObjectKey`，检测框坐标转换为 `[x, y, width, height]`。检测回调成功确认后才调用结构化 VLM 并发送 `description_result`；VLM 或描述回调失败不会撤销已确认的检测结果。

回调目标必须命中 `gateway.callback_allowed_hosts`，或解析到私有地址。HTTP 仅在 `gateway.allow_insecure_http: true` 且目标为私有地址时允许；正式环境应使用 HTTPS 和主机白名单。

## 标注流输出

输出任务需挂在已创建的会话下。一个任务可同时处理多个算法，所有检测框绘制在同一路输出流上。

`POST /api/v1/live-sessions/{sessionId}/output`：

```json
{
  "streamUrl": "rtmp://media.internal/live/drone-camera",
  "algorithmIds": ["crack", "spalling"],
  "outputStreamUrl": "rtmp://172.16.135.48:1935/ai/camera-stream"
}
```

`outputStreamUrl` 必须是 app 为 `ai`、不含 query string/secret 的 RTMP 地址。固定推流密钥通过环境变量注入：

```bash
export LIVE_OUTPUT_RTMP_SECRET="<由媒体服务器管理员线下提供>"
```

服务在实际拉到首帧后才启动 publisher，并在运行时追加 `?secret=...`；密钥不会出现在 API 响应或应用日志中。成功受理返回 `sessionId`、`outputTaskId` 和 `starting` 状态；运行状态为 `pushing`、`stopped` 或 `failed`。重复 POST 不会启动第二个 publisher；失败任务可通过相同 POST 重试。

```text
GET    /api/v1/live-sessions/{sessionId}/output
DELETE /api/v1/live-sessions/{sessionId}/output
```

标注流保留 ffprobe 探测到的源分辨率和帧率，使用 H.264/RTMP；GOP 设置为约 2 秒。输入读取使用有界队列和背压，不主动抽帧或丢弃应用队列中的帧。算法推理达不到输入帧率时，端到端延迟无法保证在 3 秒内，部署时须验证模型吞吐、网络带宽和编码负载。

媒体输出主机须填写 `gateway.output_allowed_hosts`，或解析到私有地址。RTMP 发布密钥不得写入配置文件、请求地址、日志或代码仓库。

## 启动配置

```bash
python -m stream_service.consumer --config stream_service/config.yaml
python -m disease_service.main --config disease_service/config.yaml
python -m unique_target_service.main --host 0.0.0.0 --port 8083
VLM_PROVIDER_BASE_URL="https://vlm-provider.example/v1" VLM_MODEL="vision-model" \
VLM_API_KEY="<由部署环境安全注入>" \
python -m vlm_service.main --host 0.0.0.0 --port 8084
```

推流需同时安装 `ffmpeg` 和 `ffprobe`。`stream_service/config.yaml` 中可配置 `stream.ffmpeg_path`、`stream.ffprobe_path`、检测/VLM/目标服务地址、并发上限、回调重试和回调/媒体主机白名单。

## VLM 响应

`POST /v1/describe` 现在返回 `summary`、`severity`、`suggestion`；`severity` 固定为 `low`、`medium`、`high` 之一。VLM Provider 必须输出符合结构的 JSON，无法解析或字段无效时请求会显式失败。

## 已移除的旧控制 API

旧版 `/v1/tasks`、`/upload`、本地 SQLite 入库回调不再作为控制器接口提供；调用方应迁移到上述网关协议。

## 本地可视化界面

独立的 `web_ui` 仅保留历史数据库只读查询和单独的视频预览，不再启停检测会话，也不直接接收 AI 网关回调；需要展示新回调结果时，应由业务系统按其数据协议处理。
# Yolo_DK
