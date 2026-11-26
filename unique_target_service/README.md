# 唯一目标与代表帧服务

本服务接收单帧图像及病害检测服务输出的原始框，按会话顺序运行
`UniqueTargetCore`，并在目标确认时将原图、标框图及元数据写入本服务的数据目录。
它不加载检测模型，不连接数据库，也不调用 VLM。

```bash
python -m unique_target_service.main --host 0.0.0.0 --port 8083
```

`POST /v1/process` 使用 multipart 字段 `session_id`、`frame_id`、
`timestamp_ms`、JSON 数组 `detections` 和 `image`。同一会话须顺序提交且帧号、
时间戳严格递增。服务返回新确认事件；事件中的 `original_image_url` 可用于取得
所选代表帧。`POST /v1/close` 释放会话并输出待人工复核清单，不会把未确认目标
强行转换成事件。

默认输出目录是本服务目录下的 `data/`，可用 `--output-dir` 指定。不同服务实例
应使用各自的数据目录。

核心代码基于房屋场景版本迁入；将其暂时视作通用算法不等于已验证适用于所有
病害类别。生产启用前仍应使用目标类别的连续视频验证召回、误合并和代表帧质量。
