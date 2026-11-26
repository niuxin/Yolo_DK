# 房屋检测、跟踪去重独立交付包 v1

这是图片帧输入的独立 HTTP 服务：YOLO 房屋检测 → 跟踪 → 多帧确认 → 视觉去重 → 代表帧选择 → 事件和图片导出。无需原 DemoHouseDetect 的其他目录，不包含拉流控制、网页、VLM 或外部业务回调。

在无人机主工程中使用房屋模型时，不需要启动本包的独立 `/v2` 服务；主工程的 `disease_service/config.yaml` 通过 `backend: house_tiled` 注册本包权重和切片推理参数，再由主工程的跟踪器、最佳帧存储和业务回调处理结果。该接入复用检测模型及推理逻辑，但不启用本包的 `UniqueTargetCore` 跨视角视觉去重。

## 1. 部署

推荐 Linux x86_64、Python 3.10、可用的 NVIDIA GPU。已验证环境为 Tesla V100 32GB / CUDA 12.1 PyTorch。安装脚本使用包内虚拟环境，不修改系统或原服务环境。首次安装需要访问 PyPI 与 PyTorch 下载源；本包包含模型，但不是离线依赖镜像，隔离网环境需要另行准备 wheelhouse。

解压后进入 `house_unique_service_v1`：

```bash
bash install.sh gpu
./start.sh --gpu 0 --port 18081
```

- `--gpu 0` 指同事机器的物理 GPU 0，不是固定原服务器卡号。可指定单个 GPU UUID。
- 本服务器测试必须用 `--gpu 2`，并用不同端口，例如 `--port 19081`；不要占用现有 18081。
- 启动后是前台服务，Ctrl+C 停止本服务。生产部署可由同事的进程管理器托管，保持单 worker。
- 默认监听 `0.0.0.0`；只需本机访问时加 `--host 127.0.0.1`。
- 默认模型随包提供；可通过 `--weights /absolute/path/model.pt` 指定，但本适配器只配置了 house 单类别，不能只换多类别模型就直接用于边坡。

CPU 验证（速度不作实时承诺）：

```bash
bash install.sh cpu
./start.sh --gpu cpu --port 18081
```

如已有兼容环境：

```bash
HOUSE_PYTHON=/你的环境/bin/python ./start.sh --gpu 0
```

检查服务：

```bash
curl http://127.0.0.1:18081/health
```

交互式 API 文档：`http://服务器IP:18081/docs`。当前是内网联调服务，不包含账号鉴权。

## 2. 一条命令验证接口

另开终端：

```bash
.venv/bin/python examples/client.py --base http://127.0.0.1:18081
```

示例创建新会话，把随包 sample.jpg 作为 8 个连续帧提交（间隔 200ms），检查相同帧重试一致性，关闭会话，下载代表图、带框图和标注到 `examples/results/<run_id>/`。

这是静态样图的接口验证，不是无人机运动、长期去重或真实精度验证。实际接入时把示例中的循环改成上游连续图片；不要每张图都创建新会话。

## 3. 对外调用契约

### 创建一次检测会话

`run_id` 必须是调用方生成的 32 位小写十六进制 UUID，`session_id` 是业务任务标识；当前 detector_id 仅支持 house。

```bash
RUN_ID=$(python3 -c 'import uuid; print(uuid.uuid4().hex)')
curl -X POST http://127.0.0.1:18081/v2/runs \
  -H 'Content-Type: application/json' \
  -d "{\"run_id\":\"$RUN_ID\",\"session_id\":\"flight-demo-001\",\"detector_id\":\"house\"}"
```

### 按顺序连续传图片

```bash
curl -X POST http://127.0.0.1:18081/v2/detect \
  -F "run_id=$RUN_ID" \
  -F "frame_id=0" \
  -F "timestamp_ms=0" \
  -F "image=@examples/sample.jpg"
```

下一帧修改 frame_id 和 timestamp_ms，并上传下一张图片。同一 run 要按帧号/时间有序发送。frame_id 可以是上游原视频帧号，抽帧后有间隔是正常的；timestamp_ms 使用源媒体时间，不要用算法完成时间替代。

输入：`run_id + frame_id + timestamp_ms + image`。JPEG/PNG，单图最多 20 MiB，最大 4000 万像素。

返回包含 `request_frame_id` 和 `events`。events 为空仅表示本次没有新确认事件，不代表没有候选房屋。达到确认条件后立即返回事件，不必等待视频结束。每条事件主要字段：

| 字段 | 含义 |
|---|---|
| event_id / target_id / run_id | 事件幂等键、业务目标 ID、会话 ID |
| frame_id / timestamp_ms | 选中代表帧的序号与源时间 |
| confirmed_frame_id / confirmed_timestamp_ms | 作出确认决定时的帧与时间 |
| class_id / class_name | 当前为 0 / house |
| bbox_xyxy | 代表原图的像素坐标 [左,上,右,下]，不是归一化坐标 |
| confidence | 代表帧对应检测置信度 |
| image_width / image_height | 代表原图尺寸 |
| assets | 原图、带框图、框文件及元数据的相对下载地址 |

框和图片属于代表帧，不能把当前请求图片替换为代表帧。将服务 base URL 与 assets 中的相对路径拼接即可下载；不要依赖 metadata 内的服务器本地绝对文件路径。

重试必须使用相同图片字节、帧号和时间戳；服务重放相同响应，可能再次返回相同事件。因此调用方仍应按 event_id 幂等消费。相同帧号但不同内容会返回 409。

### 获取已确认结果与关闭

```bash
curl "http://127.0.0.1:18081/v2/runs/$RUN_ID/events"
curl -X POST "http://127.0.0.1:18081/v2/runs/$RUN_ID/close"
```

close 返回完整事件集合并释放该会话的跟踪状态；不会强行把证据不足的候选目标变成确认事件。停止/异常时也应调用 close，目前没有空闲会话自动回收。关闭后继续推图需要新 run_id。

### 可选批量接口

`POST /v2/detect-batch`：multipart 包含 run_id、frames（JSON 列表，每项 frame_id/timestamp_ms）、同序 images 文件；每批 1–4 张，总计最多 64 MiB。可用 after_frame_id 指定前一批末帧，第一批用 -1。先用单帧串行接口跑通，再对批量进行调优；不要无界并发推进同一跟踪状态。

`/v2/detect-local` 为原项目同机共享缓冲优化接口，仍保留兼容，但不是同事跨机器上传图片的入口。独立调用默认使用 detect / detect-batch。

常见状态：409 会话已关闭/乱序/帧内容冲突；422 参数或 detector_id 错误；413 图片超限；415 图片无法解码；429 活跃会话/批次超限。

## 4. 包内结构与输出

```text
serve.py / start.sh / install.sh     独立启动与安装
requirements.txt                    主要依赖版本；torch 由安装脚本单独安装
trained_model/house_v3_recommended.pt 模型
code/disease_service/unique_app.py   HTTP、会话、持久化与幂等
unique_target_core/                  跟踪、去重、选帧、导出
test_house_detection/detector.py    单帧检测适配器
examples/client.py / sample.jpg     可运行接口示例
vendor/                             python-multipart 0.0.20 随包副本及元数据
verification/                       来源环境和本包验证记录
```

运行产物都在包内 `outputs/integration_detection/`：SQLite 会话/事件日志，以及 `<run_id>/targets/<target_id>/` 下的 original.jpg、annotated.jpg、bbox.txt、bbox_yolo.txt、metadata.json。无需原项目共享目录。

## 5. 适用范围与迁移

- 一个进程、一个模型，状态更新串行；最多两个活跃 run 是资源限制，不代表保证两路实时吞吐。不要直接增加 Uvicorn workers，多个进程不共享内核状态。
- 默认候选阈值 0.4；确认阈值 0.85，1.5 秒内 5 次高置信度，并有背景静态与选帧质量约束。不是所有目标都能在 1–2 秒内确认。
- 不承诺任意视角下绝对物理去重。跨 run / 服务重启不继承内核状态；事件文件可保留读取，但旧 run 不能继续跟踪。
- 当前房屋模型、静态假设、小框/边缘规则不能未经验证直接移植到所有边坡灾害。未来需要调整类别输出、按类别的框合并及确认规则。
- 本包不拉 RTSP/RTMP、不抽视频帧、不调用 VLM、不调用外部业务回调。由上游控制模块完成这些编排。
- 本包已在原服务器的已验证依赖环境下，用清空 PYTHONPATH、任意工作目录、独立端口进行实际测试；未在同事的新机器上执行全新联网安装。安装脚本和依赖已提供，驱动、网络下载源及新硬件兼容性仍需部署时检查。

## 6. 停止与校验

前台 Ctrl+C 停止；先关闭任务 run 再停止服务。不要用 pkill 等宽泛命令停止同事的其他服务。

包外附 SHA256 校验文件；包内 MANIFEST.json 记录模型和源文件哈希。部署前可运行 `python3 verify_package.py` 检查。
