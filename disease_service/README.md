# disease_service 检测目录说明

本目录只负责单帧目标检测及 HTTP API，不维护跨帧跟踪、不做视角去重或代表帧选择，也不负责拉流、业务入库、VLM 或外部回调。

## 文件职责

```text
disease_service/
├── __init__.py        公共导出
├── README.md          本说明
├── main.py            命令行入口，加载 YAML 后启动 Uvicorn
├── app.py             FastAPI、图像解码和性能指标
├── config.py          YAML 解析与配置校验
├── config.yaml        默认部署配置
├── types.py           Detection 和 DetectorBackend 公共协议
├── registry.py        detector_id 注册、延迟加载和同模型串行推理
├── house_detector.py  旧导入兼容入口；新代码不要继续依赖
└── backends/
    ├── __init__.py
    ├── ultralytics.py  通用 Ultralytics YOLO 后端
    ├── house.py        房屋模型协议适配
    └── house_impl.py   房屋全图和切片推理实现
```

## 请求链路

```text
multipart 图像
  -> app.py 解码为 RGB
  -> registry.py 按 detector_id 选择模型
  -> 内置或自定义 DetectorBackend 推理
  -> 返回该帧的原始检测框
```

`session_id` 和 `frame_id` 仅用于请求及日志关联，检测服务不保存跨请求状态。

## 配置模型

通用 Ultralytics 模型只需在 `models` 中注册：

```yaml
default_model_id: default

model:
  weights: "/models/disease.pt"
  device: "0"
  image_size: 1280
  confidence_threshold: 0.25
  iou_threshold: 0.45
  class_names: []

models:
  default: {}
  fruit:
    weights: "/models/fruit.pt"
    device: "0"
    image_size: 1280
    confidence_threshold: 0.30
    iou_threshold: 0.45
    class_names: ["fruit_disease"]
```

未显式配置 `backend` 时使用 `ultralytics`。每个模型配置会继承顶层 `model` 的值，模型条目只覆盖需要变化的字段。相对权重路径按本 YAML 所在目录解析。

房屋切片模型使用内置 `house_tiled`：

```yaml
models:
  house:
    backend: house_tiled
    weights: "/models/house_v3_recommended.pt"
    device: "0"
    image_size: 640
    confidence_threshold: 0.4
    iou_threshold: 0.45
    class_names: ["house"]
    tiled: true
    tile_size: 1280
    max_det: 300
    nms_iou_threshold: 0.45
```

## 后续注册一个全新模型

新模型分为两类：

1. **兼容 Ultralytics 权重**：无需新增 Python 文件，只需在 `models` 增加新 `detector_id`。
2. **有特殊预处理、切片、推理或输出格式**：实现一个后端类，并在 YAML 注册。

自定义后端必须接收 `ModelSettings`，并实现：

```python
class MyDetectorBackend:
    def __init__(self, settings: ModelSettings) -> None:
        ...

    def detect(self, image_rgb: np.ndarray) -> list[Detection]:
        ...
```

建议把新实现放在 `disease_service/backends/` 下，例如：

```text
disease_service/backends/
└── my_detector.py
```

然后在 YAML 中同时指定模块名和类名：

```yaml
models:
  my_model:
    backend: custom
    backend_module: "disease_service.backends.my_detector"
    backend_class: "MyDetectorBackend"
    weights: "/models/my-model.pt"
    device: "0"
    image_size: 1280
    confidence_threshold: 0.25
    iou_threshold: 0.45
    class_names: []
```

`backend_module` 必须是项目可以导入的模块；部署包中不要依赖只存在于同事交付目录中的模块或相对导入。

自定义后端的 `detect` 返回主工程统一类型：

```python
Detection(
    disease_type="crack",
    confidence=0.92,
    bbox=(x1, y1, x2, y2),
)
```

坐标必须是输入图像的像素坐标，原点在图像左上角。返回列表是该帧的原始检测结果；跨帧稳定性和视角去重由独立目标服务处理。

## 启动

```bash
python -m disease_service.main \
  --config disease_service/config.yaml
```

未提供 `detector_id` 时使用 `default_model_id`。调用示例：

```bash
curl -X POST "http://127.0.0.1:8081/v1/detect" \
  -F "session_id=task-001" \
  -F "frame_id=1" \
  -F "detector_id=house" \
  -F "image=@frame.jpg"
```

## 健康检查

```bash
curl http://127.0.0.1:8081/health
```

健康响应包含默认模型文件名和已注册 `detector_id`。

## 扩展和兼容约定

- 新增模型优先只改 YAML 和后端实现。
- 内置后端位于 `backends/`，注册逻辑位于 `registry.py`。
- `detector.py` 是旧导入兼容入口，不再承载主实现。
- `house_detector.py` 是旧房屋导入兼容入口，新代码应直接使用 `backends.house_impl` 或通过注册表加载。
- 模型后端只处理当前帧；跨帧跟踪、视角去重和代表帧选择由独立目标服务处理。
- 模型实例按首次请求延迟加载，同一模型不会由并发请求同时推理。
- 权重路径、设备参数和类别白名单应只通过配置维护，不在后端代码中硬编码。
