# -*- coding: utf-8 -*-
"""读取并校验病害检测服务配置。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ApiSettings:
    """HTTP 服务参数。"""

    host: str
    port: int
    max_upload_mb: int


@dataclass(frozen=True)
class ModelSettings:
    """模型推理参数。"""

    weights: Path
    device: str
    image_size: int
    confidence_threshold: float
    iou_threshold: float
    class_names: tuple[str, ...]
    backend: str = "ultralytics"
    tiled: bool = False
    tile_size: int = 1280
    max_det: int = 300
    nms_iou_threshold: float = 0.45
    backend_module: str | None = None
    backend_class: str | None = None


@dataclass(frozen=True)
class DiseaseServiceSettings:
    """病害检测服务完整配置。"""

    api: ApiSettings
    model: ModelSettings
    models: dict[str, ModelSettings]
    default_model_id: str


def load_settings(config_path: str | Path) -> DiseaseServiceSettings:
    """从 YAML 文件加载服务配置并完成类型和范围校验。

    Args:
        config_path: YAML 配置文件路径，相对路径按当前工作目录解析。

    Returns:
        校验后的完整服务配置，模型权重相对配置文件所在目录解析。

    Example:
        >>> settings = load_settings("code/disease_service/config.yaml")
        >>> settings.model.weights.name
        'disease.pt'
    """
    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"病害服务配置不存在: {path}")
    with path.open("r", encoding="utf-8-sig") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError("病害服务配置根节点必须是 YAML 对象")

    api_data = _mapping(data.get("api"), "api")
    model_data = _mapping(data.get("model"), "model")
    api = ApiSettings(
        host=str(api_data.get("host", "0.0.0.0")),
        port=_positive_int(api_data.get("port", 8081), "api.port"),
        max_upload_mb=_positive_int(
            api_data.get("max_upload_mb", 10), "api.max_upload_mb"
        ),
    )
    model = _parse_model(model_data, path.parent, "model")
    models_data = data.get("models")
    if models_data is None:
        models = {"default": model}
    else:
        model_entries = _mapping(models_data, "models")
        if not model_entries:
            raise ValueError("models 至少需要配置一个算法")
        models = {}
        for model_id, model_overrides in model_entries.items():
            normalized_id = str(model_id).strip()
            if not normalized_id:
                raise ValueError("models 中的算法 ID 不能为空")
            overrides = _mapping(model_overrides, f"models.{normalized_id}")
            merged = {**model_data, **overrides}
            models[normalized_id] = _parse_model(
                merged, path.parent, f"models.{normalized_id}"
            )
    default_model_id = str(data.get("default_model_id", "default")).strip()
    if default_model_id not in models:
        raise ValueError("default_model_id 必须对应 models 中已配置的算法 ID")
    model = models[default_model_id]
    return DiseaseServiceSettings(
        api=api,
        model=model,
        models=models,
        default_model_id=default_model_id,
    )


def _parse_model(
    model_data: dict[str, Any], config_directory: Path, name: str
) -> ModelSettings:
    """解析单个算法配置；相对权重路径按配置文件目录解析。"""
    weights = Path(str(model_data.get("weights", "models/disease.pt"))).expanduser()
    if not weights.is_absolute():
        weights = config_directory / weights
    backend = str(model_data.get("backend", "ultralytics")).strip()
    backend_module = model_data.get("backend_module")
    backend_class = model_data.get("backend_class")
    custom_backend = backend_module is not None or backend_class is not None
    if not custom_backend and backend not in {"ultralytics", "house_tiled"}:
        raise ValueError(
            f"{name}.backend 必须是 ultralytics 或 house_tiled"
        )
    if custom_backend and (
        not isinstance(backend_module, str)
        or not backend_module.strip()
        or not isinstance(backend_class, str)
        or not backend_class.strip()
    ):
        raise ValueError(
            f"{name}.backend_module 和 {name}.backend_class 必须同时配置"
        )
    return ModelSettings(
        weights=weights.resolve(),
        device=str(model_data.get("device", "0")),
        image_size=_positive_int(
            model_data.get("image_size", 1280), f"{name}.image_size"
        ),
        confidence_threshold=_ratio(
            model_data.get("confidence_threshold", 0.25),
            f"{name}.confidence_threshold",
        ),
        iou_threshold=_ratio(
            model_data.get("iou_threshold", 0.45), f"{name}.iou_threshold"
        ),
        class_names=_string_tuple(model_data.get("class_names", [])),
        backend=backend,
        tiled=_boolean(model_data.get("tiled", False), f"{name}.tiled"),
        tile_size=_positive_int(
            model_data.get("tile_size", 1280), f"{name}.tile_size"
        ),
        max_det=_positive_int(model_data.get("max_det", 300), f"{name}.max_det"),
        nms_iou_threshold=_ratio(
            model_data.get("nms_iou_threshold", 0.45),
            f"{name}.nms_iou_threshold",
        ),
        backend_module=(
            str(backend_module).strip() if custom_backend else None
        ),
        backend_class=(
            str(backend_class).strip() if custom_backend else None
        ),
    )


def _mapping(value: Any, name: str) -> dict[str, Any]:
    """将 YAML 节点转换为对象并给出带字段名的错误。

    Args:
        value: 待检查的 YAML 节点。
        name: 配置字段名，用于错误提示。

    Returns:
        原字典节点；空节点转换为空字典。

    Example:
        >>> _mapping(None, "api")
        {}
    """
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{name} 必须是 YAML 对象")
    return value


def _positive_int(value: Any, name: str) -> int:
    """将配置值转换为正整数。

    Args:
        value: 原始配置值。
        name: 配置字段名。

    Returns:
        大于零的整数。

    Example:
        >>> _positive_int("3", "min_hits")
        3
    """
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是正整数") from exc
    if number <= 0:
        raise ValueError(f"{name} 必须是正整数")
    return number


def _ratio(value: Any, name: str) -> float:
    """将配置值转换为 0 到 1 之间的比例。

    Args:
        value: 原始配置值。
        name: 配置字段名。

    Returns:
        闭区间 [0, 1] 内的浮点数。

    Example:
        >>> _ratio(0.25, "confidence_threshold")
        0.25
    """
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是 0 到 1 之间的数") from exc
    if not 0 <= number <= 1:
        raise ValueError(f"{name} 必须是 0 到 1 之间的数")
    return number


def _boolean(value: Any, name: str) -> bool:
    """将 YAML 布尔值或常见字符串转换为布尔值。

    Args:
        value: 原始配置值。
        name: 配置字段名。

    Returns:
        解析后的布尔值。

    Example:
        >>> _boolean("true", "require_interior_box")
        True
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        return value.lower() == "true"
    raise ValueError(f"{name} 必须是 true 或 false")


def _string_tuple(value: Any) -> tuple[str, ...]:
    """将可选的类别列表转换为去空白的字符串元组。

    Args:
        value: YAML 中的类别列表。

    Returns:
        类别名称元组；空元组表示接受模型全部类别。

    Example:
        >>> _string_tuple(["crack", " spall "])
        ('crack', 'spall')
    """
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError("model.class_names 必须是列表")
    return tuple(str(item).strip() for item in value if str(item).strip())