# -*- coding: utf-8 -*-
"""按 detector_id 延迟加载模型，并串行执行同一模型。"""
from __future__ import annotations

from importlib import import_module
import threading
from typing import Mapping

import numpy as np

from .backends import HouseTiledDetectorBackend, UltralyticsDetectorBackend
from .config import ModelSettings
from .types import Detection, DetectorBackend


def create_detector_backend(settings: ModelSettings) -> DetectorBackend:
    """根据 ModelSettings 创建内置或自定义后端。"""
    if settings.backend == "ultralytics":
        return UltralyticsDetectorBackend(settings)
    if settings.backend == "house_tiled":
        return HouseTiledDetectorBackend(settings)
    if settings.backend_module and settings.backend_class:
        module = import_module(settings.backend_module)
        backend_type = getattr(module, settings.backend_class)
        backend = backend_type(settings)
        if not callable(getattr(backend, "detect", None)):
            raise TypeError(
                f"自定义后端缺少 detect 方法: "
                f"{settings.backend_module}.{settings.backend_class}"
            )
        return backend
    raise ValueError(f"不支持的检测后端: {settings.backend}")


class DetectorRegistry:
    """模型注册表；未知 ID 会拒绝，模型首次使用时加载。"""

    def __init__(
        self,
        models: Mapping[str, ModelSettings],
        backend_overrides: Mapping[str, DetectorBackend] | None = None,
    ) -> None:
        if not models:
            raise ValueError("至少需要注册一个检测模型")
        self._settings = dict(models)
        self._backends = dict(backend_overrides or {})
        unknown_overrides = self._backends.keys() - self._settings.keys()
        if unknown_overrides:
            raise ValueError(
                f"注入的检测后端未配置对应模型 ID: {sorted(unknown_overrides)}"
            )
        self._backend_locks = {
            model_id: threading.Lock() for model_id in self._settings
        }
        self._lock = threading.Lock()

    @property
    def model_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._settings))

    def detect(self, detector_id: str, image_rgb: np.ndarray) -> list[Detection]:
        if detector_id not in self._settings:
            raise ValueError(f"未配置检测算法: {detector_id}")
        with self._lock:
            backend = self._backends.get(detector_id)
            if backend is None:
                backend = create_detector_backend(self._settings[detector_id])
                self._backends[detector_id] = backend
        with self._backend_locks[detector_id]:
            return backend.detect(image_rgb)
