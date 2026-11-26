# -*- coding: utf-8 -*-
"""通用 Ultralytics YOLO 后端。"""
from __future__ import annotations

import numpy as np

from ..config import ModelSettings
from ..types import Detection


class UltralyticsDetectorBackend:
    """使用 Ultralytics YOLO 加载和推理已注册权重。"""

    def __init__(self, settings: ModelSettings) -> None:
        if not settings.weights.is_file():
            raise FileNotFoundError(f"模型权重不存在: {settings.weights}")
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "Ultralytics 后端需要安装 ultralytics 及匹配的 torch 环境"
            ) from exc
        self._settings = settings
        self._model = YOLO(str(settings.weights))

    def detect(self, image_rgb: np.ndarray) -> list[Detection]:
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError("image_rgb 必须是 HxWx3 的 RGB 图像")

        results = self._model.predict(
            source=image_rgb,
            device=self._settings.device,
            imgsz=self._settings.image_size,
            conf=self._settings.confidence_threshold,
            iou=self._settings.iou_threshold,
            verbose=False,
        )
        if not results:
            return []

        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []

        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        names = result.names
        allowed = set(self._settings.class_names)
        detections: list[Detection] = []
        for box, confidence, class_id in zip(boxes, confidences, classes):
            disease_type = str(names.get(int(class_id), class_id))
            if allowed and disease_type not in allowed:
                continue
            detections.append(
                Detection(
                    disease_type=disease_type,
                    confidence=float(confidence),
                    bbox=tuple(float(value) for value in box[:4]),
                    class_id=int(class_id),
                )
            )
        return detections
