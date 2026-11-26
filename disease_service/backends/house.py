# -*- coding: utf-8 -*-
"""房屋切片检测后端。"""
from __future__ import annotations

import numpy as np

from ..config import ModelSettings
from ..types import Detection


class HouseTiledDetectorBackend:
    """包装全图与重叠切片推理，并转换为主工程检测协议。"""

    def __init__(self, settings: ModelSettings) -> None:
        from .house_impl import HouseTiledDetector

        self._settings = settings
        self._detector = HouseTiledDetector(settings)

    def detect(self, image_rgb: np.ndarray) -> list[Detection]:
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError("image_rgb 必须是 HxWx3 的 RGB 图像")

        image_bgr = np.ascontiguousarray(image_rgb[:, :, ::-1])
        allowed = set(self._settings.class_names)
        detections: list[Detection] = []
        for item in self._detector.detect(image_bgr):
            disease_type = str(item["class_name"])
            if allowed and disease_type not in allowed:
                continue
            x1, y1, x2, y2 = item["bbox_xyxy"]
            detections.append(
                Detection(
                    disease_type=disease_type,
                    confidence=float(item["confidence"]),
                    bbox=(float(x1), float(y1), float(x2), float(y2)),
                    class_id=0,
                )
            )
        return detections
