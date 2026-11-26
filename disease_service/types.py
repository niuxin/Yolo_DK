# -*- coding: utf-8 -*-
"""检测服务的公共数据类型和后端协议。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class Detection:
    """一帧图像中的单个原始检测框。"""

    disease_type: str
    confidence: float
    bbox: tuple[float, float, float, float]
    class_id: int = 0


class DetectorBackend(Protocol):
    """模型后端协议；所有实现都接收 HxWx3 RGB 图像。"""

    def detect(self, image_rgb: np.ndarray) -> list[Detection]:
        """返回原始像素坐标检测框。"""
