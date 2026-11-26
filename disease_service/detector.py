# -*- coding: utf-8 -*-
"""兼容旧导入：新代码请使用 disease_service.registry 和 types。"""

from .backends import HouseTiledDetectorBackend, UltralyticsDetectorBackend
from .registry import DetectorRegistry, create_detector_backend
from .types import Detection, DetectorBackend

__all__ = [
    "Detection",
    "DetectorBackend",
    "DetectorRegistry",
    "UltralyticsDetectorBackend",
    "HouseTiledDetectorBackend",
    "create_detector_backend",
]
