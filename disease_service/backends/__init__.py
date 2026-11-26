"""内置检测后端。"""

from .house import HouseTiledDetectorBackend
from .house_impl import HouseTiledDetector
from .ultralytics import UltralyticsDetectorBackend

__all__ = [
    "HouseTiledDetector",
    "HouseTiledDetectorBackend",
    "UltralyticsDetectorBackend",
]
