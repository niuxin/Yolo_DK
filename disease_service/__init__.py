"""无状态病害检测服务。"""

from .registry import DetectorRegistry
from .types import Detection, DetectorBackend

__all__ = ["Detection", "DetectorBackend", "DetectorRegistry"]