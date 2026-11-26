# -*- coding: utf-8 -*-
"""病害检测 API 客户端。

职责：将 BGR 帧编码为 JPEG、调用远程检测 API，并把 JSON 结果校验为
强类型检测对象。输入是图像、会话/帧 ID 和 detector_id；返回检测列表。
"""
from __future__ import annotations

import io
import json
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import requests
from PIL import Image

from .config import DiseaseApiSettings, UniqueTargetApiSettings, VLMSettings


@dataclass(frozen=True)
class BBox:
    """图像像素坐标 bbox。"""

    x1: float
    y1: float
    x2: float
    y2: float

    def as_dict(self) -> dict[str, float]:
        """转换为便于数据库和前端使用的字典。

        Returns:
            四个坐标组成的字典。

        Example:
            >>> BBox(1, 2, 30, 40).as_dict()
            {'x1': 1, 'y1': 2, 'x2': 30, 'y2': 40}
        """
        return {
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
        }


@dataclass(frozen=True)
class DiseaseDetection:
    """单帧病害模型的原始检测结果。"""

    disease_type: str
    confidence: float
    bbox: BBox
    class_id: int = 0


@dataclass(frozen=True)
class DiseaseDescription:
    """VLM 生成的业务回调字段。"""

    summary: str
    severity: Literal["low", "medium", "high"]
    suggestion: str


class DiseaseApiClient:
    """封装 /v1/detect 的 HTTP 调用。"""

    def __init__(self, settings: DiseaseApiSettings) -> None:
        """创建 API 客户端。

        Args:
            settings: 服务基础地址、超时和会话前缀。

        Example:
            >>> client = DiseaseApiClient(settings.disease_api)
        """
        self._settings = settings
        self._session = requests.Session()

    def detect(
        self,
        image_bgr: np.ndarray,
        session_id: str,
        frame_id: int,
        detector_id: str = "default",
    ) -> list[DiseaseDetection]:
        """提交 JPEG 图像并返回该帧的原始检测框。

        Args:
            image_bgr: HxWx3 BGR uint8 图像。
            session_id: 同一路视频优先复用的检测会话 ID。
            frame_id: 当前视频帧序号。
            detector_id: 病害服务端预先配置的模型标识。

        Returns:
            当前帧模型检测列表。

        Raises:
            requests.RequestException: 连接或 HTTP 错误。
            ValueError: 返回 JSON 不符合约定。

        Example:
            >>> client.detect(frame, "drone-01", 120)
            []
        """
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError("image_bgr 必须是 HxWx3 的 BGR 图像")
        if not detector_id.strip():
            raise ValueError("detector_id 不能为空")
        buffer = io.BytesIO()
        Image.fromarray(image_bgr[:, :, ::-1]).save(
            buffer, format="JPEG", quality=self._settings.upload_jpeg_quality
        )
        response = self._session.post(
            f"{self._settings.base_url}/v1/detect",
            data={
                "session_id": session_id,
                "frame_id": str(frame_id),
                "detector_id": detector_id,
            },
            files={"image": ("frame.jpg", buffer.getvalue(), "image/jpeg")},
            timeout=self._settings.timeout_s,
        )
        response.raise_for_status()
        payload = response.json()
        detections = payload.get("detections")
        if not isinstance(detections, list):
            raise ValueError("病害服务响应缺少 detections 列表")
        return [_parse_detection(item) for item in detections]

    def close(self) -> None:
        """释放底层 HTTP 连接池。

        Example:
            >>> client.close()
        """
        self._session.close()


def _parse_detection(payload: Any) -> DiseaseDetection:
    """将 API JSON 对象转换为强类型检测结果。

    Args:
        payload: /v1/detect 返回的单条检测。

    Returns:
        校验后的病害检测。

    Raises:
        ValueError: 缺少字段或坐标非法。

    Example:
        >>> _parse_detection({"disease_type": "crack",
        ...                   "confidence": 0.9, "class_id": 0,
        ...                   "bbox": {"x1": 1, "y1": 2, "x2": 3, "y2": 4}})
    """
    if not isinstance(payload, dict):
        raise ValueError("病害检测结果必须是 JSON 对象")
    bbox = payload.get("bbox")
    if not isinstance(bbox, dict):
        raise ValueError("病害检测结果缺少 bbox")
    parsed = DiseaseDetection(
        disease_type=str(payload["disease_type"]),
        confidence=float(payload["confidence"]),
        bbox=BBox(
            x1=float(bbox["x1"]),
            y1=float(bbox["y1"]),
            x2=float(bbox["x2"]),
            y2=float(bbox["y2"]),
        ),
        class_id=int(payload.get("class_id", 0)),
    )
    if parsed.bbox.x2 <= parsed.bbox.x1 or parsed.bbox.y2 <= parsed.bbox.y1:
        raise ValueError("病害 bbox 宽高必须大于零")
    return parsed


class UniqueTargetApiClient:
    """串行提交图像帧及其检测框，下载新确认事件的代表图。"""

    def __init__(self, settings: UniqueTargetApiSettings) -> None:
        self._settings = settings
        self._session = requests.Session()

    def process(
        self,
        image_bgr: np.ndarray,
        session_id: str,
        frame_id: int,
        timestamp_ms: float,
        detections: list[DiseaseDetection],
    ) -> list[dict[str, Any]]:
        buffer = io.BytesIO()
        Image.fromarray(image_bgr[:, :, ::-1]).save(buffer, format="JPEG", quality=90)
        payload = [
            {
                "class_id": item.class_id,
                "class_name": item.disease_type,
                "confidence": item.confidence,
                "bbox_xyxy": [
                    item.bbox.x1,
                    item.bbox.y1,
                    item.bbox.x2,
                    item.bbox.y2,
                ],
            }
            for item in detections
        ]
        response = self._session.post(
            f"{self._settings.base_url}/v1/process",
            data={
                "session_id": session_id,
                "frame_id": str(frame_id),
                "timestamp_ms": str(timestamp_ms),
                "detections": json.dumps(payload, ensure_ascii=False),
            },
            files={"image": ("frame.jpg", buffer.getvalue(), "image/jpeg")},
            timeout=self._settings.timeout_s,
        )
        response.raise_for_status()
        events = response.json().get("events")
        if not isinstance(events, list):
            raise ValueError("唯一目标服务响应缺少 events 列表")
        for event in events:
            if not isinstance(event, dict) or not isinstance(
                event.get("original_image_url"), str
            ):
                raise ValueError("唯一目标事件缺少代表帧图片地址")
            image_response = self._session.get(
                f"{self._settings.base_url}{event['original_image_url']}",
                timeout=self._settings.timeout_s,
            )
            image_response.raise_for_status()
            event["image_bytes"] = image_response.content
        return events

    def close_session(self, session_id: str) -> None:
        response = self._session.post(
            f"{self._settings.base_url}/v1/close",
            data={"session_id": session_id},
            timeout=self._settings.timeout_s,
        )
        response.raise_for_status()

    def close(self) -> None:
        self._session.close()



class VLMApiClient:
    """调用独立 VLM 服务描述一个检测框。"""

    def __init__(self, settings: VLMSettings) -> None:
        self._settings = settings
        self._session = requests.Session()

    def describe(
        self,
        image_bgr: np.ndarray,
        detection: DiseaseDetection,
    ) -> DiseaseDescription:
        buffer = io.BytesIO()
        Image.fromarray(image_bgr[:, :, ::-1]).save(buffer, format="JPEG", quality=90)
        response = self._session.post(
            f"{self._settings.base_url}/v1/describe",
            data={
                "disease_type": detection.disease_type,
                "x1": str(detection.bbox.x1),
                "y1": str(detection.bbox.y1),
                "x2": str(detection.bbox.x2),
                "y2": str(detection.bbox.y2),
            },
            files={"image": ("frame.jpg", buffer.getvalue(), "image/jpeg")},
            timeout=self._settings.timeout_s,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("VLM 服务响应必须是 JSON 对象")
        summary = payload.get("summary")
        severity = payload.get("severity")
        suggestion = payload.get("suggestion")
        if not isinstance(summary, str) or not summary.strip():
            raise ValueError("VLM 服务响应缺少非空 summary")
        if not isinstance(severity, str) or severity not in {
            "low",
            "medium",
            "high",
        }:
            raise ValueError("VLM 服务响应 severity 无效")
        if not isinstance(suggestion, str) or not suggestion.strip():
            raise ValueError("VLM 服务响应缺少非空 suggestion")
        return DiseaseDescription(summary.strip(), severity, suggestion.strip())

    def close(self) -> None:
        self._session.close()