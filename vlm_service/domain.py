# -*- coding: utf-8 -*-
"""VLM 病害描述适配层。

职责：按 bbox 裁剪 RGB 图像，并通过可替换的 Provider 生成可见病害描述。
``VLMService.describe`` 接收完整 RGB 图像、像素 bbox 和类别，返回描述文本；
OpenAICompatibleProvider 负责具体的 HTTP 请求。
"""
from __future__ import annotations

import base64
import io
import json
import os
from dataclasses import dataclass
from typing import Callable, Literal, Protocol

import numpy as np
import requests
from PIL import Image


@dataclass(frozen=True)
class BBox:
    """Image-pixel bounding box."""

    x1: float
    y1: float
    x2: float
    y2: float


@dataclass(frozen=True)
class VLMSettings:
    """Provider settings for the independent VLM process."""

    enabled: bool
    base_url: str
    model: str
    api_key_env: str
    timeout_s: float
    crop_margin_ratio: float
    jpeg_quality: int


@dataclass(frozen=True)
class VLMDescription:
    """VLM 生成的结构化病害描述。"""

    summary: str
    severity: Literal["low", "medium", "high"]
    suggestion: str


class VLMProvider(Protocol):
    """VLM 后端协议；返回病害描述、严重程度和处置建议。"""

    def describe_crop(
        self, image_rgb: np.ndarray, disease_type: str
    ) -> VLMDescription:
        """结构化描述病害裁剪图；网络或响应错误应抛出异常。"""

    def close(self) -> None:
        """释放 Provider 持有的网络资源。"""


VLMProviderFactory = Callable[[VLMSettings], VLMProvider]


class OpenAICompatibleProvider:
    """调用兼容 OpenAI Chat Completions 的视觉语言模型接口。"""

    def __init__(self, settings: VLMSettings) -> None:
        """创建 Provider；API Key 从配置指定的环境变量读取。"""
        api_key = os.environ.get(settings.api_key_env, "").strip()
        if not api_key:
            raise RuntimeError(
                f"VLM API 密钥环境变量未设置: {settings.api_key_env}"
            )
        self._settings = settings
        self._api_key = api_key
        self._session = requests.Session()

    def describe_crop(
        self, image_rgb: np.ndarray, disease_type: str
    ) -> VLMDescription:
        """请求 VLM 返回契约规定的结构化病害描述。

        Args:
            image_rgb: HxWx3 RGB uint8 裁剪图。
            disease_type: 检测模型给出的病害类别。

        Returns:
            描述、严重程度和建议。

        Raises:
            ValueError: 图像或模型响应格式不正确。
            requests.RequestException: 网络或 HTTP 请求失败。
        """
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError("VLM 输入必须是 HxWx3 RGB 图像")
        if image_rgb.size == 0:
            raise ValueError("VLM 输入裁剪图不能为空")
        buffer = io.BytesIO()
        Image.fromarray(image_rgb.astype(np.uint8, copy=False)).save(
            buffer, format="JPEG", quality=self._settings.jpeg_quality
        )
        encoded_image = base64.b64encode(buffer.getvalue()).decode("ascii")
        response = self._session.post(
            f"{self._settings.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={
                "model": self._settings.model,
                "temperature": 0.2,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "你是病害图像观察助手。只依据图像和检测类别判断，"
                            "不得编造病因。必须只输出一个 JSON 对象，字段为 "
                            "summary、severity、suggestion。summary 用简洁中文"
                            "描述可见事实；severity 只能是 low、medium、high，"
                            "证据不足时使用 low；suggestion 用简洁中文给出"
                            "保守的复核或处置建议。"
                        ),
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    f"检测类别：{disease_type}。请判断框内可见"
                                    "病害并按约定 JSON 格式返回。"
                                ),
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{encoded_image}"
                                },
                            },
                        ],
                    },
                ],
            },
            timeout=self._settings.timeout_s,
        )
        response.raise_for_status()
        payload = response.json()
        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError("VLM 响应缺少 choices[0].message.content") from exc
        if isinstance(content, list):
            text = "".join(
                item.get("text", "")
                for item in content
                if isinstance(item, dict) and item.get("type") == "text"
            ).strip()
        elif isinstance(content, str):
            text = content.strip()
        else:
            text = ""
        if not text:
            raise ValueError("VLM 返回了空结构化描述")
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError("VLM 必须返回 JSON 格式描述") from exc
        if not isinstance(data, dict):
            raise ValueError("VLM 描述必须是 JSON 对象")
        summary = data.get("summary")
        severity = data.get("severity")
        suggestion = data.get("suggestion")
        if not isinstance(summary, str) or not summary.strip():
            raise ValueError("VLM 描述缺少非空 summary")
        if not isinstance(severity, str) or severity not in {
            "low",
            "medium",
            "high",
        }:
            raise ValueError("VLM severity 必须是 low、medium 或 high")
        if not isinstance(suggestion, str) or not suggestion.strip():
            raise ValueError("VLM 描述缺少非空 suggestion")
        return VLMDescription(summary.strip(), severity, suggestion.strip())

    def close(self) -> None:
        """关闭 HTTP 连接池。"""
        self._session.close()


class VLMService:
    """负责 bbox 裁剪和后端调用的领域服务。"""

    def __init__(self, provider: VLMProvider, crop_margin_ratio: float) -> None:
        """创建 VLM 服务。

        Args:
            provider: 实现 ``VLMProvider`` 的模型适配器。
            crop_margin_ratio: bbox 周边上下文边距，取值范围为 0 到 1。
        """
        if not 0 <= crop_margin_ratio <= 1:
            raise ValueError("crop_margin_ratio 必须在 0 到 1 之间")
        self._provider = provider
        self._crop_margin_ratio = crop_margin_ratio

    def describe(
        self, image_rgb: np.ndarray, bbox: BBox, disease_type: str
    ) -> VLMDescription:
        """裁剪目标区域并生成病害描述。

        Args:
            image_rgb: 完整 HxWx3 RGB uint8 图像。
            bbox: 图像像素坐标，原点位于左上角。
            disease_type: 检测算法返回的病害类别。

        Returns:
            包含非空描述文本的 ``VLMDescription``。

        Raises:
            ValueError: 图像格式错误或 bbox 无有效面积。
            requests.RequestException: Provider 调用失败。
        """
        crop = crop_bbox(image_rgb, bbox, self._crop_margin_ratio)
        result = self._provider.describe_crop(crop, disease_type)
        if not isinstance(result, VLMDescription):
            raise ValueError("VLM Provider 必须返回 VLMDescription")
        if not isinstance(result.severity, str) or result.severity not in {
            "low",
            "medium",
            "high",
        }:
            raise ValueError("VLM severity 必须是 low、medium 或 high")
        if (
            not isinstance(result.summary, str)
            or not result.summary.strip()
            or not isinstance(result.suggestion, str)
            or not result.suggestion.strip()
        ):
            raise ValueError("VLM summary 和 suggestion 不能为空")
        return result

    def close(self) -> None:
        """释放底层 Provider 资源。"""
        self._provider.close()


def crop_bbox(
    image_rgb: np.ndarray, bbox: BBox, margin_ratio: float = 0.1
) -> np.ndarray:
    """将 bbox 扩展少量上下文后裁剪到图像范围。

    Args:
        image_rgb: 完整 HxWx3 RGB 图像。
        bbox: 像素坐标 bbox。
        margin_ratio: 按 bbox 宽高分别计算的扩展比例。

    Returns:
        拥有独立内存的 RGB 裁剪图。

    Raises:
        ValueError: 图像格式错误、边距非法或 bbox 无有效面积。
    """
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("image_rgb 必须是 HxWx3 RGB 图像")
    if not 0 <= margin_ratio <= 1:
        raise ValueError("margin_ratio 必须在 0 到 1 之间")
    height, width = image_rgb.shape[:2]
    x1, y1, x2, y2 = bbox.x1, bbox.y1, bbox.x2, bbox.y2
    if x2 <= x1 or y2 <= y1:
        raise ValueError("bbox 宽高必须大于零")
    margin_x = (x2 - x1) * margin_ratio
    margin_y = (y2 - y1) * margin_ratio
    left = max(0, int(np.floor(x1 - margin_x)))
    top = max(0, int(np.floor(y1 - margin_y)))
    right = min(width, int(np.ceil(x2 + margin_x)))
    bottom = min(height, int(np.ceil(y2 + margin_y)))
    if right <= left or bottom <= top:
        raise ValueError("bbox 与图像没有有效交集")
    return image_rgb[top:bottom, left:right].copy()


def create_vlm_service(
    settings: VLMSettings,
    provider_factory: VLMProviderFactory = OpenAICompatibleProvider,
) -> VLMService | None:
    """根据配置创建 VLM 服务；Provider 工厂可注入以替换模型供应商。

    VLM 关闭时返回 None；启用但缺少密钥或 Provider 构造失败时显式抛错。
    """
    if not settings.enabled:
        return None
    provider = provider_factory(settings)
    return VLMService(provider, settings.crop_margin_ratio)
