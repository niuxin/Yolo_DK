# -*- coding: utf-8 -*-
"""房屋模型的全图与重叠切片推理实现。"""
from __future__ import annotations

import numpy as np

from ..config import ModelSettings


class HouseTiledDetector:
    """保留房屋模型原有切片、边缘过滤和全局 NMS 参数。"""

    def __init__(self, settings: ModelSettings) -> None:
        if not settings.weights.is_file():
            raise FileNotFoundError(f"房屋模型权重不存在: {settings.weights}")

        try:
            import torch
            from torchvision.ops import nms
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "house_tiled 后端需要 ultralytics、torch 和 torchvision"
            ) from exc

        self._settings = settings
        self._model = YOLO(str(settings.weights))
        self._torch = torch
        self._nms = nms

    def detect(
        self,
        image_bgr: np.ndarray,
        frame_id: int | None = None,
    ) -> list[dict[str, object]]:
        if image_bgr is None or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError("Expected HxWx3 BGR image")

        height, width = image_bgr.shape[:2]
        regions: list[tuple[int, int, int, int, bool]] = [
            (0, 0, width, height, True)
        ]
        if self._settings.tiled and max(width, height) > 1600:
            tile_size = min(self._settings.tile_size, width, height)
            step = max(1, int(tile_size * 0.8))
            xs = sorted(set(list(range(0, width - tile_size + 1, step)) + [width - tile_size]))
            ys = sorted(set(list(range(0, height - tile_size + 1, step)) + [height - tile_size]))
            regions.extend(
                (x, y, x + tile_size, y + tile_size, False)
                for y in ys
                for x in xs
            )

        all_boxes: list[list[float]] = []
        scores: list[float] = []
        for offset in range(0, len(regions), 4):
            batch = regions[offset : offset + 4]
            results = self._model.predict(
                [image_bgr[y:y2, x:x2] for x, y, x2, y2, _ in batch],
                imgsz=self._settings.image_size,
                conf=self._settings.confidence_threshold,
                iou=self._settings.iou_threshold,
                max_det=self._settings.max_det,
                device=self._settings.device,
                quantize=32 if self._settings.device == "cpu" else 16,
                verbose=False,
            )
            for region, result in zip(batch, results):
                x, y, x2, y2, full = region
                if result.boxes is None:
                    continue
                for row in result.boxes.data.cpu().numpy():
                    a, b, c, d, score, _ = row
                    if not full and (
                        (a < 4 and x > 0)
                        or (b < 4 and y > 0)
                        or (c > x2 - x - 4 and x2 < width)
                        or (d > y2 - y - 4 and y2 < height)
                    ):
                        continue
                    box = [
                        max(0.0, float(a + x)),
                        max(0.0, float(b + y)),
                        min(float(width), float(c + x)),
                        min(float(height), float(d + y)),
                    ]
                    if box[2] > box[0] and box[3] > box[1]:
                        all_boxes.append(box)
                        scores.append(float(score))

        if not all_boxes:
            return []

        keep = self._nms(
            self._torch.tensor(all_boxes),
            self._torch.tensor(scores),
            self._settings.nms_iou_threshold,
        ).tolist()[: self._settings.max_det]
        return [
            {
                "frame_id": frame_id,
                "class_id": 0,
                "class_name": "house",
                "bbox_xyxy": all_boxes[index],
                "confidence": scores[index],
            }
            for index in keep
        ]
