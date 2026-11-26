import asyncio
import io
import unittest

import numpy as np
from fastapi import UploadFile
from fastapi.routing import APIRoute
from PIL import Image

from vlm_service.app import create_app
from vlm_service.domain import (
    BBox,
    VLMDescription,
    VLMService,
    VLMSettings,
    crop_bbox,
)


class _FakeProvider:
    def __init__(self) -> None:
        self.image: np.ndarray | None = None
        self.disease_type: str | None = None

    def describe_crop(
        self, image_rgb: np.ndarray, disease_type: str
    ) -> VLMDescription:
        self.image = image_rgb
        self.disease_type = disease_type
        return VLMDescription(
            "可见叶片局部褐色斑点",
            "medium",
            "建议现场复核并持续观察。",
        )

    def close(self) -> None:
        pass


class VLMTests(unittest.TestCase):
    def test_crop_expands_bbox_and_clamps_to_image(self) -> None:
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        crop = crop_bbox(image, BBox(0, 0, 4, 4), margin_ratio=0.25)
        self.assertEqual(crop.shape, (5, 5, 3))
        self.assertTrue(crop.flags.owndata)

    def test_crop_rejects_bbox_without_image_intersection(self) -> None:
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "没有有效交集"):
            crop_bbox(image, BBox(20, 20, 30, 30))

    def test_service_passes_crop_and_disease_to_provider(self) -> None:
        provider = _FakeProvider()
        service = VLMService(provider, crop_margin_ratio=0)
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        result = service.describe(image, BBox(2, 3, 7, 8), "leaf_spot")

        self.assertEqual(result.summary, "可见叶片局部褐色斑点")
        self.assertEqual(result.severity, "medium")
        self.assertEqual(result.suggestion, "建议现场复核并持续观察。")
        self.assertEqual(provider.image.shape, (5, 5, 3))
        self.assertEqual(provider.disease_type, "leaf_spot")

    def test_http_endpoint_returns_structured_provider_result(self) -> None:
        image_buffer = io.BytesIO()
        Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8)).save(
            image_buffer, format="JPEG"
        )
        provider = _FakeProvider()
        settings = VLMSettings(
            enabled=True,
            base_url="https://provider.invalid/v1",
            model="test-model",
            api_key_env="TEST_KEY",
            timeout_s=1,
            crop_margin_ratio=0,
            jpeg_quality=90,
        )
        app = create_app(settings, provider_factory=lambda _settings: provider)
        app.state.service = VLMService(provider, crop_margin_ratio=0)
        endpoint = next(
            route.endpoint
            for route in app.routes
            if isinstance(route, APIRoute) and route.path == "/v1/describe"
        )
        response = asyncio.run(
            endpoint(
                disease_type="leaf_spot",
                x1=2,
                y1=3,
                x2=7,
                y2=8,
                image=UploadFile(
                    filename="frame.jpg", file=io.BytesIO(image_buffer.getvalue())
                ),
            )
        )

        self.assertEqual(response.summary, "可见叶片局部褐色斑点")
        self.assertEqual(response.severity, "medium")
        self.assertEqual(response.suggestion, "建议现场复核并持续观察。")
        self.assertEqual(provider.disease_type, "leaf_spot")


if __name__ == "__main__":
    unittest.main()
