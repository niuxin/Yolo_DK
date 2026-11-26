import asyncio
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from fastapi import UploadFile
from fastapi.routing import APIRoute
from PIL import Image

from disease_service.app import _AppState, create_app
from disease_service.config import load_settings
from disease_service.registry import DetectorRegistry
from disease_service.types import Detection


class _EmptyDetector:
    def detect(self, image_rgb: np.ndarray) -> list[Detection]:
        return []


class _RawDetector:
    def detect(self, image_rgb: np.ndarray) -> list[Detection]:
        return [
            Detection(
                disease_type="crack",
                confidence=0.93,
                bbox=(1.0, 2.0, 6.0, 7.0),
                class_id=4,
            )
        ]


class DiseaseServiceMetricsTests(unittest.TestCase):
    def test_successful_detect_request_is_reported_in_metrics_log(self) -> None:
        image_buffer = io.BytesIO()
        Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(
            image_buffer, format="JPEG"
        )
        config = "api: {}\nmodel: {}\ntracking: {}\n"

        with tempfile.TemporaryDirectory() as temporary_directory:
            config_path = Path(temporary_directory) / "config.yaml"
            config_path.write_text(config, encoding="utf-8")
            settings = load_settings(config_path)
            app = create_app(config_path, detector=_EmptyDetector())
            with (
                patch("disease_service.app.METRICS_LOG_INTERVAL_SECONDS", 0.000001),
                self.assertLogs("disease_service.app", level="INFO") as logs,
            ):
                app.state.service = _AppState(
                    settings,
                    DetectorRegistry(
                        settings.models,
                        {"default": _EmptyDetector()},
                    ),
                )
                endpoint = next(
                    route.endpoint
                    for route in app.routes
                    if isinstance(route, APIRoute)
                    and route.path == "/v1/detect"
                )
                for frame_id in (1, 2):
                    response = asyncio.run(
                        endpoint(
                            session_id="flight-1",
                            frame_id=frame_id,
                            detector_id=None,
                            image=UploadFile(
                                filename="frame.jpg",
                                file=io.BytesIO(image_buffer.getvalue()),
                            ),
                        )
                    )

        self.assertEqual(response.session_id, "flight-1")
        self.assertIn("session_id=flight-1", logs.output[0])
        self.assertIn("detector_id=default", logs.output[0])
        self.assertIn("frames=2 throughput=", logs.output[0])
        self.assertIn("avg_decode=", logs.output[0])
        self.assertIn("avg_model=", logs.output[0])
        self.assertIn("avg_handler=", logs.output[0])

    def test_detect_returns_the_same_raw_boxes_without_tracking_fields(self) -> None:
        image_buffer = io.BytesIO()
        Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(
            image_buffer, format="JPEG"
        )
        config = "api: {}\nmodel: {}\n"

        with tempfile.TemporaryDirectory() as temporary_directory:
            config_path = Path(temporary_directory) / "config.yaml"
            config_path.write_text(config, encoding="utf-8")
            settings = load_settings(config_path)
            app = create_app(config_path, detector=_RawDetector())
            app.state.service = _AppState(
                settings,
                DetectorRegistry(
                    settings.models,
                    {"default": _RawDetector()},
                ),
            )
            endpoint = next(
                route.endpoint
                for route in app.routes
                if isinstance(route, APIRoute) and route.path == "/v1/detect"
            )
            responses = [
                asyncio.run(
                    endpoint(
                        session_id="flight-raw",
                        frame_id=frame_id,
                        detector_id=None,
                        image=UploadFile(
                            filename="frame.jpg",
                            file=io.BytesIO(image_buffer.getvalue()),
                        ),
                    )
                )
                for frame_id in (1, 2)
            ]

        self.assertEqual(responses[0].detections, responses[1].detections)
        detection = responses[0].detections[0].model_dump()
        self.assertEqual(detection["class_id"], 4)
        self.assertEqual(detection["disease_type"], "crack")
        self.assertEqual(
            detection["bbox"],
            {"x1": 1.0, "y1": 2.0, "x2": 6.0, "y2": 7.0},
        )
        self.assertNotIn("track_id", detection)
        self.assertNotIn("quality_score", detection)


if __name__ == "__main__":
    unittest.main()
