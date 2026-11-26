import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from disease_service.config import ModelSettings, load_settings
from disease_service.registry import DetectorRegistry
from disease_service.types import Detection


class _FakeDetector:
    def __init__(self, disease_type: str) -> None:
        self.disease_type = disease_type

    def detect(self, image_rgb: np.ndarray) -> list[Detection]:
        width, height = image_rgb.shape[1], image_rgb.shape[0]
        return [
            Detection(
                self.disease_type,
                0.9,
                (1, 2, float(width - 1), float(height - 1)),
            )
        ]


class _FakeHouseDetector:
    def __init__(self) -> None:
        self.image: np.ndarray | None = None

    def detect(self, image_bgr: np.ndarray) -> list[dict[str, object]]:
        self.image = image_bgr
        return [
            {
                "class_name": "house",
                "confidence": 0.91,
                "bbox_xyxy": [1, 2, 8, 9],
            }
        ]


class _FakeCustomDetector:
    def __init__(self, settings: ModelSettings) -> None:
        self.settings = settings

    def detect(self, image_rgb: np.ndarray) -> list[Detection]:
        return [
            Detection(
                "custom",
                0.88,
                (1, 2, float(image_rgb.shape[1] - 1), float(image_rgb.shape[0] - 1)),
            )
        ]


class DetectorRegistryTests(unittest.TestCase):
    def test_detector_id_routes_to_its_backend(self) -> None:
        settings = ModelSettings(
            weights=Path("unused.pt"),
            device="cpu",
            image_size=640,
            confidence_threshold=0.25,
            iou_threshold=0.45,
            class_names=(),
        )
        registry = DetectorRegistry(
            {"leaf": settings, "fruit": settings},
            {
                "leaf": _FakeDetector("leaf_spot"),
                "fruit": _FakeDetector("fruit_rot"),
            },
        )
        image = np.zeros((10, 10, 3), dtype=np.uint8)

        self.assertEqual(registry.detect("leaf", image)[0].disease_type, "leaf_spot")
        self.assertEqual(registry.detect("fruit", image)[0].disease_type, "fruit_rot")
        with self.assertRaisesRegex(ValueError, "未配置检测算法"):
            registry.detect("unknown", image)

    def test_model_registry_configuration_selects_default(self) -> None:
        yaml = """
api: {}
model:
  device: cpu
models:
  leaf:
    weights: leaf.pt
  fruit:
    weights: fruit.pt
default_model_id: fruit
tracking: {}
"""
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_path = Path(temporary_directory) / "disease.yaml"
            config_path.write_text(yaml, encoding="utf-8")
            settings = load_settings(config_path)

        self.assertEqual(settings.default_model_id, "fruit")
        self.assertEqual(settings.model.weights.name, "fruit.pt")
        self.assertEqual(set(settings.models), {"leaf", "fruit"})

    def test_house_model_configuration_and_rgb_adapter(self) -> None:
        yaml = """
api: {}
model: {}
models:
  default: {}
  house:
    backend: house_tiled
    weights: ../weights/house_v3_recommended.pt
    device: cpu
    image_size: 640
    confidence_threshold: 0.4
    iou_threshold: 0.45
    class_names: [house]
    tiled: true
    tile_size: 1280
    max_det: 300
    nms_iou_threshold: 0.45
default_model_id: default
tracking: {}
"""
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_path = Path(temporary_directory) / "disease.yaml"
            config_path.write_text(yaml, encoding="utf-8")
            settings = load_settings(config_path).models["house"]

        self.assertEqual(settings.backend, "house_tiled")
        self.assertEqual(settings.weights.name, "house_v3_recommended.pt")
        self.assertTrue(settings.tiled)
        self.assertEqual(settings.tile_size, 1280)

        fake_detector = _FakeHouseDetector()
        registry = DetectorRegistry({"house": settings})
        with patch(
            "disease_service.backends.house_impl.HouseTiledDetector",
            return_value=fake_detector,
        ) as house_detector:
            image_rgb = np.zeros((10, 10, 3), dtype=np.uint8)
            image_rgb[0, 0] = [10, 20, 30]
            detections = registry.detect("house", image_rgb)

        house_detector.assert_called_once_with(settings)
        self.assertEqual(fake_detector.image[0, 0].tolist(), [30, 20, 10])
        self.assertEqual(
            detections,
            [Detection("house", 0.91, (1.0, 2.0, 8.0, 9.0))],
        )

    def test_custom_backend_is_loaded_by_module_and_class(self) -> None:
        settings = ModelSettings(
            weights=Path("unused.pt"),
            device="cpu",
            image_size=640,
            confidence_threshold=0.25,
            iou_threshold=0.45,
            class_names=("custom",),
            backend="custom",
            backend_module="third_party.custom_detector",
            backend_class="CustomDetector",
        )
        registry = DetectorRegistry({"custom": settings})
        with patch(
            "disease_service.registry.import_module",
            return_value=SimpleNamespace(CustomDetector=_FakeCustomDetector),
        ):
            detections = registry.detect(
                "custom",
                np.zeros((10, 10, 3), dtype=np.uint8),
            )

        self.assertEqual(detections[0].disease_type, "custom")
        self.assertEqual(detections[0].confidence, 0.88)

    def test_custom_backend_configuration_is_parsed(self) -> None:
        yaml = """
api: {}
model: {}
models:
  custom:
    backend: custom
    backend_module: disease_service.backends.my_detector
    backend_class: MyDetectorBackend
    weights: my-model.pt
default_model_id: custom
tracking: {}
"""
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_path = Path(temporary_directory) / "disease.yaml"
            config_path.write_text(yaml, encoding="utf-8")
            settings = load_settings(config_path).models["custom"]

        self.assertEqual(settings.backend, "custom")
        self.assertEqual(settings.backend_module, "disease_service.backends.my_detector")
        self.assertEqual(settings.backend_class, "MyDetectorBackend")


if __name__ == "__main__":
    unittest.main()
