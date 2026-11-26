import tempfile
import unittest

import numpy as np

from unique_target_service.app import TargetDetection, UniqueTargetEngine


class _FakeCore:
    def __init__(self, stream_id: str) -> None:
        self.stream_id = stream_id
        self.calls = []
        self.review_items = []

    def push(self, image, frame_id, timestamp_ms, detections):
        self.calls.append((image.shape, frame_id, timestamp_ms, detections))
        return []

    def flush(self):
        return []

    def stats(self):
        return {"frames": len(self.calls)}


class UniqueTargetServiceTests(unittest.TestCase):
    def test_receives_raw_detections_and_enforces_per_session_order(self) -> None:
        cores = []

        def core_factory(**kwargs):
            core = _FakeCore(**kwargs)
            cores.append(core)
            return core

        with tempfile.TemporaryDirectory() as temporary_directory:
            engine = UniqueTargetEngine(temporary_directory, core_factory)
            detection = TargetDetection(
                class_id=3,
                class_name="crack",
                confidence=0.91,
                bbox_xyxy=(10, 20, 40, 50),
            )
            frame = np.zeros((60, 80, 3), dtype=np.uint8)
            response = engine.process(
                "flight-a", 1, 200, frame, [detection]
            )

            self.assertEqual(response["events"], [])
            self.assertEqual(cores[0].calls[0][0], (60, 80, 3))
            self.assertEqual(cores[0].calls[0][1:3], (1, 200))
            self.assertEqual(
                cores[0].calls[0][3][0]["bbox_xyxy"], detection.bbox_xyxy
            )
            with self.assertRaisesRegex(ValueError, "严格递增"):
                engine.process("flight-a", 1, 400, frame, [detection])

            engine.process("flight-b", 1, 200, frame, [detection])
            self.assertEqual(len(cores), 2)


if __name__ == "__main__":
    unittest.main()
