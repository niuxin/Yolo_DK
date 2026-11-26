import base64
import io
import unittest

import numpy as np
from PIL import Image

from stream_service.consumer import (
    _build_callback_frames,
    _monotonic_media_timestamp,
)
from stream_service.stream import VideoInfo


def _jpeg_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(np.full((32, 48, 3), 80, dtype=np.uint8)).save(
        buffer, format="JPEG"
    )
    return buffer.getvalue()


class ConsumerPipelineTests(unittest.TestCase):
    def test_media_timestamp_remains_increasing_after_source_reconnect(self) -> None:
        last_timestamp = None
        offset = 0.0
        adjusted = []
        for timestamp in (-40.0, 0.0, 40.0, -40.0, 0.0):
            last_timestamp, offset = _monotonic_media_timestamp(
                timestamp, last_timestamp, offset, 25.0
            )
            adjusted.append(last_timestamp)

        self.assertEqual(adjusted, [0.0, 40.0, 80.0, 120.0, 160.0])

    def test_best_frame_events_are_grouped_as_protocol_payload(self) -> None:
        image_bytes = _jpeg_bytes()
        events = [
            (
                "crack",
                {
                    "target_id": "finding-1",
                    "class_id": 2,
                    "class_name": "crack",
                    "frame_id": 7,
                    "timestamp_ms": 240.0,
                    "bbox_xyxy": [4, 5, 24, 21],
                    "confidence": 0.9,
                    "image_bytes": image_bytes,
                },
            ),
            (
                "spalling",
                {
                    "target_id": "finding-2",
                    "class_id": 3,
                    "class_name": "spalling",
                    "frame_id": 7,
                    "timestamp_ms": 240.0,
                    "bbox_xyxy": [8, 9, 28, 29],
                    "confidence": 0.8,
                    "image_bytes": image_bytes,
                },
            ),
        ]

        frames, descriptions = _build_callback_frames(
            events,
            {7: 1_700_000_000_000},
            VideoInfo(48, 32, 25.0, 0.0),
        )

        self.assertEqual(len(frames), 1)
        frame = frames[0]
        self.assertEqual(frame["frameId"], "frame-7")
        self.assertEqual(frame["frameSeq"], 7)
        self.assertEqual(frame["mediaTimestampMs"], 240.0)
        self.assertEqual(frame["capturedAt"], 1_700_000_000_000)
        self.assertEqual((frame["width"], frame["height"]), (48, 32))
        encoded = str(frame["frameObjectKey"]).split(",", 1)[1]
        self.assertEqual(base64.b64decode(encoded), image_bytes)
        algorithm_results = frame["algorithmResults"]
        self.assertEqual(
            {item["algorithmId"] for item in algorithm_results},
            {"crack", "spalling"},
        )
        crack_detection = algorithm_results[0]["detections"][0]
        self.assertEqual(crack_detection["findingId"], "finding-1")
        self.assertEqual(crack_detection["bbox"], [4.0, 5.0, 20.0, 16.0])
        self.assertEqual(len(descriptions), 2)


if __name__ == "__main__":
    unittest.main()
