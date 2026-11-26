import unittest

import numpy as np

from stream_service.api_client import BBox, DiseaseDetection
from stream_service.config import load_settings
from stream_service.consumer import _draw_detections, _with_publish_secret
from stream_service.stream import build_puller_command, build_pusher_command


class StreamServicePushTests(unittest.TestCase):
    def test_gateway_puller_preserves_native_dimensions_and_pts(self) -> None:
        settings = load_settings("stream_service/config.yaml").stream
        command = build_puller_command(
            settings,
            every_frame=True,
            native_size=True,
            with_timestamps=True,
        )

        self.assertIn("showinfo", command[command.index("-vf") + 1])
        self.assertNotIn("scale=1280:720", command[command.index("-vf") + 1])
        self.assertNotIn("fps=5", command[command.index("-vf") + 1])

    def test_callback_puller_uses_configured_sampling_with_source_pts(self) -> None:
        settings = load_settings("stream_service/config.yaml").stream
        command = build_puller_command(
            settings, native_size=True, with_timestamps=True
        )
        filters = command[command.index("-vf") + 1]

        self.assertEqual(filters, "fps=5,showinfo")

    def test_pusher_preserves_source_resolution_and_frame_rate(self) -> None:
        settings = load_settings("stream_service/config.yaml").stream
        command = build_pusher_command(
            settings,
            "rtmp://media.example/ai/camera?secret=hidden",
            width=1920,
            height=1080,
            frame_rate=30,
        )

        self.assertEqual(command[command.index("-video_size") + 1], "1920x1080")
        self.assertEqual(command[command.index("-framerate") + 1], "30")
        self.assertEqual(command[command.index("-g") + 1], "60")

    def test_publish_secret_is_added_only_to_runtime_url(self) -> None:
        url = _with_publish_secret(
            "rtmp://media.example/ai/camera", "test-secret"
        )

        self.assertEqual(
            url,
            "rtmp://media.example/ai/camera?secret=test-secret",
        )

    def test_multiple_algorithms_are_drawn_on_the_same_frame(self) -> None:
        frame = np.zeros((20, 20, 3), dtype=np.uint8)
        first = DiseaseDetection("crack", 0.9, BBox(2, 2, 12, 12), 1)
        second = DiseaseDetection("spalling", 0.8, BBox(4, 4, 16, 16), 2)

        annotated = _draw_detections(
            frame,
            [("crack", [first]), ("spalling", [second])],
        )

        self.assertEqual(annotated.shape, frame.shape)
        self.assertGreater(int(annotated.sum()), 0)
        self.assertEqual(frame.sum(), 0)


if __name__ == "__main__":
    unittest.main()
