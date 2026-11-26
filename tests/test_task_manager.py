import asyncio
import threading
import unittest

from fastapi.routing import APIRoute

from stream_service.control_api import (
    CreateSessionRequest,
    OutputRequest,
    create_app,
)
from stream_service.task_manager import (
    DetectionTask,
    OutputAlreadyExists,
    OutputManager,
    OutputTask,
    TaskAlreadyExists,
    TaskCapacityExceeded,
    TaskManager,
)


CALLBACK_URL = "http://127.0.0.1:6789/manage/api/v1/ai-gateway/events"


class TaskManagerTests(unittest.TestCase):
    def test_create_is_idempotent_for_same_session_request(self) -> None:
        manager = TaskManager(lambda _task, stop_event: stop_event.wait(), 1)
        task = DetectionTask(
            "session-1",
            "rtsp://camera.example/live",
            ("crack",),
            CALLBACK_URL,
        )
        try:
            first = manager.start(task)
            second = manager.start(task)

            self.assertEqual(second.gateway_session_id, first.gateway_session_id)
            self.assertEqual(second.accepted_at, first.accepted_at)
            with self.assertRaises(TaskAlreadyExists):
                manager.start(
                    DetectionTask(
                        "session-1",
                        "rtsp://camera.example/other",
                        ("crack",),
                        CALLBACK_URL,
                    )
                )
        finally:
            manager.shutdown()

    def test_active_session_limit_is_enforced(self) -> None:
        manager = TaskManager(lambda _task, stop_event: stop_event.wait(), 1)
        try:
            manager.start(
                DetectionTask(
                    "session-1",
                    "rtmp://camera.example/live",
                    ("crack",),
                    CALLBACK_URL,
                )
            )
            with self.assertRaises(TaskCapacityExceeded):
                manager.start(
                    DetectionTask(
                        "session-2",
                        "rtmp://camera.example/live",
                        ("crack",),
                        CALLBACK_URL,
                    )
                )
        finally:
            manager.shutdown()

    def test_output_stop_waits_until_worker_disconnects(self) -> None:
        entered = threading.Event()
        manager = OutputManager(
            lambda _task, stop_event, set_status: (
                set_status("pushing"),
                entered.set(),
                stop_event.wait(),
            ),
            1,
        )
        try:
            manager.start(
                OutputTask(
                    "session-1",
                    "rtsp://camera.example/live",
                    ("crack",),
                    "rtmp://media.example/ai/camera",
                )
            )
            self.assertTrue(entered.wait(timeout=2))

            stopped = manager.stop("session-1")

            self.assertEqual(stopped.status, "stopped")
        finally:
            manager.shutdown()

    def test_output_start_is_idempotent_after_stop(self) -> None:
        entered = threading.Event()
        runs = 0

        def worker(_task, stop_event, set_status) -> None:
            nonlocal runs
            runs += 1
            set_status("pushing")
            entered.set()
            stop_event.wait()

        manager = OutputManager(worker, 1)
        task = OutputTask(
            "session-1",
            "rtsp://camera.example/live",
            ("crack",),
            "rtmp://media.example/ai/camera",
        )
        try:
            first = manager.start(task)
            self.assertTrue(entered.wait(timeout=2))
            manager.stop("session-1")

            repeated = manager.start(task)

            self.assertEqual(repeated.output_task_id, first.output_task_id)
            self.assertEqual(runs, 1)
            with self.assertRaises(OutputAlreadyExists):
                manager.start(
                    OutputTask(
                        "session-1",
                        task.stream_url,
                        task.algorithm_ids,
                        "rtmp://media.example/ai/other",
                    )
                )
        finally:
            manager.shutdown()

    def test_gateway_routes_create_session_and_start_output(self) -> None:
        session_entered = threading.Event()
        output_entered = threading.Event()

        def session_worker(_task, stop_event) -> None:
            session_entered.set()
            stop_event.wait()

        def output_worker(_task, stop_event, set_status) -> None:
            set_status("pushing")
            output_entered.set()
            stop_event.wait()

        sessions = TaskManager(session_worker, 2)
        outputs = OutputManager(output_worker, 2)
        try:
            app = create_app(sessions, outputs)
            endpoints = {
                (method, route.path): route.endpoint
                for route in app.routes
                if isinstance(route, APIRoute)
                for method in route.methods
            }
            create_session = endpoints[("POST", "/api/v1/live-sessions")]
            stop_session = endpoints[
                ("POST", "/api/v1/live-sessions/{session_id}:stop")
            ]
            start_output = endpoints[
                ("POST", "/api/v1/live-sessions/{session_id}/output")
            ]
            get_output = endpoints[
                ("GET", "/api/v1/live-sessions/{session_id}/output")
            ]
            stop_output = endpoints[
                ("DELETE", "/api/v1/live-sessions/{session_id}/output")
            ]
            create_response = asyncio.run(
                create_session(
                    CreateSessionRequest(
                        sessionId="session-1",
                        jobId="job-1",
                        streamUrl="rtsp://camera.example/live",
                        algorithmIds=["crack"],
                        callbackUrl=CALLBACK_URL,
                    )
                )
            )
            self.assertTrue(session_entered.wait(timeout=2))
            self.assertEqual(create_response.status, "accepted")
            self.assertEqual(create_response.session_id, "session-1")
            self.assertNotIn(("POST", "/v1/tasks"), endpoints)
            self.assertNotIn(("POST", "/upload"), endpoints)

            output_response = asyncio.run(
                start_output(
                    "session-1",
                    OutputRequest(
                        streamUrl="rtsp://camera.example/live",
                        algorithmIds=["crack"],
                        outputStreamUrl="rtmp://media.example/ai/camera",
                    ),
                )
            )
            self.assertTrue(output_entered.wait(timeout=2))
            current = asyncio.run(get_output("session-1"))
            self.assertIn(current.status, {"starting", "pushing"})
            stopped = asyncio.run(stop_output("session-1"))
            self.assertEqual(stopped.output_task_id, output_response.output_task_id)
            self.assertEqual(stopped.status, "stopped")
            stop_response = asyncio.run(stop_session("session-1", None))
            self.assertEqual(stop_response.status, "stopping")
        finally:
            outputs.shutdown()
            sessions.shutdown()

    def test_output_url_requires_ai_app_and_no_secret(self) -> None:
        with self.assertRaisesRegex(ValueError, "ai app 路径"):
            OutputRequest(
                streamUrl="rtsp://camera.example/live",
                algorithmIds=["crack"],
                outputStreamUrl="rtmp://media.example/live/camera",
            )
        with self.assertRaisesRegex(ValueError, "不含密钥"):
            OutputRequest(
                streamUrl="rtsp://camera.example/live",
                algorithmIds=["crack"],
                outputStreamUrl="rtmp://media.example/ai/camera?secret=inline",
            )

    def test_task_rejects_non_stream_url(self) -> None:
        with self.assertRaisesRegex(ValueError, "streamUrl 地址无效"):
            DetectionTask(
                "session-1",
                "https://",
                ("crack",),
                CALLBACK_URL,
            )


if __name__ == "__main__":
    unittest.main()
