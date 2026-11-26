import hashlib
import hmac
import json
import os
import unittest
from unittest.mock import Mock, patch

from stream_service.callback_client import GatewayCallbackClient
from stream_service.config import GatewaySettings


class CallbackClientTests(unittest.TestCase):
    def test_retry_reuses_body_and_resigns_timestamp(self) -> None:
        settings = GatewaySettings(
            callback_app_id_env="TEST_APP_ID",
            callback_app_secret_env="TEST_APP_SECRET",
            output_secret_env="TEST_OUTPUT_SECRET",
            callback_allowed_hosts=(),
            output_allowed_hosts=(),
            allow_insecure_http=True,
            callback_retry_attempts=2,
            callback_retry_backoff_s=0,
        )
        session = Mock()
        retryable = Mock(status_code=503)
        accepted = Mock(status_code=200)

        def accepted_json():
            return {
                "eventId": json.loads(session.post.call_args.kwargs["data"])["eventId"],
                "accepted": True,
                "duplicate": True,
            }

        accepted.json.side_effect = accepted_json
        session.post.side_effect = [retryable, accepted]
        with (
            patch.dict(
                os.environ,
                {
                    "TEST_APP_ID": "gateway-test",
                    "TEST_APP_SECRET": "not-a-real-secret",
                },
            ),
            patch("stream_service.callback_client.requests.Session", return_value=session),
            patch("stream_service.callback_client.time.time", side_effect=[1000, 1001]),
            patch("stream_service.callback_client.time.sleep"),
        ):
            client = GatewayCallbackClient(
                "http://127.0.0.1/manage/api/v1/ai-gateway/events",
                settings,
            )
            try:
                event_id = client.send_description(
                    "session-1",
                    "crack",
                    "finding-1",
                    "可见裂缝",
                    "medium",
                    "建议复核",
                )
            finally:
                client.close()

        first, second = session.post.call_args_list
        self.assertEqual(first.kwargs["data"], second.kwargs["data"])
        self.assertNotEqual(
            first.kwargs["headers"]["X-Signature"],
            second.kwargs["headers"]["X-Signature"],
        )
        self.assertEqual(first.kwargs["headers"]["X-Timestamp"], "1000")
        self.assertEqual(second.kwargs["headers"]["X-Timestamp"], "1001")
        self.assertEqual(
            json.loads(first.kwargs["data"])["eventId"],
            event_id,
        )
        body_hash = hashlib.sha256(first.kwargs["data"]).hexdigest()
        signing_text = "\n".join(
            (
                "POST",
                "/manage/api/v1/ai-gateway/events",
                "gateway-test",
                "1000",
                body_hash,
            )
        )
        expected = hmac.new(
            b"not-a-real-secret",
            signing_text.encode(),
            hashlib.sha256,
        ).hexdigest()
        self.assertEqual(first.kwargs["headers"]["X-Signature"], expected)


if __name__ == "__main__":
    unittest.main()
