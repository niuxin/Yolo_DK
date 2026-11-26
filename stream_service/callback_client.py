# -*- coding: utf-8 -*-
"""签名并可靠投递业务检测、病害描述回调。"""
from __future__ import annotations

import hashlib
import hmac
import ipaddress
import json
import os
import socket
import time
from urllib.parse import urlsplit
from uuid import uuid4

import requests

from .config import GatewaySettings


class GatewayCallbackClient:
    """按业务协议发送原始 JSON 字节并对可重试错误退避重试。"""

    def __init__(
        self,
        callback_url: str,
        settings: GatewaySettings,
        timeout_s: float = 10,
    ) -> None:
        self._url = callback_url
        self._settings = settings
        self._timeout_s = timeout_s
        self._app_id = os.environ.get(settings.callback_app_id_env, "").strip()
        self._secret = os.environ.get(settings.callback_app_secret_env, "")
        if not self._app_id:
            raise RuntimeError(
                f"回调 appId 环境变量未设置: {settings.callback_app_id_env}"
            )
        if not self._secret:
            raise RuntimeError(
                f"回调 appSecret 环境变量未设置: {settings.callback_app_secret_env}"
            )
        self._validate_target()
        self._session = requests.Session()

    def send_detection(self, session_id: str, frames: list[dict[str, object]]) -> str:
        event_id = str(uuid4())
        payload = {
            "eventType": "detection_result",
            "eventId": event_id,
            "sessionId": session_id,
            "frames": frames,
        }
        self._post(payload, event_id)
        return event_id

    def send_description(
        self,
        session_id: str,
        algorithm_id: str,
        finding_id: str,
        summary: str,
        severity: str,
        suggestion: str,
    ) -> str:
        event_id = str(uuid4())
        payload = {
            "eventType": "description_result",
            "eventId": event_id,
            "sessionId": session_id,
            "algorithmId": algorithm_id,
            "findingId": finding_id,
            "summary": summary,
            "severity": severity,
            "suggestion": suggestion,
        }
        self._post(payload, event_id)
        return event_id

    def close(self) -> None:
        self._session.close()

    def _post(self, payload: dict[str, object], event_id: str) -> None:
        body = json.dumps(
            payload, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        parsed = urlsplit(self._url)
        path_and_query = parsed.path or "/"
        if parsed.query:
            path_and_query = f"{path_and_query}?{parsed.query}"
        last_error: Exception | None = None

        for attempt in range(self._settings.callback_retry_attempts):
            timestamp = str(int(time.time()))
            body_hash = hashlib.sha256(body).hexdigest()
            signing_text = "\n".join(
                (
                    "POST",
                    path_and_query,
                    self._app_id,
                    timestamp,
                    body_hash,
                )
            )
            signature = hmac.new(
                self._secret.encode("utf-8"),
                signing_text.encode("utf-8"),
                hashlib.sha256,
            ).hexdigest()
            try:
                response = self._session.post(
                    self._url,
                    data=body,
                    headers={
                        "Content-Type": "application/json",
                        "X-App-Id": self._app_id,
                        "X-Timestamp": timestamp,
                        "X-Signature": signature,
                    },
                    timeout=self._timeout_s,
                    allow_redirects=False,
                )
            except (requests.Timeout, requests.ConnectionError) as exc:
                last_error = exc
            else:
                if response.status_code == 200:
                    try:
                        ack = response.json()
                    except ValueError as exc:
                        raise RuntimeError("业务回调返回了无效 JSON") from exc
                    if (
                        not isinstance(ack, dict)
                        or ack.get("eventId") != event_id
                        or ack.get("accepted") is not True
                    ):
                        raise RuntimeError("业务回调确认与 eventId 不匹配")
                    return
                if response.status_code != 429 and response.status_code < 500:
                    raise RuntimeError(
                        f"业务回调拒绝事件: HTTP {response.status_code}"
                    )
                last_error = RuntimeError(
                    f"业务回调暂时不可用: HTTP {response.status_code}"
                )

            if attempt + 1 < self._settings.callback_retry_attempts:
                time.sleep(
                    self._settings.callback_retry_backoff_s * (2**attempt)
                )

        raise RuntimeError(
            f"业务回调重试耗尽: event_id={event_id}"
        ) from last_error

    def _validate_target(self) -> None:
        parsed = urlsplit(self._url)
        hostname = (parsed.hostname or "").rstrip(".").lower()
        if not hostname:
            raise ValueError("callbackUrl 缺少主机名")
        if parsed.username or parsed.password or parsed.fragment:
            raise ValueError("callbackUrl 不能包含凭证或 fragment")
        allowlisted = hostname in self._settings.callback_allowed_hosts
        try:
            addresses = {
                ipaddress.ip_address(result[4][0])
                for result in socket.getaddrinfo(
                    hostname,
                    parsed.port or (443 if parsed.scheme == "https" else 80),
                    type=socket.SOCK_STREAM,
                )
            }
        except OSError as exc:
            raise ValueError("callbackUrl 主机名无法解析") from exc
        private_only = bool(addresses) and all(
            address.is_private or address.is_loopback or address.is_link_local
            for address in addresses
        )
        if not allowlisted and not private_only:
            raise ValueError("callbackUrl 主机必须位于白名单或解析到私有地址")
        if parsed.scheme.lower() == "http":
            if not self._settings.allow_insecure_http or not private_only:
                raise ValueError("HTTP callback 仅允许显式开启的可信内网地址")
