# -*- coding: utf-8 -*-
"""病害库只读访问层。

界面服务以 SQLite 只读 URI 打开 ``stream_service`` 的病害库，不建表、不写入，
因此不会与检测进程争锁，也不会改变任何现有行为。库文件不存在时返回空结果。
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from .config import UiStorageSettings


class RecordRepository:
    """只读查询病害记录和代表帧图片。"""

    def __init__(self, settings: UiStorageSettings) -> None:
        """保存存储配置。

        Args:
            settings: 病害库路径、图片目录和轮询间隔。
        """
        self._settings = settings

    @property
    def database_path(self) -> Path:
        """返回病害库文件路径。"""
        return self._settings.database_path

    @property
    def image_dir(self) -> Path:
        """返回代表帧图片目录。"""
        return self._settings.image_dir

    @property
    def poll_interval_s(self) -> float:
        """返回轮询新记录的间隔秒数。"""
        return self._settings.poll_interval_s

    def available(self) -> bool:
        """返回病害库文件是否存在。"""
        return self._settings.database_path.is_file()

    def list_records(
        self,
        limit: int = 50,
        offset: int = 0,
        session_id: str | None = None,
        after_id: int | None = None,
    ) -> list[dict[str, Any]]:
        """按 id 倒序分页查询病害记录。

        Args:
            limit: 最多返回条数，必须大于零。
            offset: 跳过条数，不能小于零。
            session_id: 只返回该会话的记录；None 表示不过滤。
            after_id: 只返回 id 大于该值的记录，用于增量轮询。

        Returns:
            前端友好的记录字典列表；库文件不存在时返回空列表。

        Example:
            >>> repository.list_records(limit=20)[0]["bbox"]
            {'x1': 100.0, 'y1': 120.0, 'x2': 220.0, 'y2': 260.0}
        """
        if limit <= 0 or offset < 0:
            raise ValueError("limit 必须大于零，offset 必须不小于零")
        conditions: list[str] = []
        parameters: list[Any] = []
        if session_id:
            conditions.append("session_id = ?")
            parameters.append(session_id)
        if after_id is not None:
            conditions.append("id > ?")
            parameters.append(int(after_id))
        where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
        parameters.extend([limit, offset])
        rows = self._query(
            f"SELECT * FROM disease_records{where} ORDER BY id DESC LIMIT ? OFFSET ?",
            parameters,
        )
        return [self._row_to_dict(row) for row in rows]

    def get_record(self, record_id: int) -> dict[str, Any] | None:
        """按主键查询单条记录。

        Args:
            record_id: 病害记录主键。

        Returns:
            记录字典；不存在时返回 None。
        """
        rows = self._query(
            "SELECT * FROM disease_records WHERE id = ?", [int(record_id)]
        )
        return self._row_to_dict(rows[0]) if rows else None

    def get_image_path(self, record_id: int) -> Path | None:
        """按主键解析代表帧图片路径。

        图片路径不通过 API 暴露给前端，只在服务端内部解析。

        Args:
            record_id: 病害记录主键。

        Returns:
            可读取的图片路径；记录或文件不存在时返回 None。
        """
        rows = self._query(
            "SELECT image_path FROM disease_records WHERE id = ?", [int(record_id)]
        )
        if not rows:
            return None
        return self.resolve_image(str(rows[0]["image_path"] or ""))

    def max_record_id(self) -> int:
        """返回当前最大记录 id；空库或库不存在时返回 0。"""
        rows = self._query("SELECT MAX(id) AS max_id FROM disease_records", [])
        if not rows:
            return 0
        value = rows[0]["max_id"]
        return int(value) if value is not None else 0

    def count_records(self, session_id: str | None = None) -> int:
        """统计记录总数，可按会话过滤。"""
        if session_id:
            rows = self._query(
                "SELECT COUNT(*) AS total FROM disease_records WHERE session_id = ?",
                [session_id],
            )
        else:
            rows = self._query(
                "SELECT COUNT(*) AS total FROM disease_records", []
            )
        return int(rows[0]["total"]) if rows else 0

    def list_sessions(self, limit: int = 50) -> list[dict[str, Any]]:
        """列出最近的检测会话及其记录数。

        Args:
            limit: 最多返回会话数。

        Returns:
            含 ``session_id``、``records``、``last_at`` 的列表。
        """
        rows = self._query(
            """
            SELECT session_id, COUNT(*) AS records, MAX(captured_at) AS last_at
            FROM disease_records
            GROUP BY session_id
            ORDER BY last_at DESC
            LIMIT ?
            """,
            [limit],
        )
        return [
            {
                "session_id": row["session_id"],
                "records": int(row["records"]),
                "last_at": row["last_at"],
            }
            for row in rows
        ]

    def resolve_image(self, image_path: str) -> Path | None:
        """把库中记录的图片路径解析为可读取的本地文件。

        库中存的是绝对路径；若文件已移动，则回退到图片目录下同名文件。
        解析结果必须位于图片目录内，防止路径穿越。

        Args:
            image_path: 库中记录的 ``image_path`` 字段。

        Returns:
            可读取的图片路径；不存在或越界时返回 None。
        """
        root = self._settings.image_dir.resolve()
        candidates = [Path(image_path)]
        candidates.append(root / Path(image_path).name)
        for candidate in candidates:
            try:
                resolved = candidate.expanduser().resolve()
            except OSError:
                continue
            if not resolved.is_file():
                continue
            if resolved != root and root not in resolved.parents:
                continue
            return resolved
        return None

    def _query(
        self, sql: str, parameters: list[Any]
    ) -> list[sqlite3.Row]:
        """以只读方式执行查询。

        Args:
            sql: SQL 语句。
            parameters: 绑定参数。

        Returns:
            查询结果行列表；库文件不存在或表未建立时返回空列表。
        """
        path = self._settings.database_path
        if not path.is_file():
            return []
        uri = f"file:{path.as_posix()}?mode=ro"
        try:
            connection = sqlite3.connect(uri, uri=True, timeout=5)
        except sqlite3.Error:
            return []
        try:
            connection.row_factory = sqlite3.Row
            return connection.execute(sql, parameters).fetchall()
        except sqlite3.Error:
            return []
        finally:
            connection.close()

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        """把 SQLite 行转换为前端友好的字典。

        Args:
            row: 查询结果行。

        Returns:
            含嵌套 ``bbox`` 的 JSON 兼容字典。

        Example:
            >>> RecordRepository._row_to_dict(row)["track_id"]
            12
        """
        keys = row.keys()
        return {
            "id": row["id"],
            "track_id": row["track_id"],
            "disease_type": row["disease_type"],
            "confidence": row["confidence"],
            "bbox": {
                "x1": row["bbox_x1"],
                "y1": row["bbox_y1"],
                "x2": row["bbox_x2"],
                "y2": row["bbox_y2"],
            },
            "quality_score": row["quality_score"],
            "image_width": row["image_width"],
            "image_height": row["image_height"],
            "session_id": row["session_id"],
            "frame_id": row["frame_id"],
            "captured_at": row["captured_at"],
            "description": row["description"] if "description" in keys else None,
            "target_id": row["target_id"] if "target_id" in keys else None,
            "image_url": f"/api/records/{row['id']}/image",
        }
