# -*- coding: utf-8 -*-
"""保存检测帧/代表帧图片并维护 SQLite 病害记录。"""
from __future__ import annotations

from contextlib import contextmanager
import io
import sqlite3
import threading
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Iterator
from uuid import uuid4

import numpy as np
from PIL import Image, UnidentifiedImageError

from .api_client import DiseaseDetection
from .config import StorageSettings


@dataclass(frozen=True)
class DiseaseRecord:
    """写入病害库的一条单帧检测或代表帧记录。"""

    track_id: int
    disease_type: str
    confidence: float
    bbox_x1: float
    bbox_y1: float
    bbox_x2: float
    bbox_y2: float
    quality_score: float
    image_path: Path
    image_width: int
    image_height: int
    session_id: str
    frame_id: int
    captured_at: datetime
    record_id: int = 0
    description: str | None = None
    target_id: str | None = None


class DiseaseRepository:
    """保存图片并维护可供前端查询的 SQLite 病害库。"""

    def __init__(self, settings: StorageSettings, jpeg_quality: int = 90) -> None:
        """初始化存储目录和数据库表。

        Args:
            settings: 图片目录、数据库文件和相关参数。

        Example:
            >>> repository = DiseaseRepository(settings.storage)
        """
        self._settings = settings
        self._jpeg_quality = jpeg_quality
        settings.image_dir.mkdir(parents=True, exist_ok=True)
        settings.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._initialize_database()

    def save_candidate_image(self, frame: np.ndarray) -> Path:
        """保存一个检测帧图片。

        Args:
            frame: HxWx3 BGR 图像。

        Returns:
            使用随机文件名保存的候选图片路径。
        """
        filename = f"{uuid4().hex}.jpg"
        image_path = self._settings.image_dir / filename
        Image.fromarray(frame[:, :, ::-1]).save(
            image_path, quality=self._jpeg_quality
        )
        return image_path

    def save_candidate_image_bytes(self, image_bytes: bytes) -> Path:
        """保存并校验目标服务返回的代表帧 JPEG。"""
        filename = f"{uuid4().hex}.jpg"
        image_path = self._settings.image_dir / filename
        try:
            with Image.open(io.BytesIO(image_bytes)) as image:
                image.load()
                image.convert("RGB").save(
                    image_path, format="JPEG", quality=self._jpeg_quality
                )
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError("代表帧图片无法解析") from exc
        return image_path

    def insert_record(
        self,
        session_id: str,
        frame_id: int,
        detection: DiseaseDetection,
        image_path: Path,
        captured_at: datetime,
        target_id: str | None = None,
        quality_score: float | None = None,
    ) -> DiseaseRecord:
        """把检测帧或代表帧元数据写入病害库。

        Args:
            session_id: 检测会话 ID。
            frame_id: 视频帧序号。
            detection: 当前记录对应的检测结果。
            image_path: 已保存的 JPEG 路径。
            captured_at: 帧采集时间。

        Returns:
            已写入的病害记录。

        Example:
            >>> record = repository.insert_record(
            ...     "drone-01", 120, detection, image_path, datetime.now()
            ... )
        """
        with Image.open(image_path) as image:
            width, height = image.size
        record = DiseaseRecord(
            track_id=0,
            disease_type=detection.disease_type,
            confidence=detection.confidence,
            bbox_x1=detection.bbox.x1,
            bbox_y1=detection.bbox.y1,
            bbox_x2=detection.bbox.x2,
            bbox_y2=detection.bbox.y2,
            quality_score=(
                detection.confidence if quality_score is None else quality_score
            ),
            image_path=image_path,
            image_width=width,
            image_height=height,
            session_id=session_id,
            frame_id=frame_id,
            captured_at=captured_at,
            target_id=target_id,
        )
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO disease_records (
                    track_id, disease_type, confidence,
                    bbox_x1, bbox_y1, bbox_x2, bbox_y2, quality_score,
                    image_path, image_width, image_height,
                    session_id, frame_id, captured_at, target_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.track_id,
                    record.disease_type,
                    record.confidence,
                    record.bbox_x1,
                    record.bbox_y1,
                    record.bbox_x2,
                    record.bbox_y2,
                    record.quality_score,
                    str(record.image_path),
                    record.image_width,
                    record.image_height,
                    record.session_id,
                    record.frame_id,
                    record.captured_at.isoformat(timespec="milliseconds"),
                    record.target_id,
                ),
            )
            record_id = int(cursor.lastrowid)
        return replace(record, record_id=record_id)

    def update_description(self, record_id: int, description: str) -> None:
        """更新病害描述字段。

        Args:
            record_id: ``insert_record`` 返回记录中的数据库主键。
            description: VLM 生成的非空描述。

        Raises:
            ValueError: ID 无效或描述为空。
            LookupError: 对应病害记录不存在。
        """
        normalized = description.strip()
        if record_id <= 0 or not normalized:
            raise ValueError("record_id 必须大于零且 description 不能为空")
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                "UPDATE disease_records SET description = ? WHERE id = ?",
                (normalized, record_id),
            )
            if cursor.rowcount != 1:
                raise LookupError(f"病害记录不存在: {record_id}")

    def list_records(
        self, limit: int = 100, offset: int = 0
    ) -> list[dict[str, object]]:
        """分页查询病害库，供前端或接口使用。

        Args:
            limit: 最多返回条数，必须大于零。
            offset: 跳过条数，不能小于零。

        Returns:
            含 bbox 对象的记录列表。

        Example:
            >>> repository.list_records(limit=20, offset=0)[0]["bbox"]
            {'x1': 100.0, 'y1': 120.0, 'x2': 220.0, 'y2': 260.0}
        """
        if limit <= 0 or offset < 0:
            raise ValueError("limit 必须大于零，offset 必须不小于零")
        with self._connect() as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                "SELECT * FROM disease_records ORDER BY id DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """打开数据库连接，退出时提交并关闭。

        ``sqlite3.Connection`` 的上下文管理器只提交事务而不关闭连接，
        在 Windows 上会一直占用数据库文件。

        Example:
            >>> with repository._connect() as connection:
            ...     connection.execute("SELECT 1")
        """
        connection = sqlite3.connect(self._settings.database_path, timeout=10)
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize_database(self) -> None:
        """创建病害记录表和索引。

        Example:
            >>> repository._initialize_database()
        """
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS disease_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    track_id INTEGER NOT NULL,
                    disease_type TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    bbox_x1 REAL NOT NULL,
                    bbox_y1 REAL NOT NULL,
                    bbox_x2 REAL NOT NULL,
                    bbox_y2 REAL NOT NULL,
                    quality_score REAL NOT NULL,
                    image_path TEXT NOT NULL,
                    image_width INTEGER NOT NULL,
                    image_height INTEGER NOT NULL,
                    session_id TEXT NOT NULL,
                    frame_id INTEGER NOT NULL,
                    captured_at TEXT NOT NULL,
                    description TEXT,
                    target_id TEXT
                )
                """
            )
            columns = {
                row[1]
                for row in connection.execute(
                    "PRAGMA table_info(disease_records)"
                ).fetchall()
            }
            for name, declaration in (
                ("description", "TEXT"),
                ("target_id", "TEXT"),
            ):
                if name not in columns:
                    connection.execute(
                        f"ALTER TABLE disease_records ADD COLUMN {name} {declaration}"
                    )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_disease_session ON disease_records(session_id)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_disease_track ON disease_records(track_id)"
            )

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, object]:
        """将 SQLite 行转换为前端友好的对象。

        Args:
            row: 查询结果行。

        Returns:
            含 bbox 和图片尺寸的 JSON 兼容对象。

        Example:
            >>> DiseaseRepository._row_to_dict(row)["track_id"]
            12
        """
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
            "image_path": row["image_path"],
            "image_width": row["image_width"],
            "image_height": row["image_height"],
            "session_id": row["session_id"],
            "frame_id": row["frame_id"],
            "captured_at": row["captured_at"],
            "description": row["description"],
            "target_id": row["target_id"],
        }