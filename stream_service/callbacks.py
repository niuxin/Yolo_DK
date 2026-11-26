# -*- coding: utf-8 -*-
"""病害记录和描述回调契约。

职责：定义流处理完成病害记录入库、描述生成后的通知接口，并提供默认空实现。
回调入参是已落本地 SQLite 的记录或其 ID 和描述；返回 ``None``。
业务系统可注入自己的实现以同步写入外部数据库。
"""
from __future__ import annotations

from typing import Protocol

from .storage import DiseaseRecord


class DiseaseCallbacks(Protocol):
    """外部病害业务系统需要实现的两个同步回调。"""

    def on_record_saved(self, record: DiseaseRecord) -> str | int:
        """保存或接收记录，返回业务数据库中该记录的 ID。"""

    def on_description_ready(
        self, record_id: str | int, description: str
    ) -> None:
        """接收 on_record_saved 返回的业务记录 ID 和描述文本。"""


class NoOpDiseaseCallbacks:
    """默认回调实现；本地 SQLite 持久化仍由 ``DiseaseRepository`` 完成。"""

    def on_record_saved(self, record: DiseaseRecord) -> str:
        """返回本地 SQLite 记录 ID，不执行外部入库通知。"""
        return str(record.record_id)

    def on_description_ready(
        self, _record_id: str | int, _description: str
    ) -> None:
        """默认不执行外部描述更新通知。"""
        _ = (_record_id, _description)
