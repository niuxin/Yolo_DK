# -*- coding: utf-8 -*-
"""兼容旧导入；新代码请使用 disease_service.backends.house_impl。"""

from .backends.house_impl import HouseTiledDetector

__all__ = ["HouseTiledDetector"]
