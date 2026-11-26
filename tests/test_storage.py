import sqlite3
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import numpy as np

from stream_service.api_client import BBox, DiseaseDetection
from stream_service.config import StorageSettings
from stream_service.storage import DiseaseRepository


class StorageTests(unittest.TestCase):
    def test_description_is_saved_for_inserted_record(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            settings = StorageSettings(
                image_dir=root / "images",
                database_path=root / "disease.db",
            )
            repository = DiseaseRepository(settings)
            detection = DiseaseDetection(
                disease_type="leaf_spot",
                confidence=0.9,
                bbox=BBox(1, 2, 8, 9),
            )
            image_path = repository.save_candidate_image(
                np.zeros((10, 10, 3), dtype=np.uint8)
            )
            record = repository.insert_record(
                "session/with/slashes",
                12,
                detection,
                image_path,
                datetime.now(),
                quality_score=0.8,
            )
            repository.update_description(record.record_id, "可见褐色斑点")

            saved = repository.list_records()[0]
            self.assertEqual(saved["id"], record.record_id)
            self.assertEqual(saved["description"], "可见褐色斑点")
            self.assertEqual(Path(saved["image_path"]).parent, settings.image_dir)

    def test_existing_database_gets_description_column(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            database = root / "legacy.db"
            with sqlite3.connect(database) as connection:
                connection.execute(
                    """
                    CREATE TABLE disease_records (
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
                        captured_at TEXT NOT NULL
                    )
                    """
                )
            settings = StorageSettings(
                image_dir=root / "images",
                database_path=database,
            )
            self.assertEqual(DiseaseRepository(settings).list_records(), [])
            with sqlite3.connect(database) as connection:
                columns = {
                    row[1]
                    for row in connection.execute(
                        "PRAGMA table_info(disease_records)"
                    ).fetchall()
                }
            self.assertIn("description", columns)


if __name__ == "__main__":
    unittest.main()
