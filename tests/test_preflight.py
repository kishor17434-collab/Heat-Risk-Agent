import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.model.predict import load_model
from src.preflight import ProjectPreflightError, validate_required_paths
from src.model.train import train_and_save


class TestPreflight(unittest.TestCase):
    def test_validate_required_paths_raises_for_missing_files(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            missing = Path(tmp_dir) / "missing.csv"

            with self.assertRaises(ProjectPreflightError) as ctx:
                validate_required_paths([("combined data", missing)])

            message = str(ctx.exception)
            self.assertIn("combined data", message)
            self.assertIn("missing.csv", message)

    def test_validate_required_paths_accepts_existing_files(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            existing = Path(tmp_dir) / "present.csv"
            existing.write_text("timestamp,value\n2024-01-01,10\n", encoding="utf-8")

            validated = validate_required_paths([("combined data", existing)])

            self.assertEqual(validated, [existing])

    def test_load_model_raises_clear_error_when_missing(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            missing_model = Path(tmp_dir) / "best_model.pkl"

            with self.assertRaises(ProjectPreflightError) as ctx:
                load_model(missing_model)

            self.assertIn("Model artifact missing", str(ctx.exception))

    def test_train_and_save_fails_with_helpful_message_on_small_data(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_path = Path(tmp_dir) / "combined.csv"
            rows = []
            for i in range(40):
                rows.append(
                    {
                        "timestamp": f"2024-06-{(i % 28) + 1:02d} 00:00:00",
                        "location": "Houston_TX",
                        "region": "COAST",
                        "temp_f": 95.0 + i,
                        "temp_c": 35.0 + i,
                        "demand_mw": 22000.0 + i * 50,
                        "hour": i % 24,
                        "day_of_week": i % 7,
                        "month": 6,
                        "is_weekend": int((i % 7) >= 5),
                    }
                )
            pd.DataFrame(rows).to_csv(data_path, index=False)

            with self.assertRaises(ValueError) as ctx:
                train_and_save(combined_path=data_path, save=False)

            message = str(ctx.exception)
            self.assertIn("at least 100 rows", message.lower())
            self.assertIn("June–August", message)


if __name__ == "__main__":
    unittest.main()
