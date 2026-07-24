import unittest
from pathlib import Path


class EvaluationFrameworkSmokeTest(unittest.TestCase):
    def test_expected_modules_exist(self):
        root = Path(__file__).resolve().parents[1]
        self.assertTrue((root / "backend" / "audit_logger.py").exists())
        self.assertTrue((root / "backend" / "backtester.py").exists())


if __name__ == "__main__":
    unittest.main()
