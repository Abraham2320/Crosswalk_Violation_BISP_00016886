import unittest
import sys
from pathlib import Path
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
from config import AppSettings
from OCR.engine import OCREngine
class OCREngineTests(unittest.TestCase):
    def test_ocr_cleaning_and_validation(self):
        engine = OCREngine(AppSettings())
        cleaned = engine.clean_text(" ab-1234 ")
        self.assertEqual(cleaned, "AB1234")
        self.assertTrue(engine.validate(cleaned, 0.9))
        self.assertFalse(engine.validate("BAD!", 0.9))
        self.assertFalse(engine.validate("AB1234", 0.1))
if __name__ == "__main__":
    unittest.main()
