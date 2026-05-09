import unittest
import sys
from pathlib import Path
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
from logic.violation import ViolationDetector
class ViolationLogicTests(unittest.TestCase):
    def test_violation_triggers_once_on_enter_transition(self):
        detector = ViolationDetector([(0, 0), (10, 0), (10, 10), (0, 10)])
        pedestrians = [{"id": 7, "zone": "upper", "direction": "STATIC"}]
        active, trigger = detector.evaluate_vehicle(
            obj_id=1,
            obj_class="vehicle",
            obj_state="enter",
            vehicle_zone="upper",
            pedestrians_data=pedestrians,
        )
        self.assertTrue(active)
        self.assertIsNotNone(trigger)
        self.assertEqual(trigger.reason, "same_zone")
        active_inside, next_trigger = detector.evaluate_vehicle(
            obj_id=1,
            obj_class="vehicle",
            obj_state="inside",
            vehicle_zone="upper",
            pedestrians_data=pedestrians,
        )
        self.assertTrue(active_inside)
        self.assertIsNone(next_trigger)
    def test_violation_clears_after_vehicle_leaves(self):
        detector = ViolationDetector([(0, 0), (10, 0), (10, 10), (0, 10)])
        pedestrians = [{"id": 9, "zone": "upper", "direction": "UP"}]
        detector.evaluate_vehicle(2, "vehicle", "enter", "upper", pedestrians)
        active, _ = detector.evaluate_vehicle(2, "vehicle", "outside", "upper", pedestrians)
        self.assertFalse(active)
        self.assertNotIn(2, detector.active_violations)
if __name__ == "__main__":
    unittest.main()
