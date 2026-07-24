import unittest

from backend.module_c_decision_engine import DecisionEngine
from backend.schemas.decision import DecisionLabel, TradeAction
from backend.schemas.ingestion import AssetClass


class DecisionEngineConfidenceGateTest(unittest.TestCase):
    def test_directional_confidence_can_trigger_action_before_strong_threshold(self):
        engine = DecisionEngine()
        label, action = engine._label_and_action_from_score(
            score=0.067,
            asset_class=AssetClass.EQUITY_GLOBAL,
            confidence_pct=53.0,
        )
        self.assertEqual(label, DecisionLabel.FRUITFUL_TRADE)
        self.assertEqual(action, TradeAction.BUY)


if __name__ == "__main__":
    unittest.main()
