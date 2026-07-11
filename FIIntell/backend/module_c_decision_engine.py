"""Module C — Intelligence Layer (Inference).

DecisionEngine consumes Module B *group scores directly*:
- Technicals (40%)
- Sentiment (30%)
- Fundamentals (20%)
- Geopolitics (10%)

It supports two modes:
1) **Model mode**: load a pre-trained XGBoost classifier from `models/xgb_classifier.pkl`
   (or a path provided at init). The model is expected to predict P(Fruitful/Trade).
2) **Fallback mode**: transparent logic-based scoring with calibrated confidence.

Why the fallback exists:
- In real deployments you want the service to be resilient and keep responding even if the
  model file isn't present or is incompatible.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from backend.schemas.decision import DecisionLabel, DecisionResult, TradeAction
from backend.schemas.features import FeatureEngineeringResult
from backend.schemas.ingestion import AssetClass

logger = logging.getLogger(__name__)


def _clamp(x: float, lo: float, hi: float) -> float:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return 0.0
    return max(lo, min(hi, x))


def _sigmoid(x: float) -> float:
    # Numerically stable sigmoid for confidence shaping.
    x = float(_clamp(x, -20.0, 20.0))
    return 1.0 / (1.0 + math.exp(-x))


@dataclass(frozen=True)
class DecisionEngineWeights:
    technical: float = 0.40
    sentiment: float = 0.30
    fundamentals: float = 0.20
    geopolitics: float = 0.10

    def normalized(self) -> "DecisionEngineWeights":
        s = self.technical + self.sentiment + self.fundamentals + self.geopolitics
        if s <= 0:
            return self
        return DecisionEngineWeights(
            technical=self.technical / s,
            sentiment=self.sentiment / s,
            fundamentals=self.fundamentals / s,
            geopolitics=self.geopolitics / s,
        )


class DecisionEngine:
    def __init__(
        self,
        model_path: str | Path | None = None,
        weights: DecisionEngineWeights | None = None,
    ) -> None:
        self._weights = (weights or DecisionEngineWeights()).normalized()
        # Allow overriding the model location without code changes.
        import os

        env_model_path = os.getenv("FIINTELL_MODEL_PATH", "").strip()
        effective_path = model_path or (Path(env_model_path) if env_model_path else None)
        self._model_path = effective_path if effective_path else Path("models") / "xgb_classifier.pkl"
        self._model: Any | None = None
        self._model_load_error: str | None = None

        self._try_load_model()

    @property
    def using_model(self) -> bool:
        return self._model is not None

    def decide(self, features: FeatureEngineeringResult) -> DecisionResult:
        warnings: list[str] = list(features.warnings)
        errors: list[str] = list(features.errors)
        reasoning: list[str] = list(features.signals)

        # If upstream feature engineering failed, stop early.
        if errors:
            return DecisionResult(
                label=DecisionLabel.RISKY_AVOID,
                action=TradeAction.HOLD,
                confidence_pct=0.0,
                score=0.0,
                reasoning=reasoning or ["Feature engineering failed; decision suppressed."],
                warnings=warnings,
                errors=errors,
            )

        # Ensure group scores are within [-1, 1].
        t = _clamp(float(features.technical_score), -1.0, 1.0)
        s = _clamp(float(features.sentiment_score), -1.0, 1.0)
        f = _clamp(float(features.fundamentals_score), -1.0, 1.0)
        g = _clamp(float(features.geopolitics_score), -1.0, 1.0)

        if self._model is not None:
            try:
                import os

                proba_trade = self._predict_trade_probability(t, s, f, g)
                # Small training sets often yield nearly flat model probabilities across tickers.
                # Blend model output with the same weighted group-score used in fallback so each
                # ticker’s technical/sentiment/fundamentals/geopolitics still moves the needle.
                weighted = float(
                    self._weights.technical * t
                    + self._weights.sentiment * s
                    + self._weights.fundamentals * f
                    + self._weights.geopolitics * g
                )
                weighted = _clamp(weighted, -1.0, 1.0)
                heur_p = (weighted + 1.0) / 2.0  # map [-1,1] -> [0,1]

                alpha = float(os.getenv("FIINTELL_MODEL_BLEND", "0.0"))
                alpha = float(_clamp(alpha, 0.0, 1.0))

                blended_p = _clamp(alpha * proba_trade + (1.0 - alpha) * heur_p, 0.0, 1.0)
                score = float(2.0 * blended_p - 1.0)
                label, action = self._label_and_action_from_score(score, features.asset_class)
                confidence = float(_clamp(blended_p * 100.0, 0.0, 100.0))
                return DecisionResult(
                    label=label,
                    action=action,
                    confidence_pct=confidence,
                    score=score,
                    reasoning=reasoning
                    + [
                        f"Decision source: XGBoost blended (α={alpha:.2f}; model_p={proba_trade:.3f}, group_prior_p={heur_p:.3f})."
                    ],
                    warnings=warnings,
                    errors=[],
                )
            except Exception as e:
                # Hard failover to fallback mode.
                warnings.append(f"model inference failed; using fallback scoring: {e!s}")

        if self._model_load_error:
            warnings.append(f"model not loaded; using fallback scoring: {self._model_load_error}")

        # Fallback mode: weighted sum in [-1, 1]
        # NOTE: This is intentionally simple and explainable:
        # - If the composite score is strongly positive, we BUY.
        # - If strongly negative, we SELL (or avoid).
        # - Otherwise HOLD.
        score = float(
            self._weights.technical * t
            + self._weights.sentiment * s
            + self._weights.fundamentals * f
            + self._weights.geopolitics * g
        )
        score = _clamp(score, -1.0, 1.0)

        label, action = self._label_and_action_from_score(score, features.asset_class)

        # Confidence is shaped by distance from 0.0; we avoid returning 100% in heuristic mode.
        # Map abs(score) in [0..1] -> confidence in [50..95] with sigmoid smoothing.
        conf = 50.0 + 45.0 * _sigmoid(4.0 * abs(score))  # 50..~95
        conf = float(_clamp(conf, 0.0, 100.0))

        # Add compact explanation of the weight mix for transparency.
        reasoning.append(
            f"Weighted score={score:.2f} using "
            f"T={self._weights.technical:.0%}, S={self._weights.sentiment:.0%}, "
            f"F={self._weights.fundamentals:.0%}, G={self._weights.geopolitics:.0%}."
        )

        return DecisionResult(
            label=label,
            action=action,
            confidence_pct=conf,
            score=score,
            reasoning=reasoning,
            warnings=warnings,
            errors=[],
        )

    def _label_and_action_from_score(self, score: float, asset_class: AssetClass) -> tuple[DecisionLabel, TradeAction]:
        score = float(_clamp(score, -1.0, 1.0))
        # Thresholds tuned to avoid over-trading on noisy signals.
        threshold = 0.30 if asset_class in {AssetClass.CRYPTO, AssetClass.COMMODITY} else 0.20
       if score >= threshold:
            return DecisionLabel.FRUITFUL_TRADE, TradeAction.BUY
        if score <= -threshold:
            return DecisionLabel.RISKY_AVOID, TradeAction.SELL
        # Neutral zone
        return DecisionLabel.RISKY_AVOID, TradeAction.HOLD

    def _try_load_model(self) -> None:
        if not self._model_path.exists():
            self._model_load_error = f"missing model file at {self._model_path.as_posix()}"
            return

        try:
            import joblib

            self._model = joblib.load(self._model_path)
            self._model_load_error = None
        except Exception as e:
            self._model = None
            self._model_load_error = str(e)

    def _predict_trade_probability(self, t: float, s: float, f: float, g: float) -> float:
        # Model contract: expects 4 features in this exact order.
        X = np.array([[t, s, f, g]], dtype=np.float32)

        # Prefer predict_proba (sklearn-style). Fall back to XGBoost Booster raw predict.
        if hasattr(self._model, "predict_proba"):
            proba = self._model.predict_proba(X)
            # Assume class 1 is Fruitful/Trade
            p1 = float(proba[0][1])
            return _clamp(p1, 0.0, 1.0)

        if hasattr(self._model, "predict"):
            # Some xgboost wrappers return probabilities directly for binary:logistic.
            y = self._model.predict(X)
            if isinstance(y, (list, tuple, np.ndarray)) and len(y) > 0:
                return _clamp(float(y[0]), 0.0, 1.0)

        raise ValueError("model object does not support probability prediction")

