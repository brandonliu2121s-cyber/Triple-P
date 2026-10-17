"""Behaviour model.

Owner: Data/ML.

`BehaviorModel.predict(x) -> (score 0-100, [(feature_name, impact_points), ...])`, where `x` is aligned
with FEATURE_NAMES.

- If `models/behavior_xgb.json` exists (produced by `python -m core.train <paysim.csv>`), the score is
  the XGBoost fraud probability x 100 and the impacts are exact TreeSHAP contributions
  (xgboost `pred_contribs`, no shap package needed at serving time).
- Otherwise it falls back to transparent rules so the pipeline still runs.

PaySim only supports MODEL_FEATURES. Signals it can't teach (new payee, device change, login cadence,
payee novelty, senior flag, amount vs personal baseline) stay rule-based: they are shown as reasons and
feed the context score, but they do not change the XGBoost score.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .features import FEATURE_NAMES, MODEL_FEATURES

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "behavior_xgb.json"

# Rule weights: the fallback scorer, and the display-only impacts for non-model features.
_RULE_WEIGHTS = {
    "amount_z": 4.0,          # per z-unit, capped below
    "balance_pct": 35.0,
    "new_beneficiary": 18.0,
    "device_change": 15.0,
    "login_gap_z": 3.0,
    "payee_novelty": 10.0,
    "is_senior": 4.0,
    "amount_log": 0.0,
    "drains_account": 0.0,
    "high_risk_type": 0.0,
}


class BehaviorModel:
    def __init__(self, path: str | Path | None = DEFAULT_MODEL_PATH) -> None:
        self.booster = None
        if path is not None and Path(path).exists():
            self._load(Path(path))

    @property
    def trained(self) -> bool:
        return self.booster is not None

    def _load(self, path: Path) -> None:
        import xgboost as xgb   # imported lazily so rule-only runs don't need xgboost
        self.booster = xgb.Booster()
        self.booster.load_model(str(path))
        self._xgb = xgb

    def _rule_impacts(self, x: list[float], names: list[str]) -> list[tuple[str, float]]:
        idx = {n: i for i, n in enumerate(FEATURE_NAMES)}
        return [(n, min(max(x[idx[n]], 0.0) * _RULE_WEIGHTS[n], 40.0)) for n in names]

    def _predict_xgb(self, x: list[float]) -> tuple[float, list[tuple[str, float]]]:
        idx = {n: i for i, n in enumerate(FEATURE_NAMES)}
        row = np.array([[x[idx[n]] for n in MODEL_FEATURES]], dtype="float32")
        dm = self._xgb.DMatrix(row, feature_names=MODEL_FEATURES)
        contribs = self.booster.predict(dm, pred_contribs=True)[0]       # last column is the bias
        prob = 1.0 / (1.0 + math.exp(-float(contribs.sum())))
        score = 100.0 * prob
        pos = np.clip(contribs[:-1], 0.0, None)                          # split the score over risk-raising features
        share = pos / pos.sum() if pos.sum() > 0 else pos
        impacts = [(n, float(score * s)) for n, s in zip(MODEL_FEATURES, share)]
        impacts += self._rule_impacts(x, [n for n in FEATURE_NAMES if n not in MODEL_FEATURES])
        impacts.sort(key=lambda t: t[1], reverse=True)
        return score, impacts

    def predict(self, x: list[float]) -> tuple[float, list[tuple[str, float]]]:
        if self.trained:
            return self._predict_xgb(x)
        impacts = self._rule_impacts(x, FEATURE_NAMES)
        score = min(sum(i for _, i in impacts), 100.0)
        impacts.sort(key=lambda t: t[1], reverse=True)
        return score, impacts


REASON_TEXT = {
    "amount_z": "Amount is far above this customer's usual transfers",
    "balance_pct": "Transfer drains a large share of the account balance",
    "new_beneficiary": "Payee was added very recently",
    "device_change": "Login from a new or unrecognised device",
    "login_gap_z": "Unusual login pattern for this customer",
    "payee_novelty": "Customer has never paid this account before",
    "is_senior": "Customer is in a higher-risk group for impersonation scams",
    "amount_log": "Unusually large transfer amount",
    "drains_account": "Transfer empties the account",
    "high_risk_type": "Transfer type commonly used in fraud (transfer / cash-out)",
    "repeat_payee": "Repeated transfers to a payee first paid minutes ago",
    "escalating": "Each transfer to this payee is larger than the last",
    "network": "Receiving account looks like part of a money-mule ring",
}
