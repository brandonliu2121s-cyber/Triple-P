"""Behaviour model.

Owner: Data/ML.

`BehaviorModel` is the interface the scorer uses. Today it is a transparent rule-based
stand-in so the pipeline runs end-to-end on day one. Replace `predict` internals with
IsolationForest (unsupervised anomaly signal) + XGBoost (supervised on PaySim `isFraud`,
explained with shap.TreeExplainer) and keep the same return shape:

    predict(x) -> (score 0-100, [(feature_name, signed_impact), ...])
"""
from __future__ import annotations

from .features import FEATURE_NAMES

# Rule weights act as pseudo-SHAP impacts until the real model is trained.
_RULE_WEIGHTS = {
    "amount_z": 4.0,          # per z-unit, capped below
    "balance_pct": 35.0,
    "new_beneficiary": 18.0,
    "device_change": 15.0,
    "login_gap_z": 3.0,
    "payee_novelty": 10.0,
    "is_senior": 4.0,
}


class BehaviorModel:
    def __init__(self) -> None:
        self.trained = False

    def fit(self, X, y=None) -> "BehaviorModel":
        """TODO(Data/ML): train IsolationForest + XGBoost here; set self.trained = True."""
        return self

    def predict(self, x: list[float]) -> tuple[float, list[tuple[str, float]]]:
        impacts = []
        for name, value in zip(FEATURE_NAMES, x):
            impacts.append((name, min(max(value, 0.0) * _RULE_WEIGHTS[name], 40.0)))
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
    "network": "Receiving account looks like part of a money-mule ring",
}
