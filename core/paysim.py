"""PaySim loading and model features.

Owner: Data/ML. Get the data (Kaggle: ealaxi/paysim1), e.g.

    kaggle datasets download -d ealaxi/paysim1 -p data --unzip

PaySim columns: step,type,amount,nameOrig,oldbalanceOrg,newbalanceOrig,nameDest,
oldbalanceDest,newbalanceDest,isFraud,isFlaggedFraud.

The feature maths here must match `core.features.featurize` for the MODEL_FEATURES columns,
or the model sees different inputs in training and at serving time (tests/test_train.py checks it).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import HIGH_RISK_TYPES, MODEL_FEATURES

USECOLS = ["step", "type", "amount", "nameOrig", "oldbalanceOrg", "nameDest", "isFraud"]


def load(path: str, limit: int | None = None) -> pd.DataFrame:
    return pd.read_csv(path, usecols=USECOLS, nrows=limit)


def model_features(df: pd.DataFrame) -> pd.DataFrame:
    amount = df["amount"].astype("float64")
    bal = df["oldbalanceOrg"].astype("float64")
    pct = np.where(bal > 0, np.minimum(amount / np.where(bal > 0, bal, 1.0), 1.0), 0.0)
    out = pd.DataFrame({
        "amount_log": np.log1p(amount),
        "balance_pct": pct,
        "drains_account": (pct >= 0.99).astype("float64"),
        "high_risk_type": df["type"].isin(HIGH_RISK_TYPES).astype("float64"),
    }, index=df.index)
    return out[MODEL_FEATURES]
