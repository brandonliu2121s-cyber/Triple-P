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


def load(path: str, limit: int | None = None, neg_frac: float = 1.0, seed: int = 42) -> pd.DataFrame:
    """Read PaySim in chunks, keeping every fraud row and a `neg_frac` random share of non-fraud rows.

    Sampling non-fraud uniformly keeps the time split valid and memory small (neg_frac=0.05 -> ~320k rows).
    """
    chunks = []
    for i, chunk in enumerate(pd.read_csv(path, usecols=USECOLS, nrows=limit, chunksize=500_000)):
        if neg_frac < 1.0:
            fraud = chunk[chunk["isFraud"] == 1]
            normal = chunk[chunk["isFraud"] == 0].sample(frac=neg_frac, random_state=seed + i)
            chunk = pd.concat([fraud, normal]).sort_index()
        chunks.append(chunk)
    return pd.concat(chunks, ignore_index=True)


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


if __name__ == "__main__":
    # Shrink the 470 MB file to a few MB you can share: python -m core.paysim big.csv data/paysim_sample.csv 0.02
    import sys
    src, dst, frac = sys.argv[1], sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 0.02
    out = load(src, neg_frac=frac)
    out.to_csv(dst, index=False)
    print(f"Wrote {len(out):,} rows ({int(out['isFraud'].sum()):,} fraud) to {dst}")
