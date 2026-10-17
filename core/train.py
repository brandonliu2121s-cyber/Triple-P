"""Train the behaviour model on PaySim.

    python -m core.train data/PS_20174392719_1491204439457_log.csv [--limit N]

Writes models/behavior_xgb.json (loaded automatically by BehaviorModel) and
models/behavior_metrics.json. Split is by time (`step`), not random, so the test set is
genuinely "the future"; judge it on PR-AUC, not accuracy (fraud is ~0.1% of rows).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score

from . import paysim
from .features import MODEL_FEATURES

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODEL_DIR / "behavior_xgb.json"
METRICS_PATH = MODEL_DIR / "behavior_metrics.json"


def train(df, train_frac_steps: float = 0.8, seed: int = 42, n_estimators: int = 300) -> tuple[xgb.XGBClassifier, dict]:
    X, y = paysim.model_features(df), df["isFraud"].astype(int)
    cut = df["step"].quantile(train_frac_steps)
    tr, te = (df["step"] <= cut).to_numpy(), (df["step"] > cut).to_numpy()
    if y[tr].sum() == 0 or y[te].sum() == 0:
        raise SystemExit("Need fraud rows in both train and test splits; load more data.")

    pos, neg = int(y[tr].sum()), int((1 - y[tr]).sum())
    clf = xgb.XGBClassifier(
        n_estimators=n_estimators, max_depth=5, learning_rate=0.1, subsample=0.8, colsample_bytree=0.9,
        scale_pos_weight=min(neg / pos, 50.0),     # capped: full imbalance weight wrecks calibration
        tree_method="hist", eval_metric="aucpr", random_state=seed, n_jobs=-1,
    )
    clf.fit(X[tr], y[tr], eval_set=[(X[te], y[te])], verbose=False)

    p = clf.predict_proba(X[te])[:, 1]
    metrics = {
        "rows": int(len(df)), "train_rows": int(tr.sum()), "test_rows": int(te.sum()),
        "train_fraud": pos, "test_fraud": int(y[te].sum()), "features": MODEL_FEATURES,
        "pr_auc": float(average_precision_score(y[te], p)), "roc_auc": float(roc_auc_score(y[te], p)),
        "at_threshold": {},
    }
    for t in (0.3, 0.5, 0.7, 0.9):
        pr, rc, f1, _ = precision_recall_fscore_support(y[te], p >= t, average="binary", zero_division=0)
        metrics["at_threshold"][str(t)] = {"precision": float(pr), "recall": float(rc), "f1": float(f1)}
    return clf, metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--limit", type=int, default=None, help="read only the first N rows (quick runs)")
    args = ap.parse_args()

    df = paysim.load(args.csv, args.limit)
    print(f"Loaded {len(df):,} rows, {int(df['isFraud'].sum()):,} fraud")
    clf, metrics = train(df)
    MODEL_DIR.mkdir(exist_ok=True)
    clf.get_booster().save_model(str(MODEL_PATH))
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print(f"Saved {MODEL_PATH}")


if __name__ == "__main__":
    main()
