"""Plumbing tests for the PaySim training path.

These use a tiny frame in PaySim's schema generated here. They verify the code works and that training
and serving features agree; they say NOTHING about real-world model quality (run core.train on real PaySim).
"""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("xgboost")
pytest.importorskip("sklearn")

from core import paysim, train
from core.features import FEATURE_NAMES, MODEL_FEATURES, BaselineStore, featurize
from core.models import BehaviorModel
from core.schemas import Event


def fake_paysim(n=4000, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    typ = rng.choice(["PAYMENT", "CASH_IN", "TRANSFER", "CASH_OUT"], n)
    bal = rng.lognormal(9, 1, n)
    fraud = (np.isin(typ, ["TRANSFER", "CASH_OUT"]) & (rng.random(n) < 0.08)).astype(int)
    amount = np.where(fraud == 1, bal, bal * rng.uniform(0.01, 0.4, n))
    return pd.DataFrame({"step": rng.integers(1, 101, n), "type": typ, "amount": amount,
                         "nameOrig": [f"C{i}" for i in range(n)], "oldbalanceOrg": bal,
                         "nameDest": [f"M{i}" for i in range(n)], "isFraud": fraud})


def test_train_runs_and_reports_metrics():
    clf, m = train.train(fake_paysim(), n_estimators=40)
    assert 0.0 <= m["pr_auc"] <= 1.0 and m["features"] == MODEL_FEATURES
    assert m["test_fraud"] > 0


def test_training_and_serving_features_match():
    df = fake_paysim(50)
    X = paysim.model_features(df)
    for i, row in df.head(25).iterrows():
        ev = Event(id=i, ts=0, user_id="u", dest_id="d", amount=float(row.amount), type=row.type,
                   balance_before=float(row.oldbalanceOrg))
        vec = dict(zip(FEATURE_NAMES, featurize(ev, BaselineStore().get("u"))))
        live = {n: vec[n] for n in MODEL_FEATURES}
        assert live == pytest.approx(X.loc[i].to_dict())


def test_behavior_model_loads_booster_and_explains(tmp_path):
    clf, _ = train.train(fake_paysim(), n_estimators=40)
    path = tmp_path / "m.json"
    clf.get_booster().save_model(str(path))
    model = BehaviorModel(path)
    assert model.trained
    ev = Event(id=1, ts=0, user_id="u", dest_id="d", amount=9000, type="TRANSFER", balance_before=9000)
    score, impacts = model.predict(featurize(ev, BaselineStore().get("u")))
    assert 0 <= score <= 100 and {n for n, _ in impacts} == set(FEATURE_NAMES)
    assert score > 50   # a full-balance TRANSFER is the pattern the fake frame labels as fraud
