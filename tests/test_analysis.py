import pandas as pd

from core import analysis as an
from core.scams import SCENARIOS


def run(**kw):
    return an.run_analysis(n_events=kw.pop("n_events", 2000), campaigns_per_type=kw.pop("campaigns_per_type", 4), **kw)


def test_campaign_labels_match_scam_events():
    # Guards against id() reuse attaching normal events to old campaigns.
    for seed in (1, 2, 3):
        r = run(seed=seed)
        df = r.events
        assert (df["campaign"] >= 0).equals(df["is_scam"])
        sizes = df[df["is_scam"]].groupby("campaign").size()
        expected = {c.campaign: len(SCENARIOS[c.scenario].build(
            type("U", (), {"user_id": "u", "balance": 1.0, "is_senior": True, "typical_amount": 1.0})(), "m"))
            for c in r.campaigns.itertuples()}
        assert sizes.to_dict() == expected


def test_same_seed_same_result():
    a, b = run(seed=7), run(seed=7)
    cols = [c for c in a.events.columns if c != "latency_ms"]
    pd.testing.assert_frame_equal(a.events[cols], b.events[cols])
    pd.testing.assert_frame_equal(a.campaigns, b.campaigns)


def test_summary_and_findings_are_consistent():
    r = run()
    s = an.summarize(r)
    assert s["tp"] + s["fn"] == s["scam_events"]
    assert s["campaigns"] == 4 * len(SCENARIOS)
    assert abs((s["stopped"] + s["leaked"]) - r.campaigns["total"].sum()) < 1e-6
    sweep = an.threshold_sweep(r.events)
    assert sweep["recall"].is_monotonic_decreasing and sweep["flagged"].is_monotonic_decreasing
    assert all(body.strip() for _, body in an.narrative(r))


def test_worst_case_data_runs():
    r = run(n_events=800, worst_case=True)
    assert r.worst_case and len(r.events) >= 800
    assert not r.events[["risk", "behavior", "network", "context"]].isna().any().any()
