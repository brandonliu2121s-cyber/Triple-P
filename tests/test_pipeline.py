from core.pipeline import Pipeline
from core.scams import SCENARIOS
from core.schemas import Tier


def drain(p: Pipeline, n: int):
    return [p.step() for _ in range(n)]


def test_normal_traffic_mostly_allowed():
    p = Pipeline(warmup=400, use_llm=False)
    out = drain(p, 300)
    blocked = sum(x.result.tier in (Tier.HOLD, Tier.ESCALATE) for x in out)
    assert blocked / len(out) < 0.03


def test_every_scenario_is_held():
    for key in SCENARIOS:
        p = Pipeline(warmup=400, use_llm=False)
        p.inject_scam(key)
        scam = [x for x in drain(p, 10) if x.event.scenario == key]
        assert scam, key
        assert any(x.result.tier in (Tier.HOLD, Tier.ESCALATE) for x in scam), key


def test_held_senior_triggers_guardian_and_intervention():
    p = Pipeline(warmup=400, use_llm=False)
    p.inject_scam("impersonation")
    drain(p, 10)
    assert p.guardians.alerts
    assert any(c.intervention and c.intervention.analyst_brief for c in p.cases)


def test_latency_budget():
    p = Pipeline(warmup=200, use_llm=False)
    drain(p, 200)
    assert p.stats.p95_latency_ms < 50
