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


# Measured with the trained PaySim model + fusion weights 0.20/0.35/0.45 (set once, not tuned to these scams).
# Every scam must at least be flagged; three reach HOLD on their first transfer. Romance only reaches NUDGE:
# its opening transfer has a new payee but no device change and a modest amount, so it is genuinely ambiguous.
EXPECTED_BEST_TIER = {"impersonation": Tier.HOLD, "investment": Tier.HOLD, "tech_support": Tier.HOLD,
                      "romance": Tier.NUDGE}
_ORDER = [Tier.ALLOW, Tier.NUDGE, Tier.HOLD, Tier.ESCALATE]


def test_every_scenario_is_flagged_at_expected_tier():
    for key in SCENARIOS:
        p = Pipeline(warmup=400, use_llm=False)
        p.inject_scam(key)
        scam = [x for x in drain(p, 10) if x.event.scenario == key]
        assert scam, key
        best = max((x.result.tier for x in scam), key=_ORDER.index)
        assert _ORDER.index(best) >= _ORDER.index(EXPECTED_BEST_TIER[key]), (key, best)


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


def test_allowed_events_carry_no_reasons_and_graph_view_is_capped():
    p = Pipeline(warmup=400, use_llm=False)
    out = drain(p, 200)
    assert all(not x.result.reasons for x in out if x.result.tier is Tier.ALLOW)
    for mule in sorted(p.graph.mule_ids)[:6]:
        assert len(p.graph.neighborhood(mule, max_nodes=8)) <= 8
