from core.features import REPEAT_WINDOW_S, PayeeHistory
from core.schemas import Event


def ev(ts, amount, dest="MULE", user="U1"):
    return Event(id=0, ts=ts, user_id=user, dest_id=dest, amount=amount, balance_before=10_000)


def test_first_transfer_has_no_signal():
    assert PayeeHistory().signal(ev(0, 100)) == (0, False)


def test_repeat_to_new_payee_with_escalation():
    h = PayeeHistory()
    h.record(ev(0, 100))
    assert h.signal(ev(60, 200)) == (1, True)
    h.record(ev(60, 200))
    assert h.signal(ev(120, 210)) == (2, False)      # repeated, but not escalating


def test_held_attempts_count_because_record_ignores_tier():
    h = PayeeHistory()
    h.record(ev(0, 500))                              # the scorer records every attempt, held or not
    assert h.signal(ev(30, 900))[0] == 1


def test_established_payee_is_not_flagged():
    h = PayeeHistory()
    h.record(ev(0, 100))
    assert h.signal(ev(REPEAT_WINDOW_S + 1, 300)) == (0, False)


def test_other_payee_or_customer_unaffected():
    h = PayeeHistory()
    h.record(ev(0, 100))
    assert h.signal(ev(10, 300, dest="OTHER")) == (0, False)
    assert h.signal(ev(10, 300, user="U2")) == (0, False)


def test_pipeline_surfaces_repeat_reason_on_romance_scam():
    from core.pipeline import Pipeline
    p = Pipeline(warmup=400, use_llm=False)
    p.inject_scam("romance")
    scam = [x for x in (p.step() for _ in range(8)) if x.event.scenario]
    assert any("Repeated transfers" in r for x in scam[1:] for r, _ in x.result.reasons)
