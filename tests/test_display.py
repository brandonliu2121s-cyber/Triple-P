from streamlit.testing.v1 import AppTest

from core.pipeline import KEEP_CASES, Pipeline
from core.scams import SCENARIOS
from core.textutil import compact_usd, isolate, md_dollars, md_escape, plural, short_id

HAZARD = "Mrs. $Lee$ **VIP** <b>& co</b>"


def test_md_escape_neutralises_markdown_math_and_html():
    out = md_escape(HAZARD)
    for ch in "$*<>&":
        assert all(out[i - 1] == "\\" for i, c in enumerate(out) if c == ch)
    assert md_escape("MULE0_1") == "MULE0\\_1"
    assert md_dollars("- **Transaction:** $9,000") == "- **Transaction:** \\$9,000"


def test_formatters():
    assert compact_usd(0) == "$0" and compact_usd(12_345) == "$12,345"
    assert compact_usd(12_345_678.9) == "$12.3M" and compact_usd(2_500_000_000) == "$2.5B"
    assert compact_usd(-4_200_000) == "-$4.2M"
    assert plural(0, "case") == "0 cases" and plural(1, "case") == "1 case" and plural(1284, "case") == "1,284 cases"
    long = "bartholomew.fitzgerald@northwind-industries-holdings.example.com"
    assert len(short_id(long)) == 18 and short_id(long).startswith("bartholom") and short_id(long).endswith("e.com")
    assert short_id("ACC0001") == "ACC0001"
    assert isolate("x") == "⁨x⁩"


def test_unbounded_lists_are_capped_but_totals_keep_counting():
    p = Pipeline(warmup=200, use_llm=False, worst_case=True)
    for _ in range(40):
        for key in SCENARIOS:
            p.inject_scam(key)
    for _ in range(40 * 12 + 50):
        p.step()
    assert len(p.cases) <= KEEP_CASES
    assert p.stats.by_tier["hold"] + p.stats.by_tier["escalate"] >= len(p.cases)
    assert p.guardians.total >= len(p.guardians.alerts)


def _flagged_page():
    import streamlit as st
    from core.pipeline import Pipeline
    from core.scams import SCENARIOS
    from ui import components as ui

    p = Pipeline(warmup=300, use_llm=False, worst_case=True)
    user = p.stream.users["U006"]                    # the hazardous name
    p.stream.inject(SCENARIOS["impersonation"].build(user, p.graph.random_mule()))
    for _ in range(8):
        p.step()
    ui.guardian_log(p)
    ui.case_detail(p, p.cases[-1])
    ui.feed_table([])


def test_hazardous_name_renders_as_literal_text():
    at = AppTest.from_function(_flagged_page, default_timeout=60).run()
    assert not at.exception
    warning = at.warning[0].value
    assert "\\$Lee\\$" in warning and "\\*\\*VIP\\*\\*" in warning and "<b>" not in warning.replace("\\<b\\>", "")
    assert any("\\$Lee\\$" in i.value for i in at.info)
    assert any("No transactions yet" in i.value for i in at.info)       # empty feed has a message
