"""Streamlit rendering helpers. Owner: UI/Integration."""
from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from core.pipeline import Pipeline, Processed
from core.schemas import Tier
from core.textutil import compact_usd, md_dollars, md_escape, short_id

TIER_ICON = {Tier.ALLOW: "🟢", Tier.NUDGE: "🟡", Tier.HOLD: "🟠", Tier.ESCALATE: "🔴"}


def metrics_row(p: Pipeline) -> None:
    s = p.stats
    c = st.columns(5)
    c[0].metric("Transactions scored", f"{s.total:,}")
    c[1].metric("Held / escalated", s.by_tier["hold"] + s.by_tier["escalate"])
    c[2].metric("Funds protected", compact_usd(s.held_amount))
    c[3].metric("Avg latency", f"{s.avg_latency_ms:.1f} ms", f"p95 {s.p95_latency_ms:.1f} ms", delta_color="off")
    recall = f"{s.scam_caught}/{s.scam_events}" if s.scam_events else "n/a"
    c[4].metric("Scam events caught", recall, f"{s.false_positives} false positives", delta_color="off")


def feed_table(history: list[Processed], n: int = 15) -> None:
    if not history:
        st.info("No transactions yet. Turn on **Stream running** in the sidebar.")
        return
    # Long ids are middle-truncated (the ends tell two accounts apart); the full id is in the case detail below.
    rows = [{
        "": TIER_ICON[x.result.tier], "id": x.event.id, "customer": short_id(x.event.user_id),
        "to": short_id(x.event.dest_id), "amount": round(x.event.amount, 2), "risk": x.result.risk,
        "tier": x.result.tier.value, "ms": x.result.latency_ms,
        "top reason": x.result.reasons[0][0] if x.result.reasons else "",
        "scenario": x.event.scenario or "",
    } for x in reversed(history[-n:])]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def case_detail(p: Pipeline, item: Processed) -> None:
    ev, res, iv = item.event, item.result, item.intervention
    st.markdown(f"#### {TIER_ICON[res.tier]} Transaction #{ev.id}: \\${ev.amount:,.0f} -> {md_escape(ev.dest_id)}  "
                f"(risk **{res.risk}**, {res.tier.value})")
    st.caption(f"Customer {md_escape(ev.user_id)}")
    left, right = st.columns(2)
    with left:
        st.caption("Why it was flagged (SHAP-style top reasons)")
        if res.reasons:
            reasons = pd.DataFrame(res.reasons, columns=["reason", "impact"])
            st.altair_chart(   # horizontal, so the full reason text has room instead of a clipped axis label
                alt.Chart(reasons).mark_bar(color="#f5a524").encode(
                    y=alt.Y("reason:N", sort="-x", title=None, axis=alt.Axis(labelLimit=300)),
                    x=alt.X("impact:Q", title="Impact"), tooltip=["reason", "impact"],
                ).properties(height=34 * len(reasons) + 30),
                width="stretch")
        st.caption(f"behaviour {res.behavior_score} · network {res.network_score} · context {res.context_score}")
    with right:
        if iv:
            st.caption(f"Message to customer ({iv.source})")
            st.info(md_escape(iv.user_message))
            st.caption("Analyst brief")
            st.markdown(md_dollars(iv.analyst_brief))


def guardian_log(p: Pipeline) -> None:
    if not p.guardians.alerts:
        st.write("No guardian alerts yet. Trigger a scam from the sidebar.")
    for a in reversed(p.guardians.alerts[-8:]):
        st.warning(f"**To {a.guardian_name}** (re: {md_escape(a.user_id)}) · {md_escape(a.message)}")


def linuxone_panel(p: Pipeline) -> None:
    st.markdown("**Scoring runs where the transactions live.**")
    st.write(f"- Avg end-to-end scoring latency: **{p.stats.avg_latency_ms:.1f} ms** (p95 {p.stats.p95_latency_ms:.1f} ms)")
    st.write("- LLM inference: local (Ollama) with template fallback, no customer data sent to an external cloud")
    st.write("- Pipeline: Python / scikit-learn / XGBoost / NetworkX on LinuxONE")
    st.caption("TODO(UI): replace with real numbers measured on the LinuxONE instance.")
