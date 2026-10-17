"""Streamlit rendering helpers. Owner: UI/Integration."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from core.pipeline import Pipeline, Processed
from core.schemas import Tier

TIER_ICON = {Tier.ALLOW: "🟢", Tier.NUDGE: "🟡", Tier.HOLD: "🟠", Tier.ESCALATE: "🔴"}


def metrics_row(p: Pipeline) -> None:
    s = p.stats
    c = st.columns(5)
    c[0].metric("Transactions scored", f"{s.total:,}")
    c[1].metric("Held / escalated", s.by_tier["hold"] + s.by_tier["escalate"])
    c[2].metric("Funds protected", f"${s.held_amount:,.0f}")
    c[3].metric("Avg latency", f"{s.avg_latency_ms:.1f} ms", f"p95 {s.p95_latency_ms:.1f} ms", delta_color="off")
    recall = f"{s.scam_caught}/{s.scam_events}" if s.scam_events else "n/a"
    c[4].metric("Scam events caught", recall, f"{s.false_positives} false positives", delta_color="off")


def feed_table(history: list[Processed], n: int = 15) -> None:
    rows = [{
        "": TIER_ICON[x.result.tier], "id": x.event.id, "customer": x.event.user_id,
        "to": x.event.dest_id, "amount": round(x.event.amount, 2), "risk": x.result.risk,
        "tier": x.result.tier.value, "ms": x.result.latency_ms,
        "top reason": x.result.reasons[0][0] if x.result.reasons else "",
        "scenario": x.event.scenario or "",
    } for x in reversed(history[-n:])]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def case_detail(p: Pipeline, item: Processed) -> None:
    ev, res, iv = item.event, item.result, item.intervention
    st.markdown(f"#### {TIER_ICON[res.tier]} Transaction #{ev.id}: ${ev.amount:,.0f} -> {ev.dest_id}  "
                f"(risk **{res.risk}**, {res.tier.value})")
    left, right = st.columns(2)
    with left:
        st.caption("Why it was flagged (SHAP-style top reasons)")
        if res.reasons:
            st.bar_chart(pd.DataFrame(res.reasons, columns=["reason", "impact"]).set_index("reason"))
        st.caption(f"behaviour {res.behavior_score} · network {res.network_score} · context {res.context_score}")
    with right:
        if iv:
            st.caption(f"Message to customer ({iv.source})")
            st.info(iv.user_message)
            st.caption("Analyst brief")
            st.markdown(iv.analyst_brief)


def guardian_log(p: Pipeline) -> None:
    if not p.guardians.alerts:
        st.write("No guardian alerts yet. Trigger a scam from the sidebar.")
    for a in reversed(p.guardians.alerts[-8:]):
        st.warning(f"**To {a.guardian_name}** (re: {a.user_id}) · {a.message}")


def linuxone_panel(p: Pipeline) -> None:
    st.markdown("**Scoring runs where the transactions live.**")
    st.write(f"- Avg end-to-end scoring latency: **{p.stats.avg_latency_ms:.1f} ms** (p95 {p.stats.p95_latency_ms:.1f} ms)")
    st.write("- LLM inference: local (Ollama) with template fallback, no customer data sent to an external cloud")
    st.write("- Pipeline: Python / scikit-learn / XGBoost / NetworkX on LinuxONE")
    st.caption("TODO(UI): replace with real numbers measured on the LinuxONE instance.")
