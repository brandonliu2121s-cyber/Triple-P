"""Live monitor page. Run the app with: streamlit run app.py   (Owner: UI/Integration)"""
import os

import streamlit as st

from core.pipeline import Pipeline
from core.scams import SCENARIOS
from ui import components as ui
from ui.graph_view import render_neighborhood

# Dev-only data switch (GUARDIANZ_DEV=1): demo data vs worst-case fixtures; the choice lives in ?data=
DEV = os.getenv("GUARDIANZ_DEV") == "1"
DATA_MODES = {"demo": "Demo", "worst": "Worst case", "empty": "Empty", "one": "One", "huge": "1,500 events"}
mode = "demo"


@st.cache_resource
def get_pipeline(mode: str = "demo") -> Pipeline:
    p = Pipeline(worst_case=(mode == "worst"))
    if mode == "one":
        p.step()
    elif mode == "huge":
        for _ in range(1500):
            p.step()
    return p


if DEV:
    start = st.query_params.get("data", "demo")
    mode = st.sidebar.segmented_control(
        "Dev: data", list(DATA_MODES), default=start if start in DATA_MODES else "demo",
        format_func=DATA_MODES.get, key="data_mode") or "demo"
    st.query_params["data"] = mode

p = get_pipeline(mode)

# ---- sidebar: controls -----------------------------------------------------
with st.sidebar:
    st.title("🛡️ GuardianZ")
    st.caption("Real-time elder scam & financial coercion interceptor")
    running = st.toggle("Stream running", value=mode in ("demo", "worst"), key=f"running_{mode}")
    rate = st.slider("Events per refresh", 1, 10, 3)
    st.subheader("Scam simulator")
    for key, sc in SCENARIOS.items():
        st.button(sc.label, help=sc.description, key=f"scam_{key}",
                  on_click=p.inject_scam, args=(key,), width="stretch")
    st.subheader("Tier thresholds")
    t = p.scorer.thresholds
    t.nudge = st.slider("Nudge ≥", 0, 100, int(t.nudge))
    t.hold = st.slider("Hold ≥", 0, 100, int(t.hold))
    t.escalate = st.slider("Escalate ≥", 0, 100, int(t.escalate))
    p.llm.use_llm = st.toggle("Use local LLM (Ollama)", value=p.llm.use_llm)

st.title("GuardianZ: live transaction protection")


@st.fragment(run_every=1.0)
def live() -> None:
    if running:
        for _ in range(rate):
            p.step()
    ui.metrics_row(p)
    feed_tab, case_tab, graph_tab, guardian_tab, z_tab = st.tabs(
        ["Live feed", "Analyst queue", "Mule graph", "Guardian alerts", "LinuxONE"])
    with feed_tab:
        ui.feed_table(p.history)
        flagged = [x for x in p.history if x.intervention]
        if flagged:
            ui.case_detail(p, flagged[-1])
    with case_tab:
        if p.cases:
            ui.case_detail(p, p.cases[-1])
            st.caption(f"{len(p.cases)} cases in queue")
        else:
            st.write("No held transactions yet.")
    with graph_tab:
        target = p.cases[-1].event.dest_id if p.cases else p.graph.random_mule()
        render_neighborhood(p.graph, target)
    with guardian_tab:
        ui.guardian_log(p)
    with z_tab:
        ui.linuxone_panel(p)


live()
