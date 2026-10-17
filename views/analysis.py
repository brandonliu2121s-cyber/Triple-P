"""Analysis page: press "Run analysis" to simulate a batch, then explore charts and written findings.

Not live: nothing refreshes until the button is pressed again. Owner: UI/Integration.
"""
import os

import altair as alt
import streamlit as st

from core import analysis as an
from core.scoring import Thresholds
from ui.graph_view import render_neighborhood

GROUP_COLORS = alt.Scale(domain=["Normal", "Scam"], range=["#8da3b8", "#e5484d"])
TIER_ORDER = ["allow", "nudge", "hold", "escalate"]
DEV = os.getenv("GUARDIANZ_DEV") == "1"

st.title("Analysis")
st.caption("A one-off batch run on simulated traffic with scripted scam campaigns mixed in. "
           "Nothing here is live; press the button to run it again.")

# ---- controls --------------------------------------------------------------
default = Thresholds()
with st.form("analysis_form"):
    c1, c2, c3 = st.columns(3)
    n_events = c1.select_slider("Transactions", [1_000, 2_500, 5_000, 10_000, 20_000], value=5_000)
    per_type = c2.slider("Scam campaigns per type", 1, 25, 8, help="Four scam types, so total campaigns = 4 x this.")
    seed = c3.number_input("Random seed", min_value=0, value=1, step=1, help="Same seed, same run.")
    with st.expander("Tier thresholds"):
        t1, t2, t3 = st.columns(3)
        nudge = t1.slider("Nudge ≥", 0, 100, int(default.nudge))
        hold = t2.slider("Hold ≥", 0, 100, int(default.hold))
        escalate = t3.slider("Escalate ≥", 0, 100, int(default.escalate))
    worst = st.checkbox("Use worst-case customers and payees (dev fixtures)") if DEV else False
    run = st.form_submit_button("Run analysis", type="primary")

if run:
    with st.spinner(f"Scoring {n_events:,} transactions..."):
        st.session_state["analysis"] = an.run_analysis(
            n_events=n_events, campaigns_per_type=per_type, seed=int(seed), worst_case=worst,
            thresholds=Thresholds(float(nudge), float(hold), float(escalate)))

res = st.session_state.get("analysis")
if res is None:
    st.info("Choose a size and press **Run analysis**. You get detection results, a risk-score distribution, "
            "a threshold tuner, signal and reason breakdowns, the mule network, and written findings.")
    st.stop()

s = an.summarize(res)
df, camp = res.events, res.campaigns
st.caption(f"Ran {res.n_events:,} transactions in {res.elapsed_s:.1f} s · seed {res.seed} · "
           f"thresholds {res.thresholds.nudge:.0f} / {res.thresholds.hold:.0f} / {res.thresholds.escalate:.0f}"
           + (" · worst-case data" if res.worst_case else ""))

# ---- headline numbers --------------------------------------------------------
m = st.columns(5)
m[0].metric("Scam campaigns caught", f"{s['campaigns_caught']}/{s['campaigns']}", an.pct(s["catch_rate"]), delta_color="off")
m[1].metric("Scam transfers flagged", an.pct(s["event_recall"]), f"{s['tp']} of {s['scam_events']}", delta_color="off")
m[2].metric("False positives", f"{s['fp']:,}", f"{an.pct(s['fpr'], 2)} of normal", delta_color="off")
m[3].metric("Funds stopped", f"${s['stopped']:,.0f}", f"${s['leaked']:,.0f} got through", delta_color="off")
m[4].metric("Latency p95", f"{s['lat_p95']:.2f} ms", f"avg {s['lat_avg']:.2f} ms", delta_color="off")

tabs = st.tabs(["Findings", "Detection", "Threshold tuning", "Signals & reasons", "Network & customers", "Data"])

# ---- written findings ------------------------------------------------------------
with tabs[0]:
    for heading, body in an.narrative(res):
        st.subheader(heading)
        st.markdown(body)
    if res.examples:
        st.subheader("Example interventions")
        st.caption("The highest-risk held transfer for each scam type, with the message the customer would see.")
        for ex in res.examples:
            with st.expander(f"{ex['scenario'].replace('_', ' ').title()} · risk {ex['risk']} · {ex['tier']}"):
                st.caption("Message to customer")
                st.info(ex["user_message"].replace("$", "\\$"))
                st.caption("Analyst brief")
                st.markdown(ex["analyst_brief"].replace("$", "\\$"))

# ---- detection ---------------------------------------------------------------------
with tabs[1]:
    left, right = st.columns(2)
    with left:
        st.markdown("**Where transfers landed, by tier**")
        mix = an.tier_mix(df)
        st.altair_chart(
            alt.Chart(mix).mark_bar().encode(
                x=alt.X("tier:N", sort=TIER_ORDER, title=None),
                y=alt.Y("share:Q", axis=alt.Axis(format="%"), title="Share of group"),
                xOffset="group:N", color=alt.Color("group:N", scale=GROUP_COLORS, title=None),
                tooltip=["group", "tier", "count", alt.Tooltip("share:Q", format=".1%")]),
            width="stretch")
    with right:
        st.markdown("**Risk score distribution**")
        hist = an.histogram(df)
        bars = alt.Chart(hist).mark_bar(opacity=0.7).encode(
            x=alt.X("risk:Q", title="Risk score", bin=alt.Bin(step=5)),
            y=alt.Y("share:Q", stack=None, axis=alt.Axis(format="%"), title="Share of group"),
            color=alt.Color("group:N", scale=GROUP_COLORS, title=None))
        t = res.thresholds
        rules = alt.Chart(alt.Data(values=[{"x": t.nudge, "l": "nudge"}, {"x": t.hold, "l": "hold"},
                                           {"x": t.escalate, "l": "escalate"}])).mark_rule(
            strokeDash=[4, 4], color="#888").encode(x="x:Q")
        st.altair_chart(bars + rules, width="stretch")
    st.markdown("**Every transfer, in order**")
    sample = df[df["is_scam"] | (df.index % max(len(df) // 2500, 1) == 0)].assign(
        group=lambda d: d["is_scam"].map({True: "Scam", False: "Normal"}))
    st.altair_chart(
        alt.Chart(sample).mark_circle(size=28, opacity=0.6).encode(
            x=alt.X("id:Q", title="Transaction #"), y=alt.Y("risk:Q", title="Risk"),
            color=alt.Color("group:N", scale=GROUP_COLORS, title=None),
            tooltip=["id", "customer", "dest_id", "amount", "risk", "tier", "scenario"]).interactive(),
        width="stretch")
    st.caption("Normal traffic is thinned to about 2,500 points for the chart; every scam transfer is shown.")

    st.markdown("**By scam type**")
    sc = an.scenario_table(res)
    a, b = st.columns(2)
    with a:
        st.altair_chart(
            alt.Chart(sc).mark_bar(color="#e5484d").encode(
                y=alt.Y("label:N", title=None, sort="-x"),
                x=alt.X("catch_rate:Q", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1]), title="Campaigns caught"),
                tooltip=["label", "campaigns", "caught"]), width="stretch")
    with b:
        money = sc.melt(id_vars="label", value_vars=["stopped", "leaked"], var_name="outcome", value_name="amount")
        st.altair_chart(
            alt.Chart(money).mark_bar().encode(
                y=alt.Y("label:N", title=None), x=alt.X("amount:Q", title="Scam value ($)", axis=alt.Axis(format="~s")),
                color=alt.Color("outcome:N", scale=alt.Scale(domain=["stopped", "leaked"], range=["#30a46c", "#e5484d"]), title=None),
                tooltip=["label", "outcome", alt.Tooltip("amount:Q", format=",.0f")]), width="stretch")
    st.dataframe(sc.rename(columns={"label": "scam type", "median_first_block": "median first stop (transfer #)"}),
                 hide_index=True, width="stretch")

# ---- threshold tuning --------------------------------------------------------------------
with tabs[2]:
    sweep = an.threshold_sweep(df)
    rec = an.recommend_threshold(sweep)
    st.markdown("**What each hold threshold would have done on this run**")
    long = sweep.melt(id_vars="threshold", value_vars=["recall", "precision", "false_positive_rate"],
                      var_name="metric", value_name="value")
    lines = alt.Chart(long).mark_line().encode(
        x=alt.X("threshold:Q", title="Hold threshold"), y=alt.Y("value:Q", axis=alt.Axis(format="%"), title=None),
        color=alt.Color("metric:N", title=None), tooltip=["threshold", "metric", alt.Tooltip("value:Q", format=".1%")])
    marks = alt.Chart(alt.Data(values=[{"x": res.thresholds.hold, "l": "current"}, {"x": float(rec["threshold"]), "l": "suggested"}])
                      ).mark_rule(strokeDash=[4, 4], color="#888").encode(x="x:Q", tooltip=["l:N"])
    st.altair_chart(lines + marks, width="stretch")
    st.info(f"Current hold ≥ {res.thresholds.hold:.0f}. Lowest threshold that keeps false positives under 1%: "
            f"**{int(rec['threshold'])}** (catches {an.pct(rec['recall'])} of scam transfers, "
            f"{an.pct(rec['false_positive_rate'], 2)} false-positive rate).")
    st.caption("Reuses this run's risk scores. Re-run with a different threshold in the form to confirm.")
    st.dataframe(sweep[sweep["threshold"] % 5 == 0], hide_index=True, width="stretch")

# ---- signals and reasons -----------------------------------------------------------------------
with tabs[3]:
    left, right = st.columns(2)
    with left:
        st.markdown("**Average signal score**")
        st.altair_chart(
            alt.Chart(an.signal_means(df)).mark_bar().encode(
                x=alt.X("signal:N", title=None, sort=["behavior", "network", "context"]),
                y=alt.Y("mean:Q", title="Mean score (0-100)"), xOffset="group:N",
                color=alt.Color("group:N", scale=GROUP_COLORS, title=None),
                tooltip=["group", "signal", alt.Tooltip("mean:Q", format=".1f")]), width="stretch")
    with right:
        st.markdown("**Top reasons on held transfers**")
        rc = an.reason_counts(df)
        if rc.empty:
            st.write("No held transfers in this run.")
        else:
            st.altair_chart(
                alt.Chart(rc).mark_bar(color="#f5a524").encode(
                    y=alt.Y("reason:N", sort="-x", title=None, axis=alt.Axis(labelLimit=320)),
                    x=alt.X("held_events:Q", title="Held transfers"), tooltip=["reason", "held_events"]),
                width="stretch")

# ---- network and customers -------------------------------------------------------------------------
with tabs[4]:
    dest = an.top_destinations(res)
    cust = an.top_customers(res)
    st.markdown("**Receiving accounts with the most held transfers**")
    st.dataframe(dest, hide_index=True, width="stretch")
    if not dest.empty:
        pick = st.selectbox("Show the account's neighbourhood", dest["dest_id"], key="analysis_dest")
        render_neighborhood(res.graph, pick)
    st.markdown("**Customers with the most held transfers**")
    st.dataframe(cust, hide_index=True, width="stretch")
    st.caption("False positives = held transfers that were not part of a scam campaign.")

# ---- raw data ------------------------------------------------------------------------------------------
with tabs[5]:
    st.markdown("**Campaigns**")
    st.dataframe(camp, hide_index=True, width="stretch")
    st.markdown("**All scored transfers**")
    st.dataframe(df, hide_index=True, width="stretch")
    st.download_button("Download transfers (CSV)", df.to_csv(index=False).encode("utf-8"),
                       file_name=f"guardianz_analysis_seed{res.seed}.csv", mime="text/csv")
