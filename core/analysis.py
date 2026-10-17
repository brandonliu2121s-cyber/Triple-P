"""Batch (non-live) analysis: run a seeded simulation end to end and return tidy tables.

Owner: UI/Integration with Data/ML. The Analysis page calls `run_analysis` once per button press and
renders the tables; nothing here touches the live dashboard's pipeline.

Scam campaigns (the same scripted scenarios as the live "Scam simulator") are scheduled at random points in
the stream and tagged, so every event has a ground-truth label and the run can report real detection numbers.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .graph import MuleGraph
from .pipeline import BLOCKING, Pipeline
from .scams import SCENARIOS
from .scoring import Thresholds


@dataclass
class AnalysisResult:
    events: pd.DataFrame                  # one row per scored event
    campaigns: pd.DataFrame               # one row per injected scam campaign
    graph: MuleGraph
    thresholds: Thresholds
    n_events: int
    seed: int
    worst_case: bool
    elapsed_s: float
    model_trained: bool
    guardian_alerts: int
    examples: list[dict] = field(default_factory=list)   # highest-risk held scam, one per scenario


def run_analysis(n_events: int = 5000, campaigns_per_type: int = 8, seed: int = 1, worst_case: bool = False,
                 thresholds: Thresholds | None = None) -> AnalysisResult:
    t0 = time.perf_counter()
    p = Pipeline(use_llm=False, worst_case=worst_case)
    if thresholds:
        p.scorer.thresholds = thresholds
    p.stream.rng.seed(seed)
    p.graph.rng.seed(seed)

    rng = random.Random(seed)
    plan = [k for k in SCENARIOS for _ in range(campaigns_per_type)]
    rng.shuffle(plan)
    lo, hi = int(n_events * 0.05), int(n_events * 0.95)
    slots = sorted(rng.sample(range(lo, hi), min(len(plan), hi - lo)))
    schedule = dict(zip(slots, plan))

    # id(event object) -> campaign index. The queue keeps each scam event alive until it is recorded, and the entry
    # is popped on record, because CPython reuses ids of freed objects and a stale entry would match a normal event.
    campaign_of: dict[int, int] = {}
    meta: list[tuple[str, str]] = []      # per campaign: (scenario, user_id)
    rows: list[dict] = []
    best: dict[str, tuple[float, dict]] = {}

    def record(x) -> None:
        ev, r = x.event, x.result
        rows.append({
            "id": ev.id, "user_id": ev.user_id, "customer": p.stream.users[ev.user_id].name,
            "dest_id": ev.dest_id, "amount": ev.amount, "balance_pct": ev.balance_pct, "type": ev.type,
            "senior": ev.user_is_senior, "new_beneficiary": ev.new_beneficiary, "device_change": ev.device_change,
            "scenario": ev.scenario or "", "campaign": campaign_of.pop(id(ev), -1), "is_scam": bool(ev.scenario),
            "risk": r.risk, "behavior": r.behavior_score, "network": r.network_score, "context": r.context_score,
            "tier": r.tier.value, "blocked": r.tier in BLOCKING, "latency_ms": r.latency_ms,
            "top_reason": r.reasons[0][0] if r.reasons else "",
            "reasons": " | ".join(t for t, _ in r.reasons),
        })
        if ev.scenario and r.tier in BLOCKING and x.intervention and r.risk > best.get(ev.scenario, (-1, {}))[0]:
            best[ev.scenario] = (r.risk, {
                "scenario": ev.scenario, "event_id": ev.id, "risk": r.risk, "tier": r.tier.value,
                "amount": ev.amount, "customer": rows[-1]["customer"],
                "user_message": x.intervention.user_message, "analyst_brief": x.intervention.analyst_brief,
            })

    for i in range(n_events):
        key = schedule.get(i)
        if key:
            user = p.stream.pick_senior()
            evs = SCENARIOS[key].build(user, p.graph.random_mule())
            for e in evs:
                campaign_of[id(e)] = len(meta)
            meta.append((key, user.user_id))
            p.stream.inject(evs)
        record(p.step())
    while p.stream.pending:               # let a campaign that started near the end finish
        record(p.step())

    df = pd.DataFrame(rows)
    return AnalysisResult(
        events=df, campaigns=_campaigns(df, meta, p), graph=p.graph, thresholds=p.scorer.thresholds,
        n_events=len(df), seed=seed, worst_case=worst_case, elapsed_s=time.perf_counter() - t0,
        model_trained=p.scorer.model.trained, guardian_alerts=len(p.guardians.alerts),
        examples=[v[1] for v in sorted(best.values(), key=lambda t: -t[0])],
    )


def _campaigns(df: pd.DataFrame, meta: list[tuple[str, str]], p: Pipeline) -> pd.DataFrame:
    out = []
    for cid, (scenario, user_id) in enumerate(meta):
        c = df[df["campaign"] == cid].sort_values("id")
        if c.empty:
            continue
        blocked = c["blocked"].to_numpy()
        first = int(np.argmax(blocked)) + 1 if blocked.any() else np.nan     # 1-based transfer number
        out.append({
            "campaign": cid, "scenario": scenario, "label": SCENARIOS[scenario].label,
            "customer": p.stream.users[user_id].name, "user_id": user_id, "transfers": len(c),
            "caught": bool(blocked.any()), "first_block": first, "max_risk": c["risk"].max(),
            "total": c["amount"].sum(), "stopped": c.loc[c["blocked"], "amount"].sum(),
            "leaked": c.loc[~c["blocked"], "amount"].sum(),
        })
    return pd.DataFrame(out)


# -- derived tables -----------------------------------------------------------------------------------
def summarize(res: AnalysisResult) -> dict:
    df, camp = res.events, res.campaigns
    scam, norm = df[df["is_scam"]], df[~df["is_scam"]]
    tp, fp = int(scam["blocked"].sum()), int(norm["blocked"].sum())
    stopped, leaked = float(camp["stopped"].sum()), float(camp["leaked"].sum())
    return {
        "events": len(df), "scam_events": len(scam), "normal_events": len(norm),
        "tp": tp, "fp": fp, "fn": len(scam) - tp,
        "event_recall": tp / len(scam) if len(scam) else np.nan,
        "precision": tp / (tp + fp) if tp + fp else np.nan,
        "fpr": fp / len(norm) if len(norm) else np.nan,
        "campaigns": len(camp), "campaigns_caught": int(camp["caught"].sum()),
        "catch_rate": float(camp["caught"].mean()) if len(camp) else np.nan,
        "median_first_block": float(camp["first_block"].median()) if camp["caught"].any() else np.nan,
        "stopped": stopped, "leaked": leaked,
        "leak_share": leaked / (stopped + leaked) if stopped + leaked else np.nan,
        "nudges": int((df["tier"] == "nudge").sum()),
        "lat_avg": float(df["latency_ms"].mean()), "lat_p95": float(df["latency_ms"].quantile(0.95)),
        "lat_max": float(df["latency_ms"].max()),
    }


def threshold_sweep(df: pd.DataFrame, lo: int = 10, hi: int = 99) -> pd.DataFrame:
    """Recall / precision / false-positive rate if the hold threshold were t. Uses the risk scores from this
    run, so it ignores the small effect a different threshold would have on user baselines."""
    risk, scam = df["risk"].to_numpy(), df["is_scam"].to_numpy()
    n_scam, n_norm = int(scam.sum()), int((~scam).sum())
    rows = []
    for t in range(lo, hi + 1):
        b = risk >= t
        tp, fp = int((b & scam).sum()), int((b & ~scam).sum())
        rows.append({"threshold": t, "recall": tp / n_scam if n_scam else np.nan,
                     "precision": tp / (tp + fp) if tp + fp else np.nan,
                     "false_positive_rate": fp / n_norm if n_norm else np.nan, "flagged": int(b.sum())})
    return pd.DataFrame(rows)


def recommend_threshold(sweep: pd.DataFrame, max_fpr: float = 0.01) -> pd.Series:
    """Lowest threshold (most recall) that keeps false positives at or under `max_fpr`."""
    ok = sweep[sweep["false_positive_rate"] <= max_fpr]
    return ok.iloc[0] if len(ok) else sweep.iloc[-1]


def histogram(df: pd.DataFrame, step: int = 5) -> pd.DataFrame:
    """Share of each population per risk bucket (normalised, since normal traffic dwarfs scams)."""
    bins = np.arange(0, 100 + step, step)
    out = []
    for name, part in (("Normal", df[~df["is_scam"]]), ("Scam", df[df["is_scam"]])):
        counts, _ = np.histogram(part["risk"], bins=bins)
        total = max(counts.sum(), 1)
        out += [{"risk": int(b), "share": c / total, "group": name} for b, c in zip(bins[:-1], counts)]
    return pd.DataFrame(out)


def tier_mix(df: pd.DataFrame) -> pd.DataFrame:
    order = ["allow", "nudge", "hold", "escalate"]
    out = []
    for name, part in (("Normal", df[~df["is_scam"]]), ("Scam", df[df["is_scam"]])):
        for tier in order:
            out.append({"tier": tier, "share": float((part["tier"] == tier).mean()) if len(part) else 0.0,
                        "count": int((part["tier"] == tier).sum()), "group": name})
    return pd.DataFrame(out)


def signal_means(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for name, part in (("Normal", df[~df["is_scam"]]), ("Scam", df[df["is_scam"]])):
        for sig in ("behavior", "network", "context"):
            out.append({"signal": sig, "mean": float(part[sig].mean()) if len(part) else 0.0, "group": name})
    return pd.DataFrame(out)


def reason_counts(df: pd.DataFrame, top: int = 8) -> pd.DataFrame:
    held = df[df["blocked"] & (df["reasons"] != "")]
    s = held["reasons"].str.split(" \\| ").explode().value_counts().head(top)
    return s.rename_axis("reason").reset_index(name="held_events")


def scenario_table(res: AnalysisResult) -> pd.DataFrame:
    c = res.campaigns
    g = c.groupby("label").agg(campaigns=("campaign", "count"), caught=("caught", "sum"),
                               median_first_block=("first_block", "median"),
                               stopped=("stopped", "sum"), leaked=("leaked", "sum")).reset_index()
    g["catch_rate"] = g["caught"] / g["campaigns"]
    return g


def top_destinations(res: AnalysisResult, n: int = 10) -> pd.DataFrame:
    held = res.events[res.events["blocked"]]
    g = held.groupby("dest_id").agg(held_events=("id", "count"), amount=("amount", "sum"),
                                    network=("network", "mean")).reset_index()
    g["known_mule"] = g["dest_id"].isin(res.graph.mule_ids)
    return g.sort_values(["held_events", "amount"], ascending=False).head(n).reset_index(drop=True)


def top_customers(res: AnalysisResult, n: int = 10) -> pd.DataFrame:
    held = res.events[res.events["blocked"]]
    g = held.groupby(["user_id", "customer"]).agg(held_events=("id", "count"), amount=("amount", "sum"),
                                                  scam_events=("is_scam", "sum")).reset_index()
    g["false_positives"] = g["held_events"] - g["scam_events"]
    return g.sort_values(["held_events", "amount"], ascending=False).head(n).reset_index(drop=True)


# -- written findings ----------------------------------------------------------------------------------
def usd(x: float) -> str:
    return f"\\${x:,.0f}"                  # escaped: a bare $ can start LaTeX in st.markdown


def pct(x: float, digits: int = 0) -> str:
    return "n/a" if x != x else f"{100 * x:.{digits}f}%"


def narrative(res: AnalysisResult) -> list[tuple[str, str]]:
    """(heading, markdown) findings written from the numbers, so they always match the charts."""
    s, df, camp = summarize(res), res.events, res.campaigns
    sweep = threshold_sweep(df)
    rec = recommend_threshold(sweep)
    cur = sweep[sweep["threshold"] == int(res.thresholds.hold)]
    cur = cur.iloc[0] if len(cur) else rec
    by_sc = scenario_table(res)
    sig = signal_means(df).pivot(index="signal", columns="group", values="mean")
    gap = (sig["Scam"] - sig["Normal"]).sort_values(ascending=False)
    missed = by_sc[by_sc["catch_rate"] < 1]
    out: list[tuple[str, str]] = []

    out.append(("Coverage", (
        f"- **{s['campaigns_caught']} of {s['campaigns']} scam campaigns ({pct(s['catch_rate'])})** triggered a hold or escalation at least once.\n"
        f"- At the event level the system flagged {s['tp']} of {s['scam_events']} scam transfers (recall {pct(s['event_recall'])}).\n"
        + ("- Every scenario was caught at least once.\n" if missed.empty else
           "- Campaigns that slipped through entirely: " + ", ".join(
               f"**{r.label}** ({int(r.campaigns - r.caught)} of {int(r.campaigns)})" for r in missed.itertuples()) + ".\n"))))

    out.append(("Timing and money", (
        f"- The median caught campaign was first stopped on **transfer {s['median_first_block']:.0f}** of its sequence.\n"
        f"- Money stopped: **{usd(s['stopped'])}**. Money that went through before or between holds: **{usd(s['leaked'])}** "
        f"({pct(s['leak_share'])} of all scam value).\n"
        + (f"- Weakest scenario by leakage: **{by_sc.sort_values('leaked', ascending=False).iloc[0]['label']}** "
           f"({usd(by_sc['leaked'].max())} leaked).\n" if len(by_sc) else ""))))

    fp_every = f"about one per {s['normal_events'] // max(s['fp'], 1):,} normal transfers" if s["fp"] else "none in this run"
    out.append(("Noise", (
        f"- **{s['fp']} false positives** out of {s['normal_events']:,} normal transfers ({pct(s['fpr'], 2)}), {fp_every}.\n"
        f"- Precision on held transfers: {pct(s['precision'])}. A further {s['nudges']:,} transfers got a soft nudge only.\n"
        f"- Guardian alerts sent for held seniors: {res.guardian_alerts}.")))

    out.append(("What separates scams from normal traffic", (
        "- Average signal scores, scam vs normal: " + "; ".join(
            f"**{k}** {sig.loc[k, 'Scam']:.0f} vs {sig.loc[k, 'Normal']:.0f}" for k in gap.index) + ".\n"
        f"- The widest gap is **{gap.index[0]}** (+{gap.iloc[0]:.0f} points), the narrowest is **{gap.index[-1]}** (+{gap.iloc[-1]:.0f}).\n"
        + ("- Behaviour is scored by the PaySim-trained XGBoost model.\n" if res.model_trained else
           "- Behaviour is scored by the **rule-based stand-in**, not a trained model; train on PaySim for realistic numbers.\n"))))

    better = ("a lower threshold would catch more with the same noise"
              if rec["threshold"] < cur["threshold"] else
              "a higher threshold would cut noise at little cost" if rec["threshold"] > cur["threshold"] else
              "the current threshold is already the best trade-off under a 1% false-positive cap")
    out.append(("Threshold check", (
        f"- At hold ≥ **{int(cur['threshold'])}**: recall {pct(cur['recall'])}, false-positive rate {pct(cur['false_positive_rate'], 2)}.\n"
        f"- Best threshold keeping false positives ≤ 1%: **{int(rec['threshold'])}** (recall {pct(rec['recall'])}, "
        f"false-positive rate {pct(rec['false_positive_rate'], 2)}), so {better}.")))

    out.append(("Speed", (
        f"- Scoring latency: average {s['lat_avg']:.2f} ms, p95 {s['lat_p95']:.2f} ms, worst {s['lat_max']:.2f} ms "
        f"over {s['events']:,} events (target is under 50 ms).")))

    out.append(("Caveats", (
        f"- Synthetic data: {s['normal_events']:,} generated normal transfers and {s['campaigns']} scripted scam campaigns, seed {res.seed}. "
        "Treat the numbers as a regression test and a tuning aid, not as real-world performance.\n"
        "- The threshold sweep reuses this run's scores and ignores how a different threshold would change user baselines.")))
    return out
