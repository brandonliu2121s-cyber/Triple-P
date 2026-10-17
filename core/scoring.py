"""Scoring engine: behaviour + network + context -> fused risk -> tier.

Owner: Data/ML (fusion, thresholds) with Graph (network signal).
Target: < 50 ms per event; `latency_ms` on every result feeds the dashboard.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from .features import BaselineStore, featurize
from .graph import MuleGraph
from .models import REASON_TEXT, BehaviorModel
from .schemas import Event, ScoreResult, Tier


@dataclass
class Thresholds:
    nudge: float = 30.0
    hold: float = 60.0
    escalate: float = 85.0

    def tier_for(self, risk: float) -> Tier:
        if risk >= self.escalate:
            return Tier.ESCALATE
        if risk >= self.hold:
            return Tier.HOLD
        if risk >= self.nudge:
            return Tier.NUDGE
        return Tier.ALLOW


# Fusion weights; the interaction bonus rewards several independent red flags at once.
W_BEHAVIOR, W_NETWORK, W_CONTEXT = 0.45, 0.30, 0.25
INTERACTION_BONUS = 0.35


class Scorer:
    def __init__(self, graph: MuleGraph, model: BehaviorModel | None = None,
                 thresholds: Thresholds | None = None) -> None:
        self.graph = graph
        self.model = model or BehaviorModel()
        self.thresholds = thresholds or Thresholds()
        self.baselines = BaselineStore()

    @staticmethod
    def _context_score(ev: Event) -> float:
        s = 0.0
        s += 40.0 if ev.device_change else 0.0
        s += 35.0 if ev.new_beneficiary else 0.0
        s += 25.0 if (ev.user_is_senior and ev.balance_pct > 0.5) else 0.0
        return s

    def score(self, ev: Event) -> ScoreResult:
        t0 = time.perf_counter()
        base = self.baselines.get(ev.user_id)
        x = featurize(ev, base)
        behavior, impacts = self.model.predict(x)
        network, _signals = self.graph.score(ev.dest_id)
        context = self._context_score(ev)

        fused = W_BEHAVIOR * behavior + W_NETWORK * network + W_CONTEXT * context
        fused += INTERACTION_BONUS * max(0.0, max(behavior, network, context) - fused)
        risk = min(fused, 100.0)

        reasons = [(REASON_TEXT[name], round(imp, 1)) for name, imp in impacts if imp >= 5.0]
        if network >= 50.0:
            reasons.append((REASON_TEXT["network"], round(network * 0.4, 1)))
        reasons.sort(key=lambda t: t[1], reverse=True)

        tier = self.thresholds.tier_for(risk)
        if tier in (Tier.ALLOW, Tier.NUDGE):      # only trusted events shape the baseline
            self.baselines.update(ev)

        return ScoreResult(
            id=ev.id, risk=round(risk, 1), behavior_score=round(behavior, 1),
            network_score=round(network, 1), context_score=round(context, 1),
            tier=tier, reasons=reasons[:4],
            latency_ms=round((time.perf_counter() - t0) * 1000, 2),
        )
