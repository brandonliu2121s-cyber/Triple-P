"""Glue: stream -> score -> intervention -> guardian, plus dashboard counters.

Owner: UI/Integration. The UI only talks to `Pipeline`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .graph import MuleGraph
from .guardian import GuardianService
from .llm import InterventionGenerator
from .scams import SCENARIOS
from .schemas import Event, Intervention, ScoreResult, Tier
from .scoring import Scorer
from .simulator import EventStream

BLOCKING = (Tier.HOLD, Tier.ESCALATE)


@dataclass
class Processed:
    event: Event
    result: ScoreResult
    intervention: Intervention | None = None


@dataclass
class Stats:
    total: int = 0
    by_tier: dict[str, int] = field(default_factory=lambda: {t.value: 0 for t in Tier})
    held_amount: float = 0.0              # money stopped from leaving
    scam_events: int = 0
    scam_caught: int = 0                  # scam events that reached HOLD+
    normal_events: int = 0
    false_positives: int = 0              # normal events that reached HOLD+
    latencies_ms: list[float] = field(default_factory=list)

    @property
    def avg_latency_ms(self) -> float:
        return sum(self.latencies_ms[-500:]) / max(len(self.latencies_ms[-500:]), 1)

    @property
    def p95_latency_ms(self) -> float:
        xs = sorted(self.latencies_ms[-500:])
        return xs[int(0.95 * (len(xs) - 1))] if xs else 0.0


class Pipeline:
    def __init__(self, warmup: int = 600, use_llm: bool | None = None, worst_case: bool = False) -> None:
        self.graph = MuleGraph()
        if worst_case:                          # dev only: awkward names, ids and amounts
            from .fixtures import WORST_DESTS, worst_case_users
            self.stream = EventStream(self.graph, users=worst_case_users(), dests=WORST_DESTS)
        else:
            self.stream = EventStream(self.graph)
        self.scorer = Scorer(self.graph)
        self.llm = InterventionGenerator(use_llm)
        self.guardians = GuardianService()
        self.stats = Stats()
        self.history: list[Processed] = []
        self.cases: list[Processed] = []        # held/escalated -> analyst queue
        for _ in range(warmup):                 # build user baselines silently
            self.scorer.score(self.stream.next_event())

    # -- control -----------------------------------------------------------
    def inject_scam(self, key: str) -> None:
        user = self.stream.pick_senior()
        self.stream.inject(SCENARIOS[key].build(user, self.graph.random_mule()))

    # -- main loop ---------------------------------------------------------
    def step(self) -> Processed:
        ev = self.stream.next_event()
        res = self.scorer.score(ev)
        name = self.stream.users[ev.user_id].name
        intervention = None
        if res.tier is not Tier.ALLOW:
            intervention = self.llm.generate(ev, res, name)
        if res.tier in BLOCKING:
            self.guardians.maybe_alert(ev, res, name)
        p = Processed(ev, res, intervention)
        self._record(p)
        return p

    def _record(self, p: Processed) -> None:
        s, ev, res = self.stats, p.event, p.result
        s.total += 1
        s.by_tier[res.tier.value] += 1
        s.latencies_ms.append(res.latency_ms)
        blocked = res.tier in BLOCKING
        if blocked:
            s.held_amount += ev.amount
            self.cases.append(p)
        if ev.scenario:
            s.scam_events += 1
            s.scam_caught += blocked
        else:
            s.normal_events += 1
            s.false_positives += blocked
        self.history.append(p)
        del self.history[:-300]
