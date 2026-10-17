"""Event stream: synthetic customers plus optional PaySim replay, with scam injection.

Owner: Streaming/Graph. `EventStream.next_event()` is the only call the pipeline needs.
"""
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass

from .graph import MuleGraph
from .schemas import Event


@dataclass
class UserProfile:
    user_id: str
    name: str
    is_senior: bool
    balance: float
    typical_amount: float


FIRST_NAMES = ["Tan", "Lim", "Wong", "Lee", "Chen", "Ng", "Ong", "Goh", "Teo", "Koh"]


def make_users(n: int, seed: int = 11) -> dict[str, UserProfile]:
    rng = random.Random(seed)
    users: dict[str, UserProfile] = {}
    for i in range(n):
        senior = rng.random() < 0.35
        balance = rng.lognormvariate(10.2 if senior else 9.4, 0.5)   # seniors hold bigger savings
        users[f"U{i:03d}"] = UserProfile(
            user_id=f"U{i:03d}",
            name=f"{'Mr.' if rng.random() < 0.5 else 'Mrs.'} {rng.choice(FIRST_NAMES)}",
            is_senior=senior,
            balance=round(balance, 2),
            typical_amount=round(rng.lognormvariate(4.8, 0.5), 2),
        )
    return users


class EventStream:
    def __init__(self, graph: MuleGraph, n_users: int = 80, seed: int = 3) -> None:
        self.rng = random.Random(seed)
        self.graph = graph
        self.users = make_users(n_users)
        self._next_id = 1
        self._clock = 1_700_000_000.0
        self._queue: deque[Event] = deque()   # scripted (scam) events take priority

    # -- public ------------------------------------------------------------
    def next_event(self) -> Event:
        if self._queue:
            ev = self._queue.popleft()
        else:
            ev = self._normal_event()
        ev.id, ev.ts = self._next_id, self._advance_clock()
        self._next_id += 1
        return ev

    def inject(self, events: list[Event]) -> None:
        self._queue.extend(events)

    def pick_senior(self) -> UserProfile:
        return self.rng.choice([u for u in self.users.values() if u.is_senior])

    # -- internals ---------------------------------------------------------
    def _advance_clock(self) -> float:
        self._clock += self.rng.expovariate(1 / 4.0)
        return self._clock

    def _normal_event(self) -> Event:
        u = self.rng.choice(list(self.users.values()))
        amount = max(1.0, self.rng.lognormvariate(0, 0.35) * u.typical_amount)
        new_payee = self.rng.random() < 0.03
        return Event(
            id=0, ts=0.0, user_id=u.user_id, dest_id=self.graph.random_normal(),
            amount=round(amount, 2), balance_before=u.balance,
            new_beneficiary=new_payee,
            device_change=self.rng.random() < 0.02,
            login_gap_s=max(60.0, self.rng.gauss(86_400, 20_000)),
            user_is_senior=u.is_senior,
        )


def load_paysim(path: str, limit: int | None = None):
    """TODO(Data): replay PaySim rows as Events.

    Map nameOrig->user_id, nameDest->dest_id, amount, oldbalanceOrg->balance_before,
    step->ts (hours), isFraud->label. Generate device_change / new_beneficiary /
    login_gap_s deterministically from a seed and say so in the pitch.
    """
    raise NotImplementedError("PaySim loader not implemented yet")
