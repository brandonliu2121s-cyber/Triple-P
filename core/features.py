"""Per-user behaviour baselines and feature vectors.

Owner: Data/ML. The baseline is updated only with events that were allowed through,
so a scam in progress can't poison the user's "normal".
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

from .schemas import Event

FEATURE_NAMES = [
    "amount_z",           # amount vs user's typical amount (z-score, clipped)
    "balance_pct",        # share of balance being moved
    "new_beneficiary",
    "device_change",
    "login_gap_z",        # unusual login cadence
    "payee_novelty",      # 1.0 if dest never seen for this user
    "is_senior",
    "amount_log",         # log1p(amount); model feature
    "drains_account",     # 1.0 if >= 99% of the balance is moved; model feature
    "high_risk_type",     # 1.0 for TRANSFER / CASH_OUT (the only PaySim types that carry fraud)
]

# Subset the PaySim-trained XGBoost uses: only signals PaySim has AND a live Event can supply.
MODEL_FEATURES = ["amount_log", "balance_pct", "drains_account", "high_risk_type"]
HIGH_RISK_TYPES = {"TRANSFER", "CASH_OUT"}


@dataclass
class UserBaseline:
    n: int = 0
    mean_amount: float = 0.0
    m2_amount: float = 0.0              # Welford running variance
    mean_login_gap: float = 86_400.0
    payees: set[str] = field(default_factory=set)

    @property
    def std_amount(self) -> float:
        return (self.m2_amount / self.n) ** 0.5 if self.n > 1 else 0.0

    def update(self, ev: Event) -> None:
        self.n += 1
        delta = ev.amount - self.mean_amount
        self.mean_amount += delta / self.n
        self.m2_amount += delta * (ev.amount - self.mean_amount)
        self.mean_login_gap += (ev.login_gap_s - self.mean_login_gap) / self.n
        self.payees.add(ev.dest_id)


class BaselineStore:
    def __init__(self) -> None:
        self._users: dict[str, UserBaseline] = defaultdict(UserBaseline)

    def get(self, user_id: str) -> UserBaseline:
        return self._users[user_id]

    def update(self, ev: Event) -> None:
        self._users[ev.user_id].update(ev)


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def featurize(ev: Event, base: UserBaseline) -> list[float]:
    """Return a vector aligned with FEATURE_NAMES."""
    std = max(base.std_amount, 0.1 * max(base.mean_amount, 1.0))
    amount_z = _clip((ev.amount - base.mean_amount) / std, -5, 10) if base.n else 0.0
    login_z = _clip(abs(ev.login_gap_s - base.mean_login_gap) / max(base.mean_login_gap, 1.0), 0, 10)
    return [
        amount_z,
        ev.balance_pct,
        float(ev.new_beneficiary),
        float(ev.device_change),
        login_z,
        0.0 if ev.dest_id in base.payees else 1.0,
        float(ev.user_is_senior),
        math.log1p(ev.amount),
        float(ev.balance_pct >= 0.99),
        float(ev.type in HIGH_RISK_TYPES),
    ]


# --- repeated-payee signal -------------------------------------------------------------------------
REPEAT_WINDOW_S = 1800.0      # a scam call coaches a victim through several transfers in under an hour
ESCALATION_RATIO = 1.5        # this transfer is >= 1.5x the previous one to the same payee


class PayeeHistory:
    """Every transfer ATTEMPT per (customer, payee), regardless of tier.

    Separate from UserBaseline on purpose: the baseline only learns from trusted events, but a held first
    transfer must still count when the victim retries to the same payee.
    """

    def __init__(self) -> None:
        self._h: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)

    def signal(self, ev: Event) -> tuple[int, bool]:
        """(prior attempts to this payee, amount escalating?) -- only while the payee is still "new":
        i.e. the first attempt to it was less than REPEAT_WINDOW_S ago."""
        hist = self._h.get((ev.user_id, ev.dest_id), [])
        if not hist or ev.ts - hist[0][0] > REPEAT_WINDOW_S:
            return 0, False
        escalating = ev.amount >= ESCALATION_RATIO * hist[-1][1]
        return len(hist), escalating

    def record(self, ev: Event) -> None:
        self._h[(ev.user_id, ev.dest_id)].append((ev.ts, ev.amount))
