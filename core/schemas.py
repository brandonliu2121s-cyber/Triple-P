"""Shared contracts. Everyone codes against these; change them only with the whole team."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Optional


class Tier(str, Enum):
    ALLOW = "allow"
    NUDGE = "nudge"
    HOLD = "hold"
    ESCALATE = "escalate"


@dataclass
class Event:
    id: int
    ts: float                      # simulated epoch seconds
    user_id: str
    dest_id: str
    amount: float
    type: str = "TRANSFER"         # PaySim: TRANSFER / CASH_OUT / PAYMENT ...
    balance_before: float = 0.0
    new_beneficiary: bool = False
    device_change: bool = False
    login_gap_s: float = 86_400.0  # seconds since previous login
    user_is_senior: bool = False
    scenario: Optional[str] = None  # set by scam injector; None for normal traffic

    @property
    def balance_pct(self) -> float:
        return min(self.amount / self.balance_before, 1.0) if self.balance_before > 0 else 0.0

    def to_dict(self) -> dict:
        return asdict(self) | {"balance_pct": self.balance_pct}


@dataclass
class ScoreResult:
    id: int
    risk: float                    # fused 0-100
    behavior_score: float          # 0-100
    network_score: float           # 0-100
    context_score: float           # 0-100
    tier: Tier
    reasons: list[tuple[str, float]] = field(default_factory=list)  # (human reason, impact), sorted desc
    latency_ms: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["tier"] = self.tier.value
        return d


@dataclass
class Intervention:
    event_id: int
    user_message: str              # cooling-off prompt shown to the customer
    analyst_brief: str             # bullet brief for the fraud desk
    source: str = "template"       # "llm" | "template"
    llm_latency_ms: float = 0.0


@dataclass
class GuardianAlert:
    event_id: int
    ts: float
    user_id: str
    guardian_name: str
    amount: float
    tier: Tier
    message: str
