"""Scripted scam scenarios for the demo ("Scam simulator" buttons).

Owner: Streaming/Graph. Each scenario returns a short sequence of events for one senior
customer: warning signals first, then escalating transfers to a mule account. Events are
tagged with `scenario` so the dashboard can measure detection.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .schemas import Event
from .simulator import UserProfile


@dataclass(frozen=True)
class Scenario:
    key: str
    label: str
    description: str
    build: Callable[[UserProfile, str], list[Event]]


def _ev(user: UserProfile, dest: str, amount: float, scenario: str, **kw) -> Event:
    return Event(id=0, ts=0.0, user_id=user.user_id, dest_id=dest, amount=round(amount, 2),
                 balance_before=user.balance, user_is_senior=user.is_senior, scenario=scenario, **kw)


def impersonation(user: UserProfile, mule: str) -> list[Event]:
    """'Bank officer' phones, claims account is compromised, asks to move money to a 'safe account'."""
    k = "impersonation"
    return [
        _ev(user, mule, user.balance * 0.10, k, new_beneficiary=True, device_change=True, login_gap_s=900),
        _ev(user, mule, user.balance * 0.35, k, login_gap_s=600),
        _ev(user, mule, user.balance * 0.80, k, login_gap_s=420),
    ]


def romance(user: UserProfile, mule: str) -> list[Event]:
    k = "romance"
    return [
        _ev(user, mule, user.typical_amount * 6, k, new_beneficiary=True),
        _ev(user, mule, user.balance * 0.20, k),
        _ev(user, mule, user.balance * 0.45, k, login_gap_s=1_800),
    ]


def investment(user: UserProfile, mule: str) -> list[Event]:
    k = "investment"
    return [
        _ev(user, mule, user.balance * 0.15, k, new_beneficiary=True, device_change=True),
        _ev(user, mule, user.balance * 0.40, k, login_gap_s=3_000),
    ]


def tech_support(user: UserProfile, mule: str) -> list[Event]:
    k = "tech_support"
    return [
        _ev(user, mule, user.balance * 0.25, k, new_beneficiary=True, device_change=True, login_gap_s=300),
        _ev(user, mule, user.balance * 0.60, k, device_change=True, login_gap_s=240),
    ]


SCENARIOS: dict[str, Scenario] = {s.key: s for s in [
    Scenario("impersonation", "Bank-officer impersonation", "Fake bank call -> 'safe account' transfers", impersonation),
    Scenario("romance", "Romance scam", "Online partner asks for escalating help", romance),
    Scenario("investment", "Investment scam", "Guaranteed-return scheme, big transfers", investment),
    Scenario("tech_support", "Tech-support scam", "Remote-access 'refund' scam", tech_support),
]}
