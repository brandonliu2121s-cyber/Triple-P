"""Trusted-guardian alerts.

Owner: GenAI/Alerts. Simulates notifying a family member when a senior customer's
transfer is held. Swap `_deliver` for a real channel (Telegram bot, email) if time allows.
"""
from __future__ import annotations

from .schemas import Event, GuardianAlert, ScoreResult, Tier

_GUARDIANS = ["Daughter", "Son", "Grandson", "Niece", "Neighbour"]


class GuardianService:
    def __init__(self) -> None:
        self.alerts: list[GuardianAlert] = []

    @staticmethod
    def guardian_for(user_id: str) -> str:
        return _GUARDIANS[sum(map(ord, user_id)) % len(_GUARDIANS)]

    def maybe_alert(self, ev: Event, res: ScoreResult, customer_name: str) -> GuardianAlert | None:
        if not ev.user_is_senior or res.tier not in (Tier.HOLD, Tier.ESCALATE):
            return None
        guardian = self.guardian_for(ev.user_id)
        top = res.reasons[0][0] if res.reasons else "unusual activity"
        alert = GuardianAlert(
            event_id=ev.id, ts=ev.ts, user_id=ev.user_id, guardian_name=guardian,
            amount=ev.amount, tier=res.tier,
            message=(f"Heads up: a ${ev.amount:,.0f} transfer from {customer_name}'s account was paused "
                     f"for safety ({top.lower()}). Please check in with them before it goes through."),
        )
        self._deliver(alert)
        self.alerts.append(alert)
        return alert

    def _deliver(self, alert: GuardianAlert) -> None:
        """TODO(GenAI/Alerts): push to a real channel; the in-app log is the demo fallback."""
