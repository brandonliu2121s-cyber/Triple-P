"""Generative intervention: cooling-off prompt for the customer + analyst brief for the fraud desk.

Owner: GenAI/Alerts. Calls a local Ollama model (data never leaves the box -> the LinuxONE
story) with a short timeout; ANY failure falls back to templates so the demo never hangs.

Env: GUARDIANZ_LLM=1 enable Ollama (default off), GUARDIANZ_MODEL (default llama3.2:3b),
     OLLAMA_URL (default http://localhost:11434).
"""
from __future__ import annotations

import os
import time

import requests

from .schemas import Event, Intervention, ScoreResult, Tier
from .textutil import isolate, md_escape

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
MODEL = os.getenv("GUARDIANZ_MODEL", "llama3.2:3b")
TIMEOUT_S = 3.0
CACHE_MAX = 256

USER_PROMPT = """You are a calm, kind bank assistant. A customer's transfer was paused because it looks like a scam.
Write 2-3 short, plain-language sentences (no jargon, no blame) asking them to take a moment, \
mentioning the key warning sign, and reminding them banks never ask you to move money to a 'safe account'.
Customer: {name}. Amount: ${amount:,.0f}. Warning signs: {reasons}."""

BRIEF_PROMPT = """You are a fraud analyst copilot. Write a concise case brief as 4-5 bullet points: \
what happened, evidence, risk level, recommended action.
Transaction ${amount:,.0f} from {user} to {dest}. Risk {risk}/100 ({tier}). Scenario hint: {scenario}.
Evidence: {reasons}. Network signal: {network}/100."""


def _ask_ollama(prompt: str) -> str:
    r = requests.post(f"{OLLAMA_URL}/api/generate", timeout=TIMEOUT_S,
                      json={"model": MODEL, "prompt": prompt, "stream": False,
                            "options": {"temperature": 0.3, "num_predict": 180}})
    r.raise_for_status()
    return r.json()["response"].strip()


def _template_user(name: str, ev: Event, res: ScoreResult) -> str:
    top = res.reasons[0][0].lower() if res.reasons else "this transfer looks unusual"
    name = isolate(name)        # the UI escapes markdown at render time; this keeps an RTL name in place
    if res.tier is Tier.NUDGE:
        return (f"Hi {name}, just checking: you're about to send ${ev.amount:,.0f}. "
                f"We noticed that {top}. If anyone is rushing you, it's safe to pause.")
    return (f"{name}, we've paused this ${ev.amount:,.0f} transfer for your protection. "
            f"We noticed that {top}. Banks and police never ask you to move money to a 'safe account'. "
            f"Please take a few minutes, and talk to someone you trust before continuing.")


def _template_brief(ev: Event, res: ScoreResult) -> str:
    # The brief is markdown, so ids are escaped here; the UI escapes only `$` when it renders it.
    bullets = [f"- **Transaction:** ${ev.amount:,.0f} {md_escape(ev.user_id)} -> {md_escape(ev.dest_id)}",
               f"- **Risk:** {res.risk}/100 ({res.tier.value.upper()}); network {res.network_score}, behaviour {res.behavior_score}"]
    bullets += [f"- **Evidence:** {r}" for r, _ in res.reasons[:3]]
    bullets.append("- **Recommended:** call the customer on a verified number before releasing funds"
                   + ("; freeze receiving account pending mule review" if res.network_score >= 50 else ""))
    return "\n".join(bullets)


class InterventionGenerator:
    def __init__(self, use_llm: bool | None = None) -> None:
        self.use_llm = use_llm if use_llm is not None else os.getenv("GUARDIANZ_LLM") == "1"
        self._cache: dict[tuple, str] = {}      # bounded in _cached

    def generate(self, ev: Event, res: ScoreResult, customer_name: str) -> Intervention:
        t0 = time.perf_counter()
        reasons = "; ".join(r for r, _ in res.reasons) or "unusual activity"
        user_msg = brief = None
        source = "template"
        if self.use_llm:
            try:
                user_msg = self._cached(("u", res.tier, reasons), USER_PROMPT.format(
                    name=customer_name, amount=ev.amount, reasons=reasons))
                if res.tier in (Tier.HOLD, Tier.ESCALATE):
                    brief = self._cached(("b", res.id), BRIEF_PROMPT.format(
                        amount=ev.amount, user=ev.user_id, dest=ev.dest_id, risk=res.risk,
                        tier=res.tier.value, scenario=ev.scenario or "unknown", reasons=reasons,
                        network=res.network_score))
                source = "llm"
            except Exception:       # Ollama down / slow -> never block the demo
                user_msg = brief = None
        return Intervention(
            event_id=ev.id,
            user_message=user_msg or _template_user(customer_name, ev, res),
            analyst_brief=brief or _template_brief(ev, res),
            source=source, llm_latency_ms=round((time.perf_counter() - t0) * 1000, 1),
        )

    def _cached(self, key: tuple, prompt: str) -> str:
        if key not in self._cache:
            if len(self._cache) >= CACHE_MAX:
                self._cache.clear()
            self._cache[key] = _ask_ollama(prompt)
        return self._cache[key]
