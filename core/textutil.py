"""Display helpers shared by the message templates and the UI.

Names, ids and amounts come from data, so they must never be able to change how a message renders.
"""
from __future__ import annotations

import re

_MD_SPECIAL = re.compile(r"([\\`*_{}\[\]<>~|$#&])")


def md_escape(text: object) -> str:
    """Escape a plain-text value for st.markdown / st.info / st.warning (so `$Lee$` is not math, `**` is not bold)."""
    return _MD_SPECIAL.sub(r"\\\1", str(text))


def md_dollars(text: str) -> str:
    """Escape only `$` in text that is already markdown (the analyst brief), leaving its bullets and bold alone."""
    return text.replace("$", "\\$")


def isolate(text: str) -> str:
    """Wrap in Unicode bidi isolates so an RTL name can't reorder the English sentence around it."""
    return f"⁨{text}⁩"


def compact_usd(x: float) -> str:
    """$12,345 below a million, then $12.3M / $1.2B, so big totals fit a metric card."""
    ax = abs(x)
    sign = "-" if x < 0 else ""
    for limit, suffix in ((1e9, "B"), (1e6, "M")):
        if ax >= limit:
            return f"{sign}${ax / limit:,.1f}{suffix}"
    return f"{sign}${ax:,.0f}"


def short_id(value: str, width: int = 18) -> str:
    """Middle-truncate a long id: the start and the end are the parts that tell two ids apart."""
    if len(value) <= width:
        return value
    keep = width - 1
    head = keep // 2 + keep % 2
    return f"{value[:head]}…{value[len(value) - (keep - head):]}"


def plural(n: int, noun: str) -> str:
    return f"{n:,} {noun}" if n == 1 else f"{n:,} {noun}s"
