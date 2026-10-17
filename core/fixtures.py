"""Worst-case data for the dashboard (dev only; see the "Dev: data" switch in app.py).

Every value is a realistic name/identifier shape or a plausible extreme, mixed across users the way
real data is. It enters through the same boundary as the demo data: EventStream(users=..., dests=...).
"""
from __future__ import annotations

from dataclasses import replace

from .simulator import UserProfile, make_users


def _u(uid: str, name: str, senior: bool, balance: float, typical: float) -> UserProfile:
    return UserProfile(user_id=uid, name=name, is_senior=senior, balance=balance, typical_amount=typical)


def worst_case_users() -> dict[str, UserProfile]:
    users = [
        _u("U001", "Mrs. Aleksandra Wiśniewska-Kowalczyk", True, 12_345_678.90, 250_000.0),   # long name, big money
        _u("U002", "Mr. Konstantin Oberhauser-Wettstein", True, 0.0, 40.0),                    # emptied account
        _u("U003", "J", True, 3.5, 1.0),                                                       # one letter, tiny balance
        _u("9f8e7d6c-5b4a-4c3d-8e2f-1a0b9c8d7e6f", "Mr. Christopher Alexander Montgomery III", True, 48_000.0, 95.0),
        _u("U005", "🦊 Fox", True, 21_500.0, 60.0),
        _u("U006", "Mrs. $Lee$ **VIP** <b>& co</b>", True, 31_000.0, 70.0),                    # markdown / LaTeX / HTML chars
        _u("U007", "نور الهدى عبد الرحمن", True, 18_250.0, 55.0),                              # RTL
        _u("U008", "Mrs. Đặng Thị Ngọc Hân", True, 27_900.0, 80.0),                           # stacked diacritics
        _u("U009", "王秀英", True, 15_000.0, 45.0),                                             # CJK, no spaces
        _u("U010", "Mr. Tan", True, 22_000.0, 65.0),                                           # two identical names
        _u("U011", "Mr. Tan", False, 9_000.0, 50.0),
        _u("U012", "  Sam   Lee ", False, 7_500.0, 40.0),                                      # stray whitespace
        _u("U013", "Mrs. Seán O'Brien-Ó Súilleabháin", True, 64_000.0, 110.0),
    ]
    out = {u.user_id: u for u in users}
    for i, u in enumerate(make_users(10, seed=5).values()):     # ordinary traffic around the awkward users
        out[f"C{i:03d}"] = replace(u, user_id=f"C{i:03d}")      # own id range so they can't overwrite the above
    return out


# Payee shapes: IBANs, an email handle, a UUID, and the shortest realistic address.
WORST_DESTS = [
    "GB82WEST12345698765432",
    "DE89370400440532013000",
    "bartholomew.fitzgerald@northwind-industries-holdings.example.com",
    "9f8e7d6c-5b4a-4c3d-8e2f-1a0b9c8d7e6f",
    "a@b.co",
]
