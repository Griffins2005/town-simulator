"""
faith.py -- Religion as a civic factor, not flavor text.

A resident's faith is a stable identity (who they worship with) plus
piety (how much that identity moves them). Same-faith neighbors are
easier to lobby, more likely to vote together, and gather at a home
place (chapel, hall, or farm). The engine never invents doctrine --
Deciders only see the flat facts on Perception.
"""

from __future__ import annotations

FAITHS = {
    "vale_covenant": "Vale Covenant",
    "hall_creed": "Hall Creed",
    "old_ways": "Old Ways",
    "unaffiliated": "Unaffiliated",
}

# Where a devout resident is pulled when piety is high.
FAITH_HOME = {
    "vale_covenant": "chapel",
    "hall_creed": "town_hall",
    "old_ways": "farm",
    "unaffiliated": None,
}

_ORDER = ("vale_covenant", "hall_creed", "old_ways", "unaffiliated")


def faith_name(faith_id: str | None) -> str:
    return FAITHS.get(faith_id or "unaffiliated", "Unaffiliated")


def faith_home(faith_id: str | None) -> str | None:
    return FAITH_HOME.get(faith_id or "unaffiliated")


def assign_faith(index: int, rng) -> tuple[str, float]:
    """Seeded bloc assignment so religions form visible congregations."""
    faith_id = _ORDER[index % len(_ORDER)]
    if faith_id == "unaffiliated":
        return faith_id, round(rng.uniform(0.08, 0.35), 3)
    return faith_id, round(rng.uniform(0.35, 0.92), 3)


def same_faith(a_faith: str | None, b_faith: str | None) -> bool:
    if not a_faith or not b_faith:
        return False
    if a_faith == "unaffiliated" or b_faith == "unaffiliated":
        return False
    return a_faith == b_faith
