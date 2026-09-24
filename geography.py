"""
geography.py -- Streets, one-hop travel, and public-space weights.

The town is a valley, not a complete graph. Agents name a destination;
actions.py walks the next open street. Flood, a closed bridge, or a
missing road can make a place unreachable. Public spaces change what
is likely once people are standing there -- a tavern accelerates
gossip, a square raises politics, a chapel thickens same-faith ties.
"""

from __future__ import annotations

from collections import deque

from world import World

# Undirected streets. `kind` is what flood / bridge-closed actually cut.
STREETS: tuple[tuple[str, str, str], ...] = (
    ("farm", "market", "bridge"),
    ("farm", "homes", "country"),
    ("homes", "market", "neighborhood"),
    ("market", "tavern", "main"),
    ("market", "bank", "main"),
    ("bank", "town_hall", "main"),
    ("tavern", "town_hall", "neighborhood"),
    ("town_hall", "chapel", "pedestrian"),
    ("chapel", "park", "greenway"),
    ("park", "clinic", "greenway"),
    ("park", "tavern", "greenway"),
    ("clinic", "bank", "neighborhood"),
    ("clinic", "workshop", "service"),
    ("workshop", "tavern", "service"),
)

# What standing here does to behavior. 1.0 is a neutral street corner.
SPACES: dict[str, dict[str, float]] = {
    "town_hall": {"encounters": 1.35, "gossip": 1.10, "trade": 0.70, "politics": 1.50, "faith_tie": 1.05, "protest": 1.40},
    "market":    {"encounters": 1.30, "gossip": 1.15, "trade": 1.70, "politics": 0.90, "faith_tie": 0.90, "protest": 1.05},
    "tavern":    {"encounters": 1.45, "gossip": 1.85, "trade": 0.90, "politics": 0.85, "faith_tie": 0.95, "protest": 1.00},
    "chapel":    {"encounters": 1.15, "gossip": 1.20, "trade": 0.40, "politics": 0.80, "faith_tie": 1.70, "protest": 0.70},
    "park":      {"encounters": 1.25, "gossip": 1.25, "trade": 0.50, "politics": 1.05, "faith_tie": 1.00, "protest": 1.20},
    "homes":     {"encounters": 0.65, "gossip": 1.05, "trade": 0.45, "politics": 0.60, "faith_tie": 1.10, "protest": 0.50},
    "farm":      {"encounters": 0.75, "gossip": 0.80, "trade": 0.55, "politics": 0.50, "faith_tie": 1.05, "protest": 0.40},
    "workshop":  {"encounters": 0.70, "gossip": 0.75, "trade": 0.60, "politics": 0.55, "faith_tie": 0.80, "protest": 0.45},
    "clinic":    {"encounters": 0.70, "gossip": 0.85, "trade": 0.50, "politics": 0.55, "faith_tie": 0.90, "protest": 0.40},
    "bank":      {"encounters": 0.85, "gossip": 0.90, "trade": 1.15, "politics": 0.75, "faith_tie": 0.70, "protest": 0.80},
}

_NEIGHBORS: dict[str, dict[str, str]] = {}
for _a, _b, _kind in STREETS:
    _NEIGHBORS.setdefault(_a, {})[_b] = _kind
    _NEIGHBORS.setdefault(_b, {})[_a] = _kind


def space(location: str) -> dict[str, float]:
    return dict(SPACES.get(location, {
        "encounters": 1.0, "gossip": 1.0, "trade": 1.0,
        "politics": 1.0, "faith_tie": 1.0, "protest": 1.0,
    }))


def blocked_kinds(world: World) -> set[str]:
    """Which street types are closed this tick."""
    closed: set[str] = set()
    if world.active_rules.get("bridge_closed"):
        closed.add("bridge")
    mag = float((world.crisis_intensity or {}).get("flood", 0.0) or 0.0)
    if "flood" in (world.active_crises or set()) or mag > 0:
        closed.add("bridge")
        if mag >= 0.5:
            closed.add("greenway")
        if mag >= 0.8:
            closed.add("country")
    return closed


def edge_kind(a: str, b: str) -> str | None:
    return _NEIGHBORS.get(a, {}).get(b)


def neighbors(location: str, world: World | None = None) -> list[str]:
    raw = _NEIGHBORS.get(location, {})
    if world is None:
        return sorted(raw)
    closed = blocked_kinds(world)
    return sorted(dest for dest, kind in raw.items() if kind not in closed)


def next_hop(origin: str, destination: str, world: World) -> str | None:
    """First open street toward `destination`, or None if no route."""
    if origin == destination:
        return origin
    path = shortest_path(origin, destination, world)
    if not path or len(path) < 2:
        return None
    return path[1]


def shortest_path(origin: str, destination: str, world: World) -> list[str] | None:
    if origin == destination:
        return [origin]
    seen = {origin}
    queue: deque[list[str]] = deque([[origin]])
    while queue:
        path = queue.popleft()
        for dest in neighbors(path[-1], world):
            if dest in seen:
                continue
            nxt = path + [dest]
            if dest == destination:
                return nxt
            seen.add(dest)
            queue.append(nxt)
    return None


def reachable(origin: str, world: World) -> list[str]:
    found = {origin}
    queue = deque([origin])
    while queue:
        here = queue.popleft()
        for dest in neighbors(here, world):
            if dest not in found:
                found.add(dest)
                queue.append(dest)
    return sorted(found)


def snapshot(world: World) -> dict:
    """Frame payload for the Mobility view."""
    closed = sorted(blocked_kinds(world))
    edges = []
    for a, b, kind in STREETS:
        edges.append({
            "a": a, "b": b, "kind": kind,
            "open": kind not in closed,
        })
    return {"edges": edges, "blocked_kinds": closed}
