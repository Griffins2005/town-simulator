"""
inventions.py -- Agents propose inventions; the world validates them.

Hybrid rule, same as every other action:

    The Decider names a problem and a catalog kind (or an LLM-invented
    label mapped onto a catalog kind). This module checks knowledge,
    resources, location, and laws. Only then does construction consume
    materials and register a world capability.

Hard-coded catalog kinds are the *effect space* -- the physics of what
the simulation can change -- not a scripted 'Iris invents a pump on
tick 76' story. Which agent invents what, and whether others adopt it,
is decided at runtime. An unrecognized LLM kind fails validation
instead of spawning an undefined effect.
"""

from __future__ import annotations

from agent import Agent
from memory import MemoryEntry
from world import World

# Catalog: allowed capability changes. Agents pick a kind; they do not
# get to invent a new economic law by writing free-form JSON.
CATALOG = {
    "water_pump": {
        "name": "Water Pump",
        "problem": "farm_scarcity",
        "description": "Raises farm food regeneration.",
        "requires": {
            "locations": ("workshop", "farm"),
            "food": 3.0,
            "money": 2.5,
            "min_industriousness": 0.4,
            "memory_kinds": ("work", "gathered", "worked"),
        },
        "effect": {"type": "farm_regen_bonus", "delta": 0.18},
    },
    "price_board": {
        "name": "Market Price Board",
        "problem": "trade_friction",
        "description": "Posted prices make unfair offers easier to refuse.",
        "requires": {
            "locations": ("workshop", "market"),
            "food": 1.0,
            "money": 2.0,
            "min_industriousness": 0.3,
            "memory_kinds": ("spoke_to", "trade_completed", "heard_gossip"),
        },
        "effect": {"type": "trade_fairness_bonus", "delta": 0.12},
    },
    "public_ledger": {
        "name": "Public Ledger",
        "problem": "corruption",
        "description": "Open books lengthen the gap between scandals.",
        "requires": {
            "locations": ("workshop", "town_hall"),
            "food": 1.0,
            "money": 3.5,
            "min_industriousness": 0.35,
            "memory_kinds": ("heard_gossip", "cast_vote", "spoke_to"),
        },
        "effect": {"type": "corruption_cooldown_bonus", "delta": 8},
    },
}

_invention_ids = 0


def catalog_public() -> list[dict]:
    """What Perception may show: kinds, names, problems -- not internals."""
    return [
        {"kind": kind, "name": spec["name"], "problem": spec["problem"],
         "description": spec["description"]}
        for kind, spec in CATALOG.items()
    ]


def observed_problems(world: World, agents: dict[str, Agent]) -> list[str]:
    """Facts an agent can treat as inventable problems this tick."""
    problems: list[str] = []
    farm = world.locations.get("farm")
    if farm is not None and farm.resources.get("food", 0.0) < 28.0:
        problems.append("farm_scarcity")
    if "famine" in world.active_crises:
        problems.append("farm_scarcity")
    if "bank_run" in world.active_crises or any(
        e.get("kind") == "trade_failed_insufficient_funds"
        for e in world.event_log[-12:]
    ):
        problems.append("trade_friction")
    if any(e.get("kind") == "corruption_scandal" for e in world.event_log[-40:]):
        problems.append("corruption")
    if "curfew_after_tick_of_day" in world.active_rules:
        problems.append("curfew")
    return problems


def farm_regen_bonus(world: World) -> float:
    return _stacked_bonus(world, "farm_regen_bonus")


def trade_fairness_bonus(world: World) -> float:
    return _stacked_bonus(world, "trade_fairness_bonus")


def corruption_cooldown_bonus(world: World) -> int:
    return int(round(_stacked_bonus(world, "corruption_cooldown_bonus") * 40))


def _stacked_bonus(world: World, effect_type: str) -> float:
    total = 0.0
    for inv in world.inventions:
        effect = inv.get("effect") or {}
        if effect.get("type") != effect_type:
            continue
        adopters = max(1, len(inv.get("adopters") or []))
        scale = min(1.4, 0.55 + 0.12 * adopters)
        total += float(effect.get("delta", 0.0)) * scale
    return total


def try_invent(actor: Agent, kind: str, world: World) -> tuple[bool, str, list, list]:
    """Validate and, if legal, construct. Returns
    (ok, reason, state_changes, consequences).
    """
    global _invention_ids
    spec = CATALOG.get(kind)
    if spec is None:
        return False, f"unknown invention kind '{kind}'", [], [
            "unvalidated invention rejected — not in the capability catalog"
        ]
    req = spec["requires"]
    if actor.location not in req["locations"]:
        return False, f"must be at {' or '.join(req['locations'])} to build this", [], [
            "construction blocked: wrong location"
        ]
    if actor.persona.industriousness < req["min_industriousness"]:
        return False, "not enough practical knowledge", [], [
            "construction blocked: industriousness below the knowledge bar"
        ]
    memory_kinds = {e.kind for e in actor.memory.all()}
    needed = set(req["memory_kinds"])
    if needed and memory_kinds.isdisjoint(needed):
        return False, "no relevant experience to draw on", [], [
            "construction blocked: no matching memories"
        ]
    food = actor.inventory.get("food", 0.0)
    if food < req["food"]:
        return False, f"need {req['food']} food, have {food:.1f}", [], [
            "construction blocked: insufficient materials"
        ]
    if actor.money < req["money"]:
        return False, f"need {req['money']} money, have {actor.money:.2f}", [], [
            "construction blocked: insufficient funds"
        ]
    if any(inv.get("kind") == kind for inv in world.inventions):
        return False, f"{spec['name']} already exists in this town", [], [
            "duplicate invention rejected"
        ]

    actor.inventory["food"] = round(food - req["food"], 2)
    actor.money = round(actor.money - req["money"], 2)
    _invention_ids += 1
    record = {
        "id": _invention_ids,
        "kind": kind,
        "name": spec["name"],
        "inventor": actor.agent_id,
        "tick": world.tick,
        "adopters": [actor.agent_id],
        "effect": dict(spec["effect"]),
    }
    world.inventions.append(record)
    actor.memory.add(MemoryEntry(
        world.tick, "invented", None,
        {"kind": kind, "name": spec["name"]}, salience=0.9,
    ))
    world.log_event(
        "invention",
        agent=actor.agent_id,
        invention_id=record["id"],
        invention_kind=kind,
        name=spec["name"],
    )
    changes = [
        {"field": "inventory.food", "delta": -req["food"]},
        {"field": "money", "delta": -req["money"]},
        {"field": "world.inventions", "added": spec["name"]},
    ]
    consequences = [
        f"{spec['name']} now exists",
        spec["description"],
        "others may adopt, ignore, or later regulate it",
    ]
    return True, f"invented {spec['name']}", changes, consequences


def try_adopt(actor: Agent, invention_id: int, world: World) -> tuple[bool, str, list, list]:
    inv = next((i for i in world.inventions if i.get("id") == invention_id), None)
    if inv is None:
        return False, f"no invention '{invention_id}'", [], ["adoption failed: unknown invention"]
    if actor.agent_id in inv["adopters"]:
        return False, "already adopted", [], ["no change"]
    inv["adopters"].append(actor.agent_id)
    actor.memory.add(MemoryEntry(
        world.tick, "adopted_invention", inv.get("inventor"),
        {"name": inv["name"], "id": invention_id}, salience=0.7,
    ))
    world.log_event(
        "invention_adopted",
        agent=actor.agent_id,
        invention_id=invention_id,
        name=inv["name"],
    )
    return True, f"adopted {inv['name']}", [
        {"field": "adopters", "added": actor.agent_id}
    ], [f"{inv['name']} gained an adopter — its effect strengthens"]


def reset() -> None:
    global _invention_ids
    _invention_ids = 0


def export_state() -> dict:
    return {"next_id": _invention_ids}


def import_state(blob: dict) -> None:
    global _invention_ids
    _invention_ids = int(blob.get("next_id") or 0)
