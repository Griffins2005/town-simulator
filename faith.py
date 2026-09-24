"""
faith.py -- Religion as an institution, not a skin.

A resident's faith is identity; piety is how hard it pulls them.
Congregations gather on a schedule, piety warms at worship and cools
after scandal or a failed rite, and someone can convert only at a
living service -- never because a Decider invented doctrine.

The engine never authors belief. This file is the rule sheet:
schedule, census, worship/convert validators, and the two civic rites
(festival, water blessing) the hall can enact.
"""

from __future__ import annotations

from agent import Agent
from memory import MemoryEntry
from world import World

FAITHS = {
    "vale_covenant": "Vale Covenant",
    "hall_creed": "Hall Creed",
    "old_ways": "Old Ways",
    "unaffiliated": "Unaffiliated",
}

FAITH_HOME = {
    "vale_covenant": "chapel",
    "hall_creed": "town_hall",
    "old_ways": "farm",
    "unaffiliated": None,
}

# Hour-of-day window (tick % 24) when that congregation is in session.
WORSHIP_HOURS = {
    "vale_covenant": (6, 10),
    "hall_creed": (12, 16),
    "old_ways": (16, 20),
}

_ORDER = ("vale_covenant", "hall_creed", "old_ways", "unaffiliated")

# Last computed congregation leads — so tick() can log a change the
# way a new town leader is news, without storing an office on World.
_last_leaders: dict[str, str] = {}


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


def hour_of_day(world: World) -> int:
    return int(world.tick) % 24


def worship_now(faith_id: str | None, world: World) -> bool:
    if not faith_id or faith_id == "unaffiliated":
        return False
    start, end = WORSHIP_HOURS.get(faith_id, (None, None))
    if start is None:
        return False
    hour = hour_of_day(world)
    if world.active_rules.get("festival_faith") == faith_id:
        return True
    return start <= hour < end


def session_at(location: str, world: World) -> str | None:
    """Which faith is in session at this place, if any."""
    for faith_id, home in FAITH_HOME.items():
        if home == location and worship_now(faith_id, world):
            return faith_id
    return None


def census(agents: dict[str, Agent]) -> dict[str, int]:
    counts = {fid: 0 for fid in FAITHS}
    for agent in agents.values():
        fid = getattr(agent.persona, "faith", "unaffiliated") or "unaffiliated"
        counts[fid] = counts.get(fid, 0) + 1
    return counts


def congregation_flock(agents: dict[str, Agent], faith_id: str) -> list[Agent]:
    """Living members of one congregation. Unaffiliated have no flock."""
    if not faith_id or faith_id == "unaffiliated":
        return []
    return [
        a for a in agents.values()
        if not a.expelled and getattr(a.persona, "faith", None) == faith_id
    ]


def congregation_leader(agents: dict[str, Agent], faith_id: str | None) -> Agent | None:
    """The sitting pastoral lead: most services led, then piety, then reputation.

    Computed each tick, never stored as a title an agent can invent —
    same shape as governance.town_leader.
    """
    flock = congregation_flock(agents, faith_id or "")
    if not flock:
        return None
    return max(flock, key=lambda a: (
        getattr(a, "pastoral_track_record", 0),
        getattr(a.persona, "piety", 0),
        a.reputation,
        a.persona.sociability,
    ))


def leaders(agents: dict[str, Agent]) -> dict[str, Agent]:
    """One lead per living congregation."""
    found: dict[str, Agent] = {}
    for faith_id in ("vale_covenant", "hall_creed", "old_ways"):
        lead = congregation_leader(agents, faith_id)
        if lead is not None:
            found[faith_id] = lead
    return found


def leaders_public(agents: dict[str, Agent]) -> list[dict]:
    """Flat list for Perception, recorder, and the lab."""
    rows = []
    for faith_id, lead in leaders(agents).items():
        rows.append({
            "faith": faith_id,
            "faith_name": faith_name(faith_id),
            "id": lead.agent_id,
            "name": lead.persona.name,
            "location": lead.location,
        })
    return rows


def reset() -> None:
    _last_leaders.clear()


def congregation(agents: dict[str, Agent], faith_id: str) -> list[str]:
    return [
        a.agent_id for a in agents.values()
        if getattr(a.persona, "faith", None) == faith_id
    ]


def present_of_faith(agents: dict[str, Agent], location: str, faith_id: str) -> list[Agent]:
    return [
        a for a in agents.values()
        if a.location == location and getattr(a.persona, "faith", None) == faith_id
    ]


def water_blessing_bonus(world: World) -> float:
    return 0.15 if world.active_rules.get("water_blessing") else 0.0


def festival_faith(world: World, expire: bool = True) -> str | None:
    fid = world.active_rules.get("festival_faith")
    until = world.active_rules.get("festival_until")
    if fid and until is not None and world.tick >= int(until):
        if expire:
            world.active_rules.pop("festival_faith", None)
            world.active_rules.pop("festival_until", None)
            world.log_event("festival_ended", faith=fid)
        return None
    return fid if fid else None


def snapshot(world: World, agents: dict[str, Agent]) -> dict:
    """Frame payload: blocs, who is in session, where they gather."""
    sessions = []
    for faith_id in ("vale_covenant", "hall_creed", "old_ways"):
        if worship_now(faith_id, world):
            home = faith_home(faith_id)
            here = present_of_faith(agents, home, faith_id) if home else []
            sessions.append({
                "faith": faith_id,
                "faith_name": faith_name(faith_id),
                "location": home,
                "present": len(here),
            })
    return {
        "census": census(agents),
        "names": dict(FAITHS),
        "homes": dict(FAITH_HOME),
        "hour": hour_of_day(world),
        "sessions": sessions,
        "festival": festival_faith(world, expire=False),
        "water_blessing": bool(world.active_rules.get("water_blessing")),
        "leaders": leaders_public(agents),
    }


def worship_allowed(actor: Agent, world: World) -> tuple[bool, str]:
    fid = getattr(actor.persona, "faith", "unaffiliated")
    home = faith_home(fid)
    if not home:
        return False, "unaffiliated — no congregation to join"
    if actor.location != home:
        return False, f"worship is at the {home}"
    if not worship_now(fid, world):
        start, end = WORSHIP_HOURS[fid]
        return False, f"service is hours {start}–{end}"
    return True, "in session"


def convert_allowed(actor: Agent, faith_id: str, world: World, agents: dict[str, Agent]) -> tuple[bool, str]:
    if faith_id not in FAITHS or faith_id == "unaffiliated":
        return False, "not a living congregation"
    current = getattr(actor.persona, "faith", "unaffiliated")
    piety = float(getattr(actor.persona, "piety", 0.2))
    if current == faith_id:
        return False, "already of that faith"
    if current != "unaffiliated" and piety >= 0.22:
        return False, "piety is still holding them"
    home = faith_home(faith_id)
    if actor.location != home:
        return False, f"conversion is at the {home} during service"
    if not worship_now(faith_id, world):
        return False, "no service in session"
    present = present_of_faith(agents, home, faith_id)
    if len(present) < 2:
        return False, "the nave is empty — no congregation to receive them"
    return True, "received"


def apply_worship(actor: Agent, world: World, agents: dict[str, Agent]) -> tuple[bool, str, list]:
    ok, reason = worship_allowed(actor, world)
    if not ok:
        return False, reason, []
    fid = actor.persona.faith
    before = actor.persona.piety
    bump = 0.045
    if festival_faith(world) == fid:
        bump += 0.025
    actor.persona.piety = min(1.0, round(before + bump, 3))
    kin = present_of_faith(agents, actor.location, fid)
    for other in kin:
        if other.agent_id == actor.agent_id:
            continue
        actor.adjust_relationship(other.agent_id, 0.015)
        other.adjust_relationship(actor.agent_id, 0.015)
    if len(kin) >= 2:
        actor.pastoral_track_record = getattr(actor, "pastoral_track_record", 0) + 1
    actor.memory.add(MemoryEntry(world.tick, "worshipped", None, {"faith": fid}))
    world.log_event("worship", agent=actor.agent_id, faith=fid, location=actor.location)
    return True, "worshipped", [f"piety {before:.2f} → {actor.persona.piety:.2f}"]


def apply_convert(actor: Agent, faith_id: str, world: World, agents: dict[str, Agent]) -> tuple[bool, str, list]:
    ok, reason = convert_allowed(actor, faith_id, world, agents)
    if not ok:
        return False, reason, []
    previous = actor.persona.faith
    actor.persona.faith = faith_id
    actor.persona.piety = 0.42
    actor.memory.add(MemoryEntry(world.tick, "converted", None,
                                 {"from": previous, "to": faith_id}))
    world.log_event(
        "converted", agent=actor.agent_id, name=actor.persona.name,
        from_=previous, to=faith_id, location=actor.location,
    )
    world.notice_board.append({
        "tick": world.tick, "from": actor.location, "about": actor.agent_id,
        "text": f"{actor.persona.name} joined the {faith_name(faith_id)}",
    })
    del world.notice_board[:-8]
    return True, "converted", [f"{previous} → {faith_id}"]


def tick(world: World, agents: dict[str, Agent], rng) -> None:
    """Clock: open a session, warm or cool piety, maybe receive a convert."""
    festival_faith(world)
    _note_leaders(world, agents)
    hour = hour_of_day(world)
    for faith_id, (start, _end) in WORSHIP_HOURS.items():
        if hour == start:
            home = faith_home(faith_id)
            world.log_event("worship_session", faith=faith_id,
                            name=faith_name(faith_id), location=home)

    for agent in agents.values():
        fid = getattr(agent.persona, "faith", "unaffiliated")
        piety = float(getattr(agent.persona, "piety", 0.2))
        home = faith_home(fid)
        if home and agent.location == home and worship_now(fid, world):
            agent.persona.piety = min(1.0, round(piety + 0.012, 3))
        else:
            agent.persona.piety = max(0.05, round(piety - 0.004, 3))

    # Scandal cools the embezzler's congregation.
    for event in world.event_log[-80:]:
        if event.get("tick") != world.tick:
            continue
        if event.get("kind") == "corruption_scandal":
            who = agents.get(event.get("agent"))
            if who is None:
                continue
            fid = getattr(who.persona, "faith", None)
            for other in agents.values():
                if same_faith(fid, getattr(other.persona, "faith", None)):
                    other.persona.piety = max(0.05, round(other.persona.piety - 0.03, 3))
        if event.get("kind") == "proposal_closed" and not event.get("passed"):
            rtype = event.get("rule_type")
            if rtype in ("festival", "water_blessing"):
                for agent in agents.values():
                    if getattr(agent.persona, "piety", 0) > 0.3:
                        agent.persona.piety = max(0.05, round(agent.persona.piety - 0.02, 3))
        if event.get("kind") == "member_welcomed":
            guest = agents.get(event.get("agent"))
            if guest and getattr(guest.persona, "faith", None) != "unaffiliated":
                guest.persona.piety = min(1.0, round(guest.persona.piety + 0.04, 3))

    # An unaffiliated (or shaken) person at a living service may be received.
    for agent in list(agents.values()):
        loc_faith = session_at(agent.location, world)
        if not loc_faith:
            continue
        ok, _reason = convert_allowed(agent, loc_faith, world, agents)
        if not ok:
            continue
        present = present_of_faith(agents, agent.location, loc_faith)
        pull = min(0.35, 0.08 * len(present))
        if rng.random() < pull:
            apply_convert(agent, loc_faith, world, agents)


def _note_leaders(world: World, agents: dict[str, Agent]) -> None:
    """Log when a congregation's lead changes — civic news, not doctrine."""
    global _last_leaders
    now = {fid: lead.agent_id for fid, lead in leaders(agents).items()}
    for faith_id, agent_id in now.items():
        if _last_leaders.get(faith_id) == agent_id:
            continue
        who = agents.get(agent_id)
        world.log_event(
            "faith_leader",
            faith=faith_id,
            name=faith_name(faith_id),
            agent=agent_id,
            agent_name=who.persona.name if who else agent_id,
        )
    _last_leaders = now
