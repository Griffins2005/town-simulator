"""
actions.py -- The legal action set. The ONLY code that mutates state.

Design principle: every Intent produced by a Decider (rule-based today, LLM
later) must pass through here before it has any effect. This module:
    1. validates the Intent against current world/agent state and any
       active_rules (this is what gives governance real teeth -- a passed
       law can make an action illegal here, not just "frowned upon"),
    2. mutates state if and only if valid,
    3. records a memory entry for every agent who should plausibly know
       about what happened (the actor, and anyone present),
    4. returns a small result record describing what happened (success/
       failure + reason), which the engine logs and can feed back to the
       Decider next tick.

A rejected/illegal Intent never raises an exception up to the engine loop
-- it returns a failure result. This matters because Phase 2 intents come
from an LLM and WILL occasionally be malformed or attempt something
illegal (e.g. moving to a nonexistent location, trading money the agent
doesn't have). The engine must keep running; the agent just "fails" that
action, the way a human bouncing off a locked door doesn't crash reality.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agent import Agent
from decision import Intent
from memory import MemoryEntry
from world import World

import economy
import faith
import geography
import governance
import inventions


@dataclass
class ActionResult:
    """Outcome of attempting to execute one Intent."""

    success: bool
    reason: str
    legal: bool = True
    state_changes: list = field(default_factory=list)
    consequences: list = field(default_factory=list)


def execute(actor: Agent, intent: Intent, world: World, agents: dict[str, Agent]) -> ActionResult:
    """Dispatch `intent` to the right handler. Single entry point used by
    engine.py -- nothing else should call the handlers below directly.
    """
    handler = _REGISTRY.get(intent.action)
    if handler is None:
        # Unrecognized action name (typo, or Phase 2 LLM hallucinated an
        # action that doesn't exist). Fail safely rather than crashing.
        world.log_event("malformed_intent", agent=actor.agent_id, action=intent.action)
        return ActionResult(False, f"unknown action '{intent.action}'")
    return handler(actor, intent, world, agents)


def _move(actor: Agent, intent: Intent, world: World, agents: dict[str, Agent]) -> ActionResult:
    """Walk one open street toward `intent.args["destination"]`.

    The Decider names a place; this handler walks the next hop on the
    street graph. A flood, a closed bridge, or a missing road can make
    the destination unreachable — the agent fails that tick instead of
    teleporting.
    """
    destination = intent.args.get("destination")
    if destination not in world.locations:
        return ActionResult(False, f"no such location '{destination}'")

    if governance.is_movement_blocked(world, actor):
        actor.memory.add(MemoryEntry(world.tick, "move_blocked_by_rule", None,
                                      {"attempted": destination}))
        RULE_VIOLATION_REPUTATION_DELTA = 0.02
        actor.reputation = max(0.0, actor.reputation - RULE_VIOLATION_REPUTATION_DELTA)
        return ActionResult(
            False, "movement blocked by active rule (e.g. curfew)",
            legal=False,
            consequences=["rule enforced", "reputation -0.02"],
        )

    previous = actor.location
    if previous == destination:
        return ActionResult(True, "already there")

    hop = geography.next_hop(previous, destination, world)
    if hop is None:
        actor.memory.add(MemoryEntry(world.tick, "road_closed", None,
                                      {"attempted": destination}))
        world.log_event("road_closed", agent=actor.agent_id, from_=previous,
                        attempted=destination)
        return ActionResult(
            False, f"no open street from {previous} toward {destination}",
            legal=True,
            consequences=["flood or a closed road blocked the way"],
        )

    actor.location = hop
    heading = destination if hop != destination else None
    world.log_event("move", agent=actor.agent_id, to=hop,
                    heading=heading, from_=previous)
    note = f"now at {hop}" if not heading else f"now at {hop}, still heading for {destination}"
    return ActionResult(
        True, "walked" if heading else "moved",
        state_changes=[{"field": "location", "from": previous, "to": hop}],
        consequences=[note],
    )


def _work(actor: Agent, intent: Intent, world: World, agents: dict[str, Agent]) -> ActionResult:
    """Handler for the "work" action. Delegates entirely to
    `economy.work`, which extracts resources from the actor's current
    location into their inventory. Takes no args.
    """
    return economy.work(actor, world)


def _trade_offer(actor: Agent, intent: Intent, world: World, agents: dict[str, Agent]) -> ActionResult:
    """Handler for the "trade_offer" action. Validates the target agent
    (`intent.args["to"]`) exists, then delegates to `economy.create_offer`
    to register a pending offer for that agent to later accept/reject.
    """
    target_id = intent.args.get("to")
    target = agents.get(target_id)
    if target is None:
        return ActionResult(False, f"no such agent '{target_id}'")
    return economy.create_offer(actor, target, intent.args)


def _trade_accept(actor: Agent, intent: Intent, world: World, agents: dict[str, Agent]) -> ActionResult:
    """Handler for the "trade_accept" action. Delegates to
    `economy.resolve_offer` with accept=True for the offer named in
    `intent.args["offer_id"]`.
    """
    return economy.resolve_offer(actor, intent.args.get("offer_id"), accept=True, world=world, agents=agents)


def _trade_reject(actor: Agent, intent: Intent, world: World, agents: dict[str, Agent]) -> ActionResult:
    """Handler for the "trade_reject" action. Delegates to
    `economy.resolve_offer` with accept=False for the offer named in
    `intent.args["offer_id"]`.
    """
    return economy.resolve_offer(actor, intent.args.get("offer_id"), accept=False, world=world, agents=agents)


def _trade_counter(actor: Agent, intent: Intent, world: World, agents: dict[str, Agent]) -> ActionResult:
    """Revise the money side of a pending offer. Engine validates the coins."""
    return economy.counter_offer(actor, intent.args, world, agents)


def _bid(actor: Agent, intent: Intent, world: World, agents: dict[str, Agent]) -> ActionResult:
    """Seal a bid on an open market lot. First bid sticks."""
    return economy.place_bid(actor, intent.args, world)


def _speak(actor: Agent, intent: Intent, world: World, agents: dict[str, Agent]) -> ActionResult:
    """Handler for the "speak" action. Records a memory of the exchange
    for both `actor` and the target (`intent.args["to"]`), and nudges
    their mutual relationship slightly upward -- speaking is mildly
    bonding. `intent.say`, if present, is stored as the spoken content.
    """
    target_id = intent.args.get("to")
    target = agents.get(target_id)
    if target is None:
        return ActionResult(False, f"no such agent '{target_id}'")

    target.memory.add(MemoryEntry(world.tick, "was_spoken_to", actor.agent_id,
                                   {"said": intent.say or ""}))
    actor.memory.add(MemoryEntry(world.tick, "spoke_to", target.agent_id,
                                  {"said": intent.say or ""}))
    # Small relationship nudge both ways -- speaking is mildly bonding.
    # This is a deliberately tiny, named constant rather than a magic
    # number buried inline, so the "social activities" tuning knob is
    # easy to find later.
    SPEAK_RELATIONSHIP_DELTA = 0.02
    weights = geography.space(actor.location)
    delta = SPEAK_RELATIONSHIP_DELTA * weights.get("encounters", 1.0)
    if faith.same_faith(getattr(actor.persona, "faith", None),
                        getattr(target.persona, "faith", None)):
        delta *= weights.get("faith_tie", 1.0)
    actor.adjust_relationship(target.agent_id, delta)
    target.adjust_relationship(actor.agent_id, delta)
    # `said` is included in the world-level event (not just the per-agent
    # memory entries above) so recorder.py's frame pass-through -- and by
    # extension live_server.py's SSE feed -- can render the actual
    # utterance without any recorder- or live_server-specific plumbing.
    # This is the ONLY line that changes to make agent dialogue visible
    # town-wide instead of only reconstructable from individual memories.
    world.log_event("speak", agent=actor.agent_id, to=target.agent_id, said=intent.say or "")
    return ActionResult(
        True, "spoke",
        state_changes=[{"field": "relationship", "with": target.agent_id, "delta": round(delta, 4)}],
        consequences=[f"relationship {delta:+.3f} both ways"],
    )


def _gossip(actor: Agent, intent: Intent, world: World, agents: dict[str, Agent]) -> ActionResult:
    """Gossip is how reputation propagates WITHOUT every agent directly
    witnessing every event -- the actual mechanism norm-formation depends
    on in a town too large for everyone to see everything. The listener's
    opinion of the gossip's subject shifts a little even though the
    listener never witnessed anything themselves; this is deliberately
    weaker than a first-hand witness update (see economy.py / governance.py
    for those), modeling the real-world fact that secondhand information
    is trusted less than direct observation.
    """
    about_id = intent.args.get("about")
    listeners = intent.args.get("to")
    about = agents.get(about_id)
    if about is None:
        return ActionResult(False, f"no such agent '{about_id}'")

    # If no explicit listener given, gossip to whoever else is at the
    # actor's current location (mirrors the rule-based Decider's usage).
    if listeners is None:
        listeners = [aid for aid, a in agents.items()
                     if a.location == actor.location and aid != actor.agent_id]
    else:
        listeners = [listeners] if isinstance(listeners, str) else listeners

    text = (intent.say or "").lower()
    tone = intent.args.get("tone")
    if not tone:
        if any(w in text for w in ("expel", "cast out", "banish", "throw out", "out of this town")):
            tone = "expel"
        elif any(w in text for w in ("scandal", "embezzl", "thief", "corrupt", "shame", "caught")):
            tone = "scandal"
        elif any(w in text for w in ("never trust", "liar", "negative", "crook", "rotten")):
            tone = "accuse"
        elif any(w in text for w in ("welcome", "new face", "join us", "glad you're here")):
            tone = "welcome"
        elif any(w in text for w in ("hero", "saved", "praise", "good soul", "stands up")):
            tone = "praise"
        else:
            tone = "chat"

    hit = {"expel": -0.05, "scandal": -0.035, "accuse": -0.025, "praise": 0.02, "welcome": 0.015}.get(tone, 0.0)
    hit *= geography.space(actor.location).get("gossip", 1.0)
    before = about.reputation
    if hit:
        about.reputation = max(0.0, min(1.0, about.reputation + hit))
    for listener_id in listeners:
        listener = agents.get(listener_id)
        if listener is None or listener_id == about_id:
            continue
        listener.memory.add(MemoryEntry(world.tick, "heard_gossip", about_id,
                                         {"from": actor.agent_id, "said": intent.say or "", "tone": tone}))
        if hit:
            listener.adjust_relationship(about_id, hit)

    if tone == "expel":
        world.log_event("call_for_expulsion", agent=actor.agent_id, about=about_id,
                        said=intent.say or "")
    if before >= 0.20 and about.reputation < 0.20 and tone in ("expel", "scandal", "accuse"):
        world.log_event("notoriety", agent=about_id, name=about.persona.name,
                        reputation=round(about.reputation, 3))

    world.log_event("gossip", agent=actor.agent_id, about=about_id, heard_by=listeners,
                     said=intent.say or "", tone=tone)
    return ActionResult(True, "gossiped", consequences=[f"tone={tone}"])


def _propose_rule(actor: Agent, intent: Intent, world: World, agents: dict[str, Agent]) -> ActionResult:
    """Handler for the "propose_rule" action. Delegates entirely to
    `governance.propose`, which opens a new proposal for the rule_type
    and rule_args given in `intent.args`.
    """
    rule_type = intent.args.get("rule_type")
    rule_args = intent.args.get("rule_args") or {}
    target_id = rule_args.get("target_agent") or rule_args.get("target_agent_id")
    if rule_type in ("expel", "suspend_vote", "welcome", "impeach", "elect") and target_id:
        target = agents.get(target_id)
        if target is None:
            return ActionResult(False, f"no such agent '{target_id}'")
        if rule_type == "expel" and target.expelled:
            return ActionResult(False, "already expelled")
        if rule_type == "welcome" and target.can_vote(world.tick) and not target.expelled:
            return ActionResult(False, "they already have a seat")
        if rule_type == "expel":
            remaining = sum(1 for a in agents.values()
                            if a.can_vote(world.tick) and a.agent_id != target_id)
            if remaining < governance.MIN_ELIGIBLE_AFTER_EXPEL:
                return ActionResult(False, "town too small to expel anyone")
        if rule_type == "impeach":
            if target_id != world.town_leader_id:
                return ActionResult(False, "impeach names the sitting leader")
            if not governance.impeach_justified(world, target):
                return ActionResult(False, "no grounds to impeach — no crisis, ruin, or collapsed reputation")
        if rule_type == "elect":
            if world.town_leader_id:
                return ActionResult(False, "someone already holds the chair")
            if target.expelled or not target.can_vote(world.tick):
                return ActionResult(False, "that name is not eligible for the chair")
            if getattr(target, "solvency", "ok") == "bankrupt":
                return ActionResult(False, "a bankrupt resident cannot take the chair")
    return governance.propose(actor, intent.args, world)


def _lobby(actor: Agent, intent: Intent, world: World, agents: dict[str, Agent]) -> ActionResult:
    """One-on-one political pressure. Engine validates; Decider only names the target."""
    return governance.apply_lobby(actor, intent.args, world, agents)


def _vote(actor: Agent, intent: Intent, world: World, agents: dict[str, Agent]) -> ActionResult:
    """Handler for the "vote" action. Delegates entirely to
    `governance.cast_vote` for the proposal named in
    `intent.args["proposal_id"]` with the choice in
    `intent.args["choice"]` ("yes" or "no").
    """
    return governance.cast_vote(actor, intent.args.get("proposal_id"), intent.args.get("choice"), world)


def _invent(actor: Agent, intent: Intent, world: World, agents: dict[str, Agent]) -> ActionResult:
    """Agent proposes a catalog invention; inventions.py validates."""
    kind = intent.args.get("invention_kind") or intent.args.get("kind")
    ok, reason, changes, consequences = inventions.try_invent(actor, kind, world)
    return ActionResult(ok, reason, legal=ok or "unknown" not in reason,
                        state_changes=changes, consequences=consequences)


def _adopt_invention(actor: Agent, intent: Intent, world: World, agents: dict[str, Agent]) -> ActionResult:
    invention_id = intent.args.get("invention_id")
    try:
        invention_id = int(invention_id)
    except (TypeError, ValueError):
        return ActionResult(False, "invention_id must be an integer", legal=False)
    ok, reason, changes, consequences = inventions.try_adopt(actor, invention_id, world)
    return ActionResult(ok, reason, state_changes=changes, consequences=consequences)


def _worship(actor: Agent, intent: Intent, world: World, agents: dict[str, Agent]) -> ActionResult:
    """Join the service at this agent's faith home. faith.py is the rule sheet."""
    ok, reason, consequences = faith.apply_worship(actor, world, agents)
    return ActionResult(ok, reason, legal=ok or "unaffiliated" not in reason,
                        consequences=consequences)


def _convert(actor: Agent, intent: Intent, world: World, agents: dict[str, Agent]) -> ActionResult:
    """Be received by a living congregation. Decider names the faith; faith.py validates."""
    faith_id = intent.args.get("faith") or intent.args.get("to_faith") or intent.args.get("convert_faith")
    if not faith_id:
        faith_id = faith.session_at(actor.location, world)
    if not faith_id:
        return ActionResult(False, "no congregation in session here")
    ok, reason, consequences = faith.apply_convert(actor, faith_id, world, agents)
    return ActionResult(ok, reason, consequences=consequences)


def _idle(actor: Agent, intent: Intent, world: World, agents: dict[str, Agent]) -> ActionResult:
    """Handler for the "idle" action. Does nothing and always succeeds
    -- the engine's and Deciders' default/fallback action when there's
    nothing else to do (or when an LLM-backed Decider's call failed; see
    llm_decider.py).
    """
    return ActionResult(True, "idled")


# Registry mapping action name -> handler. Adding a new action means
# adding one function above and one entry here -- engine.py and
# decision.py never need to change. Mirrors the "one handler per
# algorithm, same dispatch shape" pattern from the NIST validation
# framework, deliberately -- it's the same architectural move applied
# to a different domain.
_REGISTRY = {
    "move": _move,
    "work": _work,
    "trade_offer": _trade_offer,
    "trade_accept": _trade_accept,
    "trade_reject": _trade_reject,
    "trade_counter": _trade_counter,
    "bid": _bid,
    "speak": _speak,
    "gossip": _gossip,
    "propose_rule": _propose_rule,
    "vote": _vote,
    "lobby": _lobby,
    "invent": _invent,
    "adopt_invention": _adopt_invention,
    "worship": _worship,
    "convert": _convert,
    "idle": _idle,
}
