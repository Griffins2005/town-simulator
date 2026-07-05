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

from dataclasses import dataclass

from agent import Agent
from decision import Intent
from memory import MemoryEntry
from world import World

import economy
import governance


@dataclass
class ActionResult:
    """Outcome of attempting to execute one Intent."""

    success: bool
    reason: str  # human-readable explanation, useful for logs/debugging
    # and, eventually, as a perception input ("your last action failed
    # because...") so an LLM-driven agent can adapt next tick.


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
    """Handler for the "move" action. Relocates `actor` to
    `intent.args["destination"]` if that location exists and no active
    rule (e.g. a curfew) currently blocks movement for this agent.
    """
    destination = intent.args.get("destination")
    if destination not in world.locations:
        return ActionResult(False, f"no such location '{destination}'")

    # Governance check: a passed rule can block movement (e.g. curfew).
    # This is the concrete mechanism that makes "governance" more than
    # theater -- active_rules is read here, not just logged elsewhere.
    if governance.is_movement_blocked(world, actor):
        actor.memory.add(MemoryEntry(world.tick, "move_blocked_by_rule", None,
                                      {"attempted": destination}))
        # A blocked attempt is itself a (mild) norm violation worth a
        # small reputation hit -- this is the actual enforcement
        # mechanism for "norms have consequences," distinct from
        # economy.py's reward-for-compliance side. Deliberately small:
        # a single curfew bump shouldn't tank reputation the way
        # repeated violations should -- and repeated violations DO
        # compound here since each one calls this same line.
        RULE_VIOLATION_REPUTATION_DELTA = 0.02
        actor.reputation = max(0.0, actor.reputation - RULE_VIOLATION_REPUTATION_DELTA)
        return ActionResult(False, "movement blocked by active rule (e.g. curfew)")

    actor.location = destination
    world.log_event("move", agent=actor.agent_id, to=destination)
    return ActionResult(True, "moved")


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
    actor.adjust_relationship(target.agent_id, SPEAK_RELATIONSHIP_DELTA)
    target.adjust_relationship(actor.agent_id, SPEAK_RELATIONSHIP_DELTA)
    world.log_event("speak", agent=actor.agent_id, to=target.agent_id)
    return ActionResult(True, "spoke")


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

    GOSSIP_REPUTATION_NUDGE = 0.03
    direction = -1 if "negative" in (intent.say or "").lower() else 0
    for listener_id in listeners:
        listener = agents.get(listener_id)
        if listener is None or listener_id == about_id:
            continue
        listener.memory.add(MemoryEntry(world.tick, "heard_gossip", about_id,
                                         {"from": actor.agent_id, "said": intent.say or ""}))
        if direction:
            listener.adjust_relationship(about_id, direction * GOSSIP_REPUTATION_NUDGE)

    world.log_event("gossip", agent=actor.agent_id, about=about_id, heard_by=listeners)
    return ActionResult(True, "gossiped")


def _propose_rule(actor: Agent, intent: Intent, world: World, agents: dict[str, Agent]) -> ActionResult:
    """Handler for the "propose_rule" action. Delegates entirely to
    `governance.propose`, which opens a new proposal for the rule_type
    and rule_args given in `intent.args`.
    """
    return governance.propose(actor, intent.args, world)


def _vote(actor: Agent, intent: Intent, world: World, agents: dict[str, Agent]) -> ActionResult:
    """Handler for the "vote" action. Delegates entirely to
    `governance.cast_vote` for the proposal named in
    `intent.args["proposal_id"]` with the choice in
    `intent.args["choice"]` ("yes" or "no").
    """
    return governance.cast_vote(actor, intent.args.get("proposal_id"), intent.args.get("choice"), world)


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
    "speak": _speak,
    "gossip": _gossip,
    "propose_rule": _propose_rule,
    "vote": _vote,
    "idle": _idle,
}
