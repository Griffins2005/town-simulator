"""
decision_record.py -- The six-stage forensic trace of one agent's tick.

This is the society-lab contract: intelligence proposes, the engine
validates, and a user can reconstruct exactly why something happened.

    perceived -> retrieved_memories -> considered_actions
        -> chosen_intent -> validation -> state_changes / consequences

A Decider fills the first four stages (what the agent saw, remembered,
weighed, and attempted). actions.py fills validation, state_changes, and
consequences (what the world actually allowed). The two sides must not
collapse into each other: an LLM cannot invent gold by writing a
successful validation, and a rule-based agent cannot hide that it
considered a vote and then gossiped instead.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from decision import DecisionDraft, Intent, Perception


def snapshot_perception(p: Perception) -> dict:
    """Compact, JSON-safe slice of Perception for the trace -- not the
    entire object (inventories and proposal blobs get large). Enough to
    answer 'what did this agent know when they chose.'
    """
    return {
        "tick": p.tick,
        "location": p.self_location,
        "money": round(p.self_money, 2),
        "food": round(p.self_inventory.get("food", 0.0), 2),
        "reputation": round(p.self_reputation, 3),
        "present": list(p.location_agents),
        "crises": sorted(p.active_crises),
        "crisis_intensity": dict(getattr(p, "crisis_intensity", {}) or {}),
        "faction": p.self_faction_name or p.self_faction,
        "problems": list(getattr(p, "observed_problems", []) or []),
        "open_proposals": len(p.open_proposals),
        "pending_offers": len(p.pending_trade_offers),
    }


@dataclass
class DecisionDraft:
    """What a Decider produced before the world validated anything."""

    intent: Intent
    retrieved_memories: list = field(default_factory=list)
    considered_actions: list = field(default_factory=list)


@dataclass
class DecisionRecord:
    """One agent's complete decide-then-validate record for one tick."""

    agent_id: str
    tick: int
    perceived: dict
    retrieved_memories: list
    considered_actions: list
    chosen_intent: dict
    validation: dict
    state_changes: list
    consequences: list
    reused_cache: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def intent_as_dict(intent: Intent) -> dict:
    return {"action": intent.action, "args": dict(intent.args), "say": intent.say}


def assemble(
    agent_id: str,
    tick: int,
    perception: Perception,
    draft: DecisionDraft,
    result,
    reused_cache: bool,
) -> DecisionRecord:
    """Join Decider draft + ActionResult into the public record shape."""
    return DecisionRecord(
        agent_id=agent_id,
        tick=tick,
        perceived=snapshot_perception(perception),
        retrieved_memories=list(draft.retrieved_memories),
        considered_actions=list(draft.considered_actions),
        chosen_intent=intent_as_dict(draft.intent),
        validation={
            "ok": bool(result.success),
            "legal": bool(getattr(result, "legal", True)),
            "reason": result.reason,
        },
        state_changes=list(getattr(result, "state_changes", []) or []),
        consequences=list(getattr(result, "consequences", []) or [result.reason]),
        reused_cache=reused_cache,
    )
