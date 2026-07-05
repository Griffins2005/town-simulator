"""
agent.py -- The Agent: persona, state, and the swappable decision interface.

Design principle: Agent holds DATA (who they are, what they have, what they
remember). It does NOT decide what to do -- that's delegated to a
`Decider` (see decision.py) injected at construction time. This is the
seam where Phase 1's rule-based stub gets swapped for a Phase 2 LLM call,
with zero changes to Agent, World, actions.py, governance.py, or
economy.py. Only decision.py changes.

This separation is the single highest-leverage design choice in this
codebase: it means the entire engine can be built, run, and debugged today,
for free, with deterministic agents -- and "plugging in the LLM" later is
additive, not a rewrite.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from memory import MemoryLog


@dataclass
class Persona:
    """Static-ish identity traits for an agent.

    These are intentionally simple scalar traits rather than free-text
    personality descriptions. Free text reads nicely in a demo but can't be
    used by rule-based logic (Phase 1) for anything -- you can't compute
    "is this agent likely to break the curfew" from a paragraph without an
    LLM call. Scalars let Phase 1's rule-based Decider make real, legible
    decisions, and Phase 2's LLM prompt-builder can still render them to
    prose for flavor ("Marcus is a stingy, industrious loner").

    All trait values are floats in [0, 1]. None of these are mutated
    directly by other modules except via Agent helper methods, to keep
    drift (trait change over time -- the "evolving" part of "dynamic
    personas") auditable through memory events rather than silent.
    """

    name: str
    industriousness: float = 0.5  # likelihood of choosing to work over idling
    generosity: float = 0.5       # likelihood of accepting unfavorable trades, gifting
    sociability: float = 0.5      # likelihood of seeking out other agents
    rule_respect: float = 0.5     # likelihood of complying with active_rules
    risk_tolerance: float = 0.5   # likelihood of breaking norms when self-interest is high


@dataclass
class Agent:
    """A single town resident.

    Attributes:
        agent_id: unique stable identifier, e.g. "agent_03". Used as the
            key everywhere else (ledger, reputation dicts, votes) rather
            than `persona.name`, since names are cosmetic and could collide
            or change; IDs never do.
        persona: see Persona above.
        location: name of the Location this agent currently occupies. Must
            always be a valid key in World.locations -- enforced by
            actions.py's move validator, never set directly.
        money: this agent's balance in the shared ledger currency. Mutated
            only by economy.py's trade resolution.
        inventory: dict of item_name -> quantity, e.g. {"food": 3}.
        relationships: agent_id -> float in [-1, 1], this agent's personal
            opinion of another agent. Distinct from town-wide `reputation`
            (below) -- relationships are private/subjective, reputation is
            the aggregated public signal. A rule-based or LLM decider can
            use both: "I personally like Marcus (relationships) even though
            the town thinks he's a thief (reputation)."
        reputation: THIS agent's *own* public reputation score, as seen by
            the town. Stored on the agent itself (rather than in some
            global registry) because it's most naturally "a fact about
            this agent." Mutated by governance.py / actions.py whenever
            this agent's witnessed behavior should move the needle.
        memory: this agent's MemoryLog.
        decider: the swappable brain. See decision.py for the interface.
    """

    agent_id: str
    persona: Persona
    location: str
    money: float = 0.0
    inventory: dict[str, float] = field(default_factory=dict)
    relationships: dict[str, float] = field(default_factory=dict)
    reputation: float = 0.5  # neutral starting reputation
    # Count of this agent's OWN proposals (including repeals) that have
    # passed a vote -- incremented by governance.py's _finalize. This is
    # the "official" signal chaos.py's corruption mechanism reads: an
    # agent with a real track record of getting things enacted is more
    # politically established than a random bystander, and that
    # establishment is exactly what makes embezzlement opportunities
    # both more available (closer to the levers of power) and more
    # consequential if discovered (further to fall). Starts at 0 for
    # every agent -- nobody is born an official.
    official_track_record: int = 0
    memory: MemoryLog = field(default_factory=MemoryLog)
    decider: object = None  # type: Decider, kept loose to avoid circular import

    def relationship_with(self, other_id: str) -> float:
        """Get this agent's opinion of `other_id`, defaulting to neutral (0.0)
        if they've never interacted."""
        return self.relationships.get(other_id, 0.0)

    def adjust_relationship(self, other_id: str, delta: float) -> None:
        """Nudge this agent's opinion of `other_id` by `delta`, clamped to
        [-1, 1]. This is the ONLY way relationships should change -- always
        via this method, never by writing `agent.relationships[x] = y`
        directly, so every relationship change is a single auditable call
        site pattern (easy to grep, easy to log if needed later).
        """
        current = self.relationship_with(other_id)
        self.relationships[other_id] = max(-1.0, min(1.0, current + delta))
