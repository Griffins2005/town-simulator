"""
world.py -- The world state container.

Design principle: `World` is a dumb data bag. It holds state but contains
almost no behavior. All state *mutation* happens through `actions.py`, never
directly on a `World` instance from elsewhere in the codebase. This is the
single most important invariant in the whole system:

    INVARIANT: World/Agent attributes are only ever mutated inside
    actions.py (or governance.py / economy.py, which actions.py delegates
    to). Every other module *reads* state; only those write it.

Why this matters: once agents are driven by an LLM, agent "decisions" are
free-form and occasionally malformed. If state could be mutated from
anywhere, a bug in decision-making could corrupt the ledger, teleport an
agent, or grant infinite reputation, and you'd have no idea where to look.
With this invariant, "money appeared from nowhere" has exactly one place
it could have happened: actions.py. That constraint is what makes a
15-20 agent simulation debuggable once the brain layer becomes
nondeterministic (Phase 2, LLM-driven agents).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Location:
    """A named place agents can occupy (e.g. 'market', 'town_hall', 'farm').

    Locations are intentionally minimal -- no internal logic, just an
    identity and an optional resource pool. Production/consumption logic
    lives in economy.py, which reads/writes `resources` here.
    """

    name: str
    # Generic resource pool for this location, e.g. {"food": 40, "wood": 10}.
    # economy.py treats this as the source agents draw from when working
    # this location, and the sink they deposit into when producing here.
    resources: dict[str, float] = field(default_factory=dict)


class World:
    """Holds all simulation state: tick counter, locations, and the active
    rule set (rules are stored here so actions.py can check legality against
    them; the rules themselves are created/voted on in governance.py).

    Agents are NOT stored on World -- they're owned by the Engine (see
    engine.py) and passed into action/decision calls explicitly. This keeps
    World free of agent-management concerns (adding/removing agents,
    iteration order) and focused purely on "the town itself": its places,
    its clock, and its laws.
    """

    def __init__(self, locations: list[Location]) -> None:
        """
        Args:
            locations: the full set of Location objects making up this
                town, indexed internally by name. Passing two locations
                with the same name will silently let the later one win
                (dict construction semantics) -- callers should ensure
                names are unique.
        """
        self.tick: int = 0
        self.locations: dict[str, Location] = {loc.name: loc for loc in locations}

        # Active rules, e.g. {"curfew_after_tick": 20}. Populated by
        # governance.py when a proposal passes a vote. actions.py reads
        # this dict to decide whether an action is currently legal --
        # this is what gives governance real teeth instead of being
        # flavor text layered on top of an unaffected simulation.
        self.active_rules: dict[str, object] = {}

        # Collective fund, separate from any individual agent's money.
        # Currently the sole purpose is to receive proceeds from an
        # enacted "wealth_tax" rule (see governance.py) before they're
        # redistributed back to the population -- kept as an explicit,
        # inspectable pool rather than redistributing instantaneously,
        # so a simulation observer (or a future LLM agent reasoning
        # about town finances) can see "how much is in the town's
        # collective fund right now" as a first-class fact, the same
        # way a real town's treasury balance is public information.
        self.treasury: float = 0.0

        # Append-only town-wide event log (distinct from each agent's
        # *personal* memory in memory.py). Useful for debugging and for
        # any agent decision logic that needs "what just happened publicly"
        # rather than "what did I personally witness."
        self.event_log: list[dict] = []

        # agent_id -> faction_id, populated by chaos.py's faction-
        # formation logic (built from REPEATED voting alignment between
        # agents over time, not assigned upfront -- a faction is
        # something that emerges from a voting history, not a label
        # stamped on agents at creation). Agents with no faction_id here
        # simply haven't aligned with anyone consistently yet. Stored on
        # World rather than computed fresh each time because faction
        # membership is genuinely persistent town state, the same way
        # active_rules is -- not something to recompute from scratch
        # every tick.
        self.factions: dict[str, str] = {}

        # A small set of currently-active crisis tags (e.g.
        # "bank_run", "famine", "political_crisis"), populated and
        # cleared by chaos.py. Kept as a set of short string tags
        # rather than individual boolean fields, since the set of
        # possible crises is expected to grow and a fixed list of
        # World attributes would need editing every time a new crisis
        # type is added -- the same reasoning as active_rules being a
        # flexible dict rather than fixed fields.
        self.active_crises: set[str] = set()

    def get_location(self, name: str) -> Location:
        """Look up a location by name.

        Raises:
            KeyError: if `name` is not a registered location. We deliberately
                do NOT silently create missing locations -- a typo'd location
                name (e.g. from a malformed LLM action in Phase 2) should
                fail loudly here rather than quietly spawning a new empty
                place in the world.
        """
        return self.locations[name]

    def log_event(self, kind: str, **details: object) -> None:
        """Append a structured event to the town-wide log.

        `kind` is a short event-type tag (e.g. "trade", "vote_cast",
        "rule_passed"). `details` is freeform but should be JSON-serializable
        since this log is the natural place to dump the simulation to disk
        for later analysis.
        """
        self.event_log.append({"tick": self.tick, "kind": kind, **details})
