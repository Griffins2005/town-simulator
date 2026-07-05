"""
memory.py -- Per-agent memory log.

Design principle: memory is structured data, not prose. Early prototypes of
LLM-agent towns (Stanford's generative agents being the canonical example)
store memory as natural-language sentences and let the LLM re-read and
synthesize them. That works for *flavor* but is exactly why those systems
struggle to support hard mechanics like norms and economies at scale: you
cannot reliably query "does agent X currently believe agent Y is a thief"
out of a pile of sentences without another expensive LLM call.

Here, memory entries are typed, structured records. They CAN be rendered to
natural language (see `MemoryEntry.as_text`) for feeding into an LLM prompt
later, but the engine and other code never has to parse prose to know what
happened -- it just reads fields. This is what lets reputation, gossip, and
"what do I know about agent Z" all be answered with plain dict/list lookups
instead of round-tripping through a language model.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MemoryEntry:
    """A single remembered event, from one agent's point of view.

    Attributes:
        tick: simulation tick at which this was recorded.
        kind: short tag for the event type, e.g. "witnessed_trade",
            "was_gossiped_about", "cast_vote", "spoke_to".
        subject: the agent_id this memory is primarily *about* (often the
            other party in an interaction). None for purely environmental
            memories (e.g. "saw the harvest fail").
        data: freeform structured payload specific to `kind`. Kept as a
            plain dict rather than a dataclass-per-kind because the set of
            event kinds will keep growing (Phase 2+ will add many), and a
            dict avoids needing a new class for every new event type. The
            tradeoff is no static typing on `data`'s shape -- acceptable
            here because each producer/consumer of a given `kind` lives
            close together in the codebase and documents its own shape.
        salience: a rough 0-1 importance score. Used to decide what surfaces
            in a condensed "recent memory" summary (e.g. for an LLM prompt)
            without dumping the entire log every time. Not used by the
            rule-based decision layer in this version, but the field exists
            now so memory.py's shape doesn't need to change in Phase 2.
    """

    tick: int
    kind: str
    subject: str | None
    data: dict = field(default_factory=dict)
    salience: float = 0.5

    def as_text(self) -> str:
        """Render this entry as a short natural-language sentence.

        This is the ONLY place in memory.py that produces prose, and it
        exists solely so Phase 2's LLM prompt-builder has something to call.
        Keep this dumb and literal -- it is not the place to inject
        opinion or interpretation; that's the LLM's job downstream.
        """
        who = f" about {self.subject}" if self.subject else ""
        return f"[t{self.tick}] {self.kind}{who}: {self.data}"


class MemoryLog:
    """An append-only list of MemoryEntry objects belonging to one agent.

    Append-only is a deliberate constraint: agents should not be able to
    edit or delete their own memories. This matters for norm-enforcement --
    if an agent could quietly erase the memory of having been caught
    breaking a rule, "reputation" would be trivially gameable.
    """

    def __init__(self) -> None:
        """Create an empty memory log."""
        self._entries: list[MemoryEntry] = []

    def add(self, entry: MemoryEntry) -> None:
        """Append `entry` to this agent's memory. The only mutator on
        this class -- there is deliberately no remove/edit method (see
        class docstring on why append-only matters).
        """
        self._entries.append(entry)

    def recent(self, n: int = 10) -> list[MemoryEntry]:
        """Return the `n` most recent entries, oldest first."""
        return self._entries[-n:]

    def about(self, agent_id: str) -> list[MemoryEntry]:
        """Return all entries whose `subject` is `agent_id`.

        This is the core query for reputation: "what have I personally
        witnessed agent_id do?" Used by decision.py to let an agent's
        behavior toward others depend on direct experience, not just
        the town-wide reputation score (see agent.py's `reputation`
        dict, which is the *aggregated* signal; this is the *raw* one).
        """
        return [e for e in self._entries if e.subject == agent_id]

    def all(self) -> list[MemoryEntry]:
        """Return every entry ever recorded for this agent, oldest
        first. Mainly useful for full-history inspection/debugging or
        dumping a complete record to disk; `recent()` is the normal
        path for feeding an agent's perception/decision logic.
        """
        return list(self._entries)
