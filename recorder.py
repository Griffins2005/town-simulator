"""
recorder.py -- Captures a full per-tick trace of a simulation run to a
single JSON file, for the browser-based visualizer to load and replay.

Design principle: this module wraps an Engine from the OUTSIDE. It never
modifies world.py, agent.py, actions.py, economy.py, governance.py, or
engine.py -- the same seam discipline as decision.py's Decider protocol
(Phase 1) and llm_decider.py (Phase 2): the validated core stays
untouched, new capability is added by composition around it.

Output format (see `Recorder.save`): one JSON object with:
  - "locations": fixed map layout (name, x, y) for every location in the
    town -- assigned once, here, since World/Location have no spatial
    concept of their own (they're just named places with resource pools).
  - "agents_static": persona/trait info that doesn't change tick to
    tick (name, traits) -- kept separate from the per-tick frames so
    the trace file doesn't repeat it 1000 times.
  - "frames": one entry per recorded tick, each with every agent's
    current position/money/inventory/reputation/location and the new
    events logged that tick (a slice of world.event_log, not the
    cumulative whole -- keeps frame size from growing linearly with
    tick count).

This format is replay-only: the browser visualizer loads a finished
trace.json and plays it back. For live tick-by-tick streaming while the
simulation is actually running, see Phase 5's `live_server.py`, which
wraps this same `Recorder.step()` loop and broadcasts frames over SSE.
"""

from __future__ import annotations

import json

from agent import Agent
from engine import Engine
from world import World

# Fixed (x, y) layout for the four standard locations, in a 0-1000 by
# 0-1000 coordinate space the frontend can scale to whatever canvas size
# it wants. Arranged as a simple town square: farm and tavern on the
# outskirts, market and town_hall more central -- this is a cosmetic
# choice, not a simulation concern, which is exactly why it lives here
# in the recorder rather than in world.py. If town_factory.py's
# LOCATIONS list ever changes, add a matching entry here.
DEFAULT_LOCATION_LAYOUT = {
    "farm": {"x": 150, "y": 150},
    "market": {"x": 500, "y": 350},
    "town_hall": {"x": 500, "y": 650},
    "tavern": {"x": 850, "y": 250},
}


class Recorder:
    """Wraps an Engine, capturing a snapshot after every `step()` call.

    Usage:
        engine = Engine(world, agents)
        recorder = Recorder(engine, location_layout=DEFAULT_LOCATION_LAYOUT)
        for _ in range(num_ticks):
            recorder.step()   # advances the engine AND records a frame
        recorder.save("trace.json")
    """

    def __init__(self, engine: Engine, location_layout: dict | None = None) -> None:
        """
        Args:
            engine: the Engine instance to wrap and record. Recorder
                calls `engine.step()` itself (see `step()` below) rather
                than the caller calling it directly -- this guarantees a
                frame is captured for every tick that actually advances,
                with no risk of the two falling out of sync.
            location_layout: dict of location_name -> {"x": ..., "y":
                ...}. Defaults to DEFAULT_LOCATION_LAYOUT above. Must
                cover every location name in `engine.world.locations` --
                a missing entry raises clearly at record time rather
                than silently omitting that location from the map.
        """
        self.engine = engine
        self.location_layout = location_layout or DEFAULT_LOCATION_LAYOUT
        missing = set(engine.world.locations.keys()) - set(self.location_layout.keys())
        if missing:
            raise ValueError(
                f"location_layout is missing coordinates for: {sorted(missing)}. "
                f"Add entries to location_layout or DEFAULT_LOCATION_LAYOUT."
            )
        self.frames: list[dict] = []
        self._last_event_log_len = 0

    def step(self) -> dict:
        """Advance the wrapped engine by one tick, capture a frame, and
        return it.

        Captures events logged strictly DURING this tick (the slice of
        world.event_log added since the last call), not the cumulative
        log -- this is what keeps each frame's size roughly constant
        regardless of how many ticks have already been recorded, rather
        than every frame re-including every event since tick 0.

        Records the tick number BEFORE engine.step() advances the
        clock, so a frame's `tick` matches the tick during which its
        events actually happened (consistent with stress_test.py's
        0-indexed convention) -- engine.step() increments world.tick at
        the very end of each call, so reading it after stepping would
        label tick 0's events as tick 1.

        Returns the frame dict that was both appended to `self.frames`
        AND returned -- added so live_server.py can broadcast a tick's
        frame immediately upon stepping, without re-deriving it from
        `self.frames[-1]` or duplicating `_snapshot`'s logic. Existing
        callers that ignore the return value (e.g. record_demo.py's
        `for _ in range(NUM_TICKS): recorder.step()`) are unaffected.
        """
        tick_of_this_frame = self.engine.world.tick
        before = len(self.engine.world.event_log)
        self.engine.step()
        new_events = [self._normalize_event(e) for e in self.engine.world.event_log[before:]]
        frame = self._snapshot(new_events, tick_of_this_frame)
        self.frames.append(frame)
        return frame

    @staticmethod
    def _normalize_event(event: dict) -> dict:
        """Add an `agents_involved` field (a flat list of agent_ids) to
        a copy of `event`, without altering the original keys.

        Why this exists: world.log_event's callers (actions.py,
        economy.py, governance.py) use inconsistent key names for "which
        agent did this" depending on the event kind -- "agent" for
        move/work/speak/gossip, "by" for trade_rejected/vote_cast, "from_"
        and "to" for trade_completed, and NO agent reference at all for
        trade_failed_insufficient_funds or proposal_closed. That's fine
        for the simulation's own purposes (each call site knows its own
        shape), but a frontend trying to answer "show me everything
        agent_07 was involved in" would need to special-case every event
        kind's key naming -- fragile, and exactly the kind of
        visualization-specific normalization that shouldn't leak back
        into world.py/actions.py just to serve this one consumer.
        Normalizing here, once, at recording time, keeps that complexity
        out of both the validated engine AND the frontend.
        """
        involved = set()
        for key in ("agent", "by", "from_", "to", "proposed_by"):
            value = event.get(key)
            if isinstance(value, str) and value.startswith("agent_"):
                involved.add(value)
        for key in ("heard_by",):
            value = event.get(key)
            if isinstance(value, list):
                involved.update(v for v in value if isinstance(v, str) and v.startswith("agent_"))
        about = event.get("about")
        if isinstance(about, str) and about.startswith("agent_"):
            involved.add(about)
        return {**event, "agents_involved": sorted(involved)}

    def _snapshot(self, new_events: list[dict], tick: int) -> dict:
        """Build one frame: every agent's current visible state, plus
        the events that happened this specific tick.
        """
        world: World = self.engine.world
        agents_frame: dict[str, dict] = {}
        agent: Agent
        for agent_id, agent in self.engine.agents.items():
            agents_frame[agent_id] = {
                "location": agent.location,
                "money": round(agent.money, 2),
                "inventory": dict(agent.inventory),
                "reputation": round(agent.reputation, 3),
            }
        return {
            "tick": tick,
            "agents": agents_frame,
            "active_rules": dict(world.active_rules),
            "treasury": round(world.treasury, 2),
            "events": new_events,
        }

    def agents_static_snapshot(self) -> dict:
        """Build the {agent_id: {name, traits, decider_kind}} static
        info block -- factored out of `save()` so callers that need
        this WITHOUT writing a trace file (see live_server.py, which
        sends this once at connection time rather than ever touching
        disk) don't have to duplicate it.
        """
        agent: Agent
        return {
            agent_id: {
                "name": agent.persona.name,
                "traits": {
                    "industriousness": round(agent.persona.industriousness, 3),
                    "generosity": round(agent.persona.generosity, 3),
                    "sociability": round(agent.persona.sociability, 3),
                    "rule_respect": round(agent.persona.rule_respect, 3),
                    "risk_tolerance": round(agent.persona.risk_tolerance, 3),
                },
                # Recorded so the visualizer can show "this agent's mind
                # was an LLM" vs "rule-based" in its detail popup,
                # without the recorder needing to import llm_decider.py
                # (which requires the groq package) just to check this --
                # a plain class-name string is dependency-free.
                "decider_kind": type(agent.decider).__name__,
            }
            for agent_id, agent in self.engine.agents.items()
        }

    def save(self, path: str) -> None:
        """Write the full recorded trace to `path` as a single JSON
        document. See this module's docstring for the top-level shape.
        """
        trace = {
            "locations": self.location_layout,
            "agents_static": self.agents_static_snapshot(),
            "frames": self.frames,
        }
        with open(path, "w") as f:
            json.dump(trace, f)
