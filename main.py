"""
main.py -- Runnable demo. Builds a small town, runs it for N ticks, and
prints a human-readable trace of what happened.

This is intentionally a SCRIPT, not a library API -- its job is to prove
the engine works end-to-end and give you something to eyeball today. The
real "research instrument" usage (pause, inspect, replay) is exercised via
direct use of Engine/World in a notebook or REPL, not through this file.

Run: python3 main.py
"""

from __future__ import annotations

import random

from engine import Engine
from town_factory import build_agents, build_world

import chaos
import economy
import governance


NUM_AGENTS = 16          # within the 15-20 range discussed
NUM_TICKS = 60
SEED = 7                  # fixed seed -> fully reproducible run, important
# for the "research artifact" use case: same seed, same trace, every time.


def print_tick_summary(world: World, agents: dict, before_log_len: int) -> None:
    """Print only the events logged THIS tick (the slice of world.event_log
    added since `before_log_len`), rather than re-printing the whole
    history every time -- keeps the trace readable across 60+ ticks.
    """
    new_events = world.event_log[before_log_len:]
    if not new_events:
        return
    print(f"--- tick {world.tick - 1} ---")
    for e in new_events:
        kind = e["kind"]
        if kind in ("move", "work", "idle"):
            continue  # too frequent to be interesting in a printed trace
        detail = {k: v for k, v in e.items() if k not in ("tick", "kind")}
        print(f"  [{kind}] {detail}")


def main() -> None:
    """Build a fresh town from town_factory, run it for NUM_TICKS ticks
    with all-RuleBasedDecider agents, print a per-tick trace of notable
    events (excluding the high-frequency move/work/idle), then print
    final agent states and aggregate event-kind counts.
    """
    rng = random.Random(SEED)
    economy.reset_offers()
    governance.reset()
    chaos.reset_buzz()
    chaos.reset_factions()
    chaos.reset_corruption_cooldown()

    world = build_world()
    agents = build_agents(rng, NUM_AGENTS)
    engine = Engine(world, agents, rng=rng)

    print(f"Starting town of {NUM_AGENTS} agents for {NUM_TICKS} ticks (seed={SEED})\n")

    for _ in range(NUM_TICKS):
        before = len(world.event_log)
        engine.step()
        print_tick_summary(world, agents, before)

    print("\n--- final state ---")
    for agent_id, agent in sorted(agents.items()):
        print(f"  {agent_id} ({agent.persona.name}): money={agent.money:.2f} "
              f"inventory={agent.inventory} reputation={agent.reputation:.2f} "
              f"location={agent.location}")

    print(f"\nActive rules at end of run: {world.active_rules}")
    print(f"Total events logged: {len(world.event_log)}")

    print("\n--- aggregate stats (what actually happened) ---")
    from collections import Counter
    kind_counts = Counter(e["kind"] for e in world.event_log)
    for kind, count in kind_counts.most_common():
        print(f"  {kind}: {count}")

    reputations = [a.reputation for a in agents.values()]
    print(f"\nReputation spread: min={min(reputations):.2f} max={max(reputations):.2f} "
          f"(flat 0.50 for everyone would mean nothing moved it)")

    location_counts = Counter(a.location for a in agents.values())
    print(f"Final location distribution: {dict(location_counts)}")


if __name__ == "__main__":
    main()
