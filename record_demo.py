"""
record_demo.py -- Runs a town and saves a trace file for the visualizer.

Run: python3 record_demo.py
Output: trace.json (in the current directory)

This uses RuleBasedDecider for all agents -- the same validated, free,
deterministic setup as main.py -- run for longer (200 ticks) so the
visualizer has enough history to show real patterns (trades, governance,
reputation drift) when you scrub through it. Swapping in LLM-backed
agents later (see main_llm.py for the pattern) requires zero changes to
recorder.py or the trace format -- agents_static already records each
agent's decider_kind, and frames don't care what produced an action.
"""

from __future__ import annotations

import random

from engine import Engine
from recorder import Recorder
from town_factory import build_agents, build_world

import chaos
import economy
import governance

NUM_AGENTS = 16
NUM_TICKS = 200
SEED = 7
OUTPUT_PATH = "trace.json"


def main() -> None:
    """Build a town, run it for NUM_TICKS ticks while recording every
    frame, and save the result to OUTPUT_PATH for the visualizer.
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
    recorder = Recorder(engine)

    print(f"Recording {NUM_AGENTS} agents for {NUM_TICKS} ticks (seed={SEED})...")
    for _ in range(NUM_TICKS):
        recorder.step()

    recorder.save(OUTPUT_PATH)
    print(f"Saved trace to {OUTPUT_PATH} ({len(recorder.frames)} frames)")


if __name__ == "__main__":
    main()
