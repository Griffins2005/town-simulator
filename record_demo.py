"""
record_demo.py -- Runs a town and saves a portable trace.

Run: python3 record_demo.py
Output: trace.json in the current directory (200 ticks, seed 7).

Rule-based agents only. Frames include metrics, decision records,
inventions, campaigns, and open proposals. The live Society Lab
(live_server.py) watches a town in real time; this file is for keeping
a finished run. LLM-backed agents later need no recorder.py changes --
agents_static already stores decider_kind.
"""

from __future__ import annotations

import random

from engine import Engine
from recorder import Recorder
from town_factory import build_agents, build_world

import chaos
import economy
import governance
import inventions

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
    chaos.reset_campaigns()
    chaos.reset_corruption_cooldown()
    inventions.reset()

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
