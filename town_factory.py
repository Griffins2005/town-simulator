"""
town_factory.py -- Shared town construction, used by both main.py (the
short demo) and stress_test.py (the long-run harness).

Factored out specifically to avoid two copies of "how do I build a town"
drifting apart over time -- a classic source of "the stress test passed
but the demo is testing something subtly different" bugs.
"""

from __future__ import annotations

import random

from agent import Agent, Persona
from decision import RuleBasedDecider
from world import Location, World

LOCATIONS = ["farm", "market", "town_hall", "tavern"]

FIRST_NAMES = [
    "Marcus", "Lena", "Tomas", "Aria", "Boris", "Nia", "Edwin", "Sofia",
    "Declan", "Maya", "Otto", "Priya", "Felix", "Yara", "Hugo", "Zara",
]


def build_world() -> World:
    """Construct the town's locations. `farm` is the only location with
    extractable resources in this minimal version -- enough to make
    scarcity bite (see economy.py) without modeling multiple resource
    chains, which would be premature complexity before the core mechanics
    are validated.
    """
    return World(locations=[
        Location("farm", resources={"food": 40.0}),
        Location("market", resources={}),
        Location("town_hall", resources={}),
        Location("tavern", resources={}),
    ])


def build_agents(rng: random.Random, num_agents: int) -> dict:
    """Construct `num_agents` agents with randomized-but-seeded traits,
    spread across locations, each with a RuleBasedDecider brain.

    Starting money/inventory is intentionally UNEQUAL (some agents start
    with more food, some with more money) -- a town where everyone starts
    identical has no reason to trade. Mild initial inequality is what
    gives the economy something to do from tick zero.
    """
    agents = {}
    for i in range(num_agents):
        agent_id = f"agent_{i:02d}"
        persona = Persona(
            name=FIRST_NAMES[i % len(FIRST_NAMES)],
            industriousness=rng.random(),
            generosity=rng.random(),
            sociability=rng.random(),
            rule_respect=rng.random(),
            risk_tolerance=rng.random(),
        )
        agent = Agent(
            agent_id=agent_id,
            persona=persona,
            location=rng.choice(LOCATIONS),
            money=round(rng.uniform(0, 20), 2),
            inventory={"food": round(rng.uniform(0, 3), 1)},
            decider=RuleBasedDecider(rng=rng),
        )
        agents[agent_id] = agent
    return agents
