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
from economy import assign_livelihood
from faith import assign_faith
from world import Location, World

LOCATIONS = [
    "farm", "workshop", "market", "town_hall", "tavern",
    "chapel", "park", "clinic", "bank", "homes",
]

FIRST_NAMES = [
    "Marcus Hale", "Lena Voss", "Tomas Reed", "Aria Cho", "Boris Klein",
    "Nia Okonkwo", "Edwin Pall", "Sofia Alvarez", "Declan Byrne", "Maya Singh",
    "Otto Berg", "Priya Shah", "Felix Marin", "Yara Haddad", "Hugo Costa",
    "Zara Malik",
]

# Used when the town expels someone and welcomes a replacement.
NEWCOMER_NAMES = [
    "Ivy Chen", "Rafi Okello", "Nora Lind", "Samir Qureshi",
    "Elise Moreau", "Kenji Sato", "Pilar Vargas", "Jonah Drake",
    "Amara Cole", "Theo Nilsen", "Laila Farouk", "Quinn Adler",
]

MAX_POPULATION = 22


def build_world() -> World:
    """Construct the town's locations. `farm` is the only location with
    extractable resources in this minimal version -- enough to make
    scarcity bite (see economy.py) without modeling multiple resource
    chains, which would be premature complexity before the core mechanics
    are validated.
    """
    return World(locations=[
        Location("farm", resources={"food": 40.0}),
        Location("workshop", resources={}),
        Location("market", resources={}),
        Location("town_hall", resources={}),
        Location("tavern", resources={}),
        Location("chapel", resources={}),
        Location("park", resources={}),
        Location("clinic", resources={}),
        Location("bank", resources={}),
        Location("homes", resources={}),
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
        faith_id, piety = assign_faith(i, rng)
        livelihood = assign_livelihood(i, faith_id)
        start_at = "farm" if livelihood == "farmer" and rng.random() < 0.55 else rng.choice(LOCATIONS)
        persona = Persona(
            name=FIRST_NAMES[i % len(FIRST_NAMES)],
            industriousness=rng.random(),
            generosity=rng.random(),
            sociability=rng.random(),
            rule_respect=rng.random(),
            risk_tolerance=rng.random(),
            faith=faith_id,
            piety=piety,
            livelihood=livelihood,
        )
        agent = Agent(
            agent_id=agent_id,
            persona=persona,
            location=start_at,
            money=round(rng.uniform(0, 20), 2),
            inventory={"food": round(rng.uniform(0, 3), 1)},
            decider=RuleBasedDecider(rng=rng),
        )
        agents[agent_id] = agent
    return agents


def next_agent_id(agents: dict) -> str:
    """Next unused agent_NN id, so newcomers don't collide with the seed town."""
    nums = []
    for agent_id in agents:
        try:
            nums.append(int(str(agent_id).split("_")[1]))
        except (IndexError, ValueError):
            continue
    return f"agent_{max(nums, default=-1) + 1:02d}"


def spawn_newcomer(rng: random.Random, agents: dict, world, replacing: str | None = None) -> Agent | None:
    """Welcome a new resident after an expulsion. Returns None if the town is full."""
    if len(agents) >= MAX_POPULATION:
        return None
    used_names = {a.persona.name for a in agents.values()}
    pool = [n for n in NEWCOMER_NAMES if n not in used_names] or NEWCOMER_NAMES
    agent_id = next_agent_id(agents)
    faith_id, piety = assign_faith(len(agents), rng)
    livelihood = assign_livelihood(len(agents), faith_id)
    persona = Persona(
        name=rng.choice(pool),
        industriousness=rng.random(),
        generosity=rng.random(),
        sociability=max(0.35, rng.random()),
        rule_respect=rng.random(),
        risk_tolerance=rng.random(),
        faith=faith_id,
        piety=piety,
        livelihood=livelihood,
    )
    agent = Agent(
        agent_id=agent_id,
        persona=persona,
        location=rng.choice(["market", "tavern", "town_hall", "homes", "park"]),
        money=round(rng.uniform(4, 12), 2),
        inventory={"food": round(rng.uniform(0.5, 2.0), 1)},
        decider=RuleBasedDecider(rng=rng),
        reputation=0.55,
        voting_rights=False,
    )
    world.log_event(
        "member_arrived",
        agent=agent_id,
        name=persona.name,
        replacing=replacing,
    )
    world.notice_board.append({
        "tick": world.tick,
        "from": "town_hall",
        "about": agent_id,
        "text": f"{persona.name} arrived — awaiting a welcome vote",
    })
    del world.notice_board[:-8]
    return agent
