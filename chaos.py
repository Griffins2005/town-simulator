"""
chaos.py -- Political, economic, and cross-sector chaos.

Design principle: every mechanic in this module is built the same way as
everything else in this codebase -- a small, named function with a clear
trigger condition, called once per tick from engine.py, mutating state
through the SAME invariant the rest of the project follows (state changes
happen in named functions, never scattered inline; see world.py's
docstring for why this matters). Nothing here bypasses actions.py's
validated execution path for agent-initiated behavior -- repeal proposals
go through governance.py's existing propose/vote/enact pipeline exactly
like any other rule, and corruption/shocks/panics are clock-driven
housekeeping (like governance.tick or economy.apply_wealth_tax_if_due),
not agent actions, because no single agent "decides" a market shock any
more than they decide a curfew's enactment.

THE ACTUAL POINT OF THIS MODULE -- cross-sector intersection, not three
independent chaos generators bolted side by side:

    reputation (norms) --> bank runs (economy)
        Low town-wide average reputation triggers panic: agents'
        existing RuleBasedDecider trade logic doesn't know about this,
        so this module directly suppresses trade willingness during a
        bank run by injecting a temporary, visible crisis flag that
        decision.py's _respond_to_trade reads (see CHAOS_INTEGRATION
        note in decision.py).

    corruption (politics) --> reputation (norms) --> politics again
        A caught embezzler's reputation collapses publicly (not
        privately) -- everyone in town gets a memory entry about the
        specific agent, which is exactly the kind of public information
        a future vote should be able to use. Combined with official
        track record being legible, an embezzling official is set up
        to lose their NEXT proposal's vote, not just take a one-time hit.

    wealth concentration (economy) --> political instability (politics)
        A sustained high Gini coefficient raises the probability of a
        spontaneous "unrest" shock that can itself trigger a forced
        emergency proposal -- inequality becomes a political fact, not
        just a number nobody reacts to.

    factions (politics) --> voting behavior (politics) --> economy
        Repeated voting alignment between two agents nudges them into
        the same faction; faction-mates' votes become correlated on
        FUTURE proposals (a faction member is more likely to vote
        however their faction's average leaning goes), which is the
        difference between "independent per-proposal coin flips" (the
        original RuleBasedDecider) and an actual emergent political
        structure.

Severity: this module is deliberately NOT safety-railed against real
collapse. A bank run can genuinely freeze trade for many ticks. A
corruption scandal can genuinely crater an official's reputation to
near zero with no fast recovery. This was an explicit design choice
(see conversation history) -- chaos that can't actually hurt isn't
chaos, it's decoration. The stress-test harness (see stress_test.py)
is what verifies this produces interesting collapse states rather than
a boring, permanent flatline; see this module's "tuning constants" for
the knobs to adjust if a stress test finds the latter.
"""

from __future__ import annotations

import random

from agent import Agent
from memory import MemoryEntry
from world import World

# CORRECTED after a 1000-tick stress test: the original constants below
# (treasury-scaled base chance + flat town_hall/official bonuses, rolled
# INDEPENDENTLY by all 16 agents every single tick) produced 191
# corruption scandals over 1000 ticks -- roughly one every 5 ticks, not
# the rare, occasional event the design called for ("mostly random
# opportunity, sometimes amplified by town_hall presence or official
# status" -- not "background radiation"). The root cause: a per-agent
# probability that feels small in isolation (1-2%) becomes a near-
# certainty in aggregate once rolled by 16 agents every tick with no
# cooldown. The consequence was severe: each scandal's 0.35 reputation
# penalty arrived faster than REPUTATION_DECAY_RATE could repair it,
# so town-wide average reputation entered a permanent one-way decline
# (0.5 -> 0.148 over 1000 ticks) that triggered a bank_run crisis which
# then NEVER ended -- the same shape of bug as the original Phase-1
# reputation ratchet, just at the population-average level instead of
# per-agent. Fixed two ways: (1) rates lowered roughly 5-8x, and (2) a
# town-wide cooldown added (CORRUPTION_COOLDOWN_TICKS) so a scandal
# can't recur from anyone for a fixed window afterward, regardless of
# how the per-tick rate is tuned -- this is what actually guarantees
# "occasional," since rate alone is fragile to population size changes.
CORRUPTION_BASE_CHANCE_PER_TREASURY_UNIT = 0.00012
CORRUPTION_TOWN_HALL_BONUS = 0.003
CORRUPTION_OFFICIAL_BONUS_PER_PASSED_PROPOSAL = 0.002
CORRUPTION_OFFICIAL_BONUS_CAP = 0.01
CORRUPTION_SKIM_FRACTION = 0.4
CORRUPTION_REPUTATION_PENALTY = 0.35
CORRUPTION_COOLDOWN_TICKS = 25

MARKET_SHOCK_CHANCE_PER_TICK = 0.01
MARKET_SHOCK_SCARCITY_MULTIPLIER = 0.15
MARKET_SHOCK_ABUNDANCE_MULTIPLIER = 3.0

BANK_RUN_REPUTATION_TRIGGER = 0.40
BANK_RUN_REPUTATION_RECOVERY = 0.46
BANK_RUN_TAG = "bank_run"

SPECULATION_BUZZ_DECAY = 0.85
SPECULATION_BUZZ_PER_GOSSIP = 0.08
SPECULATION_MAX_PRICE_DISTORTION = 0.6

UNREST_GINI_TRIGGER = 0.55
UNREST_CHANCE_PER_TICK_ABOVE_TRIGGER = 0.02
UNREST_TAG = "unrest"
UNREST_REPUTATION_DAMPING = 0.02

FACTION_FORM_THRESHOLD_AGREEMENTS = 3
FACTION_VOTE_CORRELATION = 0.3

_buzz: dict = {}
_agreement_counts: dict = {}
# Tick of the most recent corruption scandal, or None if there hasn't
# been one yet. Used by apply_corruption_if_opportunity to enforce
# CORRUPTION_COOLDOWN_TICKS -- see that constant's docstring for why
# a cooldown, not just a low rate, is what actually keeps corruption
# "occasional" regardless of how many agents are rolling each tick.
_last_corruption_tick: int | None = None
# faction_id -> list of recent "yes"/"no" votes cast by ANY member of
# that faction, most recent last, capped at FACTION_LEAN_HISTORY
# entries. This is what get_faction_lean reads to compute a faction's
# current voting tendency -- kept as a short rolling history (not a
# single running average) so a faction's lean can shift over time as
# its membership's actual votes shift, rather than being permanently
# anchored to its very first few votes.
_faction_vote_history: dict = {}
FACTION_LEAN_HISTORY = 6


def apply_corruption_if_opportunity(world: World, agents: dict[str, Agent], rng: random.Random) -> None:
    """Called once per tick by engine.py. Each agent independently rolls
    against a corruption probability built from treasury size (the
    dominant term) plus small bonuses for being at town_hall right now
    or having an established political track record. On a hit, the
    agent skims CORRUPTION_SKIM_FRACTION of the CURRENT treasury into
    their own pocket.

    Enforces CORRUPTION_COOLDOWN_TICKS after any scandal -- see that
    constant's docstring for why a cooldown (not just a low per-tick
    rate) is necessary: with 16 agents independently rolling every
    tick, even a small per-agent rate compounds into a near-constant
    background event without an explicit "nothing else can happen for
    a while after a scandal" floor.

    This is deliberately NOT hidden from the rest of the town: a
    corruption event is logged publicly and gives every other agent a
    direct memory of "agent X embezzled," which is the actual point --
    a scandal that nobody can ever find out about isn't political
    chaos, it's a silent economic leak indistinguishable from a bug.

    Args:
        world: the simulation world (treasury, tick, event log).
        agents: the full agent registry.
        rng: an injected random.Random instance -- CORRECTED from an
            earlier version that called the global `random` module
            directly, which broke this codebase's seeded-reproducibility
            guarantee (every other randomized component, including
            RuleBasedDecider, takes an injected rng). Callers should
            pass the SAME instance used elsewhere in the same run (see
            Engine.__init__'s `rng` parameter) for genuinely
            reproducible chaos behavior.
    """
    global _last_corruption_tick
    if world.treasury <= 0:
        return
    if _last_corruption_tick is not None and world.tick - _last_corruption_tick < CORRUPTION_COOLDOWN_TICKS:
        return
    for agent in agents.values():
        chance = world.treasury * CORRUPTION_BASE_CHANCE_PER_TREASURY_UNIT
        if agent.location == "town_hall":
            chance += CORRUPTION_TOWN_HALL_BONUS
        official_bonus = min(
            CORRUPTION_OFFICIAL_BONUS_CAP,
            agent.official_track_record * CORRUPTION_OFFICIAL_BONUS_PER_PASSED_PROPOSAL,
        )
        chance += official_bonus
        if rng.random() >= chance:
            continue

        skimmed = round(world.treasury * CORRUPTION_SKIM_FRACTION, 4)
        world.treasury = round(world.treasury - skimmed, 4)
        agent.money = round(agent.money + skimmed, 4)
        agent.reputation = max(0.0, agent.reputation - CORRUPTION_REPUTATION_PENALTY)

        world.log_event("corruption_scandal", agent=agent.agent_id, skimmed=skimmed,
                         was_at_town_hall=(agent.location == "town_hall"),
                         official_track_record=agent.official_track_record)
        for other in agents.values():
            if other.agent_id != agent.agent_id:
                other.memory.add(MemoryEntry(world.tick, "witnessed_corruption_scandal",
                                              agent.agent_id, {"skimmed": skimmed}))
        _last_corruption_tick = world.tick
        return


def reset_corruption_cooldown() -> None:
    """Clear the corruption cooldown timestamp. Test/run isolation, same
    rationale as reset_buzz/reset_factions -- without this, a second
    simulation run in the same process would inherit the first run's
    _last_corruption_tick and could incorrectly suppress corruption for
    its own early ticks.
    """
    global _last_corruption_tick
    _last_corruption_tick = None


def apply_market_shocks_if_triggered(world: World, rng: random.Random) -> None:
    """Called once per tick by engine.py. With small independent
    probability, triggers a scarcity (blight) or abundance (bumper
    harvest) shock at a randomly chosen resource-bearing location,
    directly multiplying that location's current resource pool.

    Args:
        world: the simulation world (locations, event log).
        rng: an injected random.Random instance -- see
            apply_corruption_if_opportunity's docstring for why this
            must be injected rather than calling the global `random`
            module directly.
    """
    if rng.random() >= MARKET_SHOCK_CHANCE_PER_TICK:
        return
    resource_locations = [loc for loc in world.locations.values() if loc.resources]
    if not resource_locations:
        return
    loc = rng.choice(resource_locations)
    is_scarcity = rng.random() < 0.6
    multiplier = MARKET_SHOCK_SCARCITY_MULTIPLIER if is_scarcity else MARKET_SHOCK_ABUNDANCE_MULTIPLIER
    for resource_kind in list(loc.resources.keys()):
        loc.resources[resource_kind] = round(loc.resources[resource_kind] * multiplier, 3)
    world.log_event("market_shock", location=loc.name,
                     shock_kind="scarcity" if is_scarcity else "abundance", multiplier=multiplier)


def update_bank_run_state(world: World, agents: dict[str, Agent]) -> None:
    """Called once per tick by engine.py. Tracks town-wide average
    reputation and toggles the BANK_RUN_TAG crisis flag in
    world.active_crises using a hysteresis band (lower threshold to
    enter, higher to exit) so the flag doesn't flicker at the boundary.
    """
    if not agents:
        return
    avg_reputation = sum(a.reputation for a in agents.values()) / len(agents)
    was_active = BANK_RUN_TAG in world.active_crises
    if not was_active and avg_reputation < BANK_RUN_REPUTATION_TRIGGER:
        world.active_crises.add(BANK_RUN_TAG)
        world.log_event("crisis_started", crisis=BANK_RUN_TAG, avg_reputation=round(avg_reputation, 3))
    elif was_active and avg_reputation >= BANK_RUN_REPUTATION_RECOVERY:
        world.active_crises.discard(BANK_RUN_TAG)
        world.log_event("crisis_ended", crisis=BANK_RUN_TAG, avg_reputation=round(avg_reputation, 3))


def update_speculation_buzz(world: World, agents: dict[str, Agent]) -> dict:
    """Called once per tick by engine.py. Maintains a per-agent "buzz"
    score (decayed each tick, bumped by gossip events mentioning that
    agent) and returns the current snapshot for engine.py to attach to
    each agent's Perception.
    """
    for agent_id in list(_buzz.keys()):
        _buzz[agent_id] *= SPECULATION_BUZZ_DECAY
        if _buzz[agent_id] < 0.01:
            del _buzz[agent_id]

    for event in world.event_log:
        if event.get("tick") != world.tick or event.get("kind") != "gossip":
            continue
        about = event.get("about")
        if about:
            _buzz[about] = _buzz.get(about, 0.0) + SPECULATION_BUZZ_PER_GOSSIP

    return dict(_buzz)


def reset_buzz() -> None:
    """Clear speculation buzz. Test/run isolation, same rationale as
    economy.reset_offers() and governance.reset().
    """
    _buzz.clear()


def update_unrest_state(world: World, agents: dict[str, Agent], rng: random.Random) -> None:
    """Called once per tick by engine.py. Computes the current wealth
    Gini coefficient and, if sustained above UNREST_GINI_TRIGGER, rolls
    a small per-tick chance of starting an "unrest" crisis. While
    active, unrest applies a mild town-wide reputation drag.

    Args:
        world: the simulation world (active_crises, event log).
        agents: the full agent registry.
        rng: an injected random.Random instance -- see
            apply_corruption_if_opportunity's docstring for why.
    """
    if not agents:
        return
    moneys = sorted(a.money for a in agents.values())
    n = len(moneys)
    total = sum(moneys)
    if n == 0 or total == 0:
        gini = 0.0
    else:
        cumulative = sum((i + 1) * v for i, v in enumerate(moneys))
        gini = (2 * cumulative) / (n * total) - (n + 1) / n

    was_active = UNREST_TAG in world.active_crises
    if gini >= UNREST_GINI_TRIGGER:
        if not was_active and rng.random() < UNREST_CHANCE_PER_TICK_ABOVE_TRIGGER:
            world.active_crises.add(UNREST_TAG)
            world.log_event("crisis_started", crisis=UNREST_TAG, gini=round(gini, 3))
        if was_active:
            for agent in agents.values():
                agent.reputation = max(0.0, agent.reputation - UNREST_REPUTATION_DAMPING)
    elif was_active:
        world.active_crises.discard(UNREST_TAG)
        world.log_event("crisis_ended", crisis=UNREST_TAG, gini=round(gini, 3))


def update_factions(world: World, agents: dict[str, Agent]) -> None:
    """Called once per tick by engine.py, after governance.tick() has
    resolved any proposals closing this tick. Looks at THIS tick's
    vote_cast events and increments a same-side-agreement counter for
    every pair of agents who voted the same way on the same proposal.
    Once a pair crosses FACTION_FORM_THRESHOLD_AGREEMENTS, they're
    merged into a shared faction in world.factions.

    Also records every vote cast by a faction member into that
    faction's rolling vote history (see _faction_vote_history and
    get_faction_lean) -- done here, in the same pass over this tick's
    vote_cast events, rather than as a separate scan.
    """
    votes_this_tick = [e for e in world.event_log if e.get("tick") == world.tick and e.get("kind") == "vote_cast"]
    by_proposal: dict = {}
    for v in votes_this_tick:
        by_proposal.setdefault(v["proposal_id"], {})[v["by"]] = v["choice"]
        faction_id = world.factions.get(v["by"])
        if faction_id:
            history = _faction_vote_history.setdefault(faction_id, [])
            history.append(v["choice"])
            del history[:-FACTION_LEAN_HISTORY]  # keep only the most recent N

    for proposal_votes in by_proposal.values():
        agent_ids = list(proposal_votes.keys())
        for i in range(len(agent_ids)):
            for j in range(i + 1, len(agent_ids)):
                a_id, b_id = agent_ids[i], agent_ids[j]
                if proposal_votes[a_id] != proposal_votes[b_id]:
                    continue
                pair = tuple(sorted((a_id, b_id)))
                _agreement_counts[pair] = _agreement_counts.get(pair, 0) + 1
                if _agreement_counts[pair] >= FACTION_FORM_THRESHOLD_AGREEMENTS:
                    _merge_into_faction(world, a_id, b_id)


def get_faction_lean(world: World, agent_id: str) -> float:
    """Returns the recent yes-rate (0.0-1.0) of `agent_id`'s faction,
    or 0.5 (neutral -- no lean either way) if the agent has no faction
    yet or that faction has no voting history. Called by engine.py when
    building each agent's Perception (see Perception.faction_lean).
    """
    faction_id = world.factions.get(agent_id)
    if not faction_id:
        return 0.5
    history = _faction_vote_history.get(faction_id)
    if not history:
        return 0.5
    return sum(1 for v in history if v == "yes") / len(history)


def _merge_into_faction(world: World, a_id: str, b_id: str) -> None:
    """Merge two agents into the same faction. If either already has a
    faction, the other adopts it -- faction_id is whichever agent_id
    became the faction's identity first, not a separately-generated ID.
    """
    existing = world.factions.get(a_id) or world.factions.get(b_id)
    faction_id = existing or a_id
    if world.factions.get(a_id) != faction_id:
        world.factions[a_id] = faction_id
        world.log_event("faction_joined", agent=a_id, faction=faction_id)
    if world.factions.get(b_id) != faction_id:
        world.factions[b_id] = faction_id
        world.log_event("faction_joined", agent=b_id, faction=faction_id)


def reset_factions() -> None:
    """Clear faction-formation state. Test/run isolation -- does NOT
    clear world.factions itself (that belongs to the World instance the
    caller controls), only this module's internal agreement counts and
    faction vote-history.
    """
    _agreement_counts.clear()
    _faction_vote_history.clear()
