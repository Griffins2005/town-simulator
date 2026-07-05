"""
engine.py -- Orchestration: the step() loop.

Design principle: this is the ONLY module that calls Decider.decide() and
actions.execute(). It owns the order of operations for one tick:

    1. for each agent: build Perception (read-only snapshot)
    2. for each agent: ask its Decider for an Intent
    3. for each agent: execute that Intent via actions.py
    4. run clock-driven housekeeping (governance tallying, wealth tax,
       reputation decay, demurrage/treasury redistribution, and the
       chaos.py hooks -- see `step()` for the exact order)
    5. advance world.tick

Steps 2 and 3 are deliberately SEPARATE passes (decide-all, then
execute-all) rather than interleaved per-agent. This matters once agents
are LLM-driven: if agent A's decision could see the *already-executed*
result of agent B's action from later in the same tick, agents earlier in
iteration order would have a perception advantage purely from dict
ordering -- an artifact of implementation, not modeled behavior. Same-tick
simultaneity (nobody can react within a tick to something that hasn't
happened yet this tick) is the simplest fix and matches how most
agent-based models (and tabletop simulations) handle turn order fairly.

This module also owns the "sparse thinking" interrupt logic discussed
during design: NOT every agent calls its Decider every tick. An agent only
"thinks" (calls decide()) when something tick-worthy has happened to it.
This is the mechanism that makes a free/rate-limited LLM tier survive
contact with 15-20 agents in Phase 2, and it's exercised honestly here in
Phase 1 even though the rule-based Decider is cheap, so the interrupt logic
itself is validated before it's load-bearing for cost reasons.
"""

from __future__ import annotations

import random

from agent import Agent
from decision import Intent, Perception
import actions
import chaos
import economy
import governance
from world import World


# How often (in ticks) an agent thinks even with no interrupt, i.e. the
# periodic "reflection" cadence mentioned in the design discussion. Kept
# small here since the rule-based Decider is free; this is the knob to
# raise significantly (e.g. to "once per simulated day") once a real LLM
# is in the loop and cost/rate-limits matter.
PASSIVE_THINK_INTERVAL = 3

# Reputation decay rate, applied once per tick, pulling every agent's
# reputation toward the neutral baseline (0.5). Added after the 1000-tick
# stress test showed reputation hitting hard 0.0/1.0 caps by tick ~450
# and staying pinned there for the remaining 550 ticks -- a real
# degenerate equilibrium: small, RNG-driven differences in curfew
# violation compounded monotonically with nothing pulling values back
# toward center, since the only existing reputation mutations (trade
# completion: +, curfew violation: -) had no opposing decay term. This
# mirrors how real-world reputation/gossip actually works -- old
# information loses weight over time absent new evidence -- and it's
# the difference between "norms emerge from sustained behavior" (the
# goal) and "one early unlucky streak permanently brands an agent"
# (the bug). Deliberately gentle: decay is much slower than the
# trade/violation deltas, so sustained good or bad behavior still moves
# reputation meaningfully; it just stops being a one-way ratchet.
REPUTATION_DECAY_RATE = 0.002
REPUTATION_NEUTRAL = 0.5

# Shared idle-intent fallback, used for agents that haven't thought yet
# and have no cached intent. Defined once at module scope rather than
# constructed per-agent per-tick, and documents the implicit default
# behavior in one obvious place.
_IDLE_INTENT = Intent(action="idle", args={})


class Engine:
    """Owns the agent registry and drives the simulation forward one tick
    at a time via `step()`. Deliberately NOT a singleton/global -- nothing
    stops you from constructing two Engines (e.g. for A/B comparison runs
    with different parameters) in the same process.
    """

    def __init__(self, world: World, agents: dict, rng: random.Random | None = None) -> None:
        """
        Args:
            world: the World instance this engine will advance.
            agents: dict of agent_id -> Agent, the full town population.
                Each Agent's `.decider` determines how it thinks (see
                decision.Decider) -- this dict can mix RuleBasedDecider
                and LLMDecider agents freely (see main_llm.py for an
                example of exactly that).
            rng: an injected random.Random instance, passed through to
                chaos.py's randomized hooks (market shocks, corruption
                opportunity rolls). Defaults to a fresh, UNSEEDED
                Random() if not provided. CORRECTED: chaos.py originally
                called the global `random` module directly, which broke
                the seeded-reproducibility guarantee every other part of
                this codebase honors (RuleBasedDecider takes an injected
                rng; main.py documents `SEED = 7` as making runs "fully
                reproducible") -- caught by diffing two same-seed runs
                of main.py and finding them silently different. Callers
                that want reproducible chaos behavior (e.g. main.py,
                stress_test.py) should construct this with the SAME
                `random.Random` instance they use for `build_agents`,
                the same pattern decision.py's RuleBasedDecider already
                follows.
        """
        self.world = world
        self.agents = agents
        self.rng = rng or random.Random()
        # Tracks the tick each agent last actually called decide() on, so
        # PASSIVE_THINK_INTERVAL can be enforced per-agent.
        self._last_thought_tick: dict = {aid: -1 for aid in agents}
        # The Intent each agent decided last time it thought, replayed on
        # ticks where it doesn't re-think (its "current plan" continuing).
        self._cached_intent: dict = {}
        # Current speculation-buzz snapshot (see chaos.update_speculation_buzz),
        # refreshed at the end of every step(). Starts empty so tick 0's
        # Perception objects (built before step() has run any chaos
        # housekeeping) correctly show no buzz yet, rather than raising
        # on a missing attribute.
        self._current_buzz: dict = {}

    def step(self) -> None:
        """Advance the simulation by exactly one tick."""
        # Regenerate location resources BEFORE agents act this tick, so
        # a `work` action this tick sees the freshly-regenerated amount
        # rather than lagging a full tick behind. Added after the
        # 1000-tick stress test showed farm food draining to 0 and
        # staying there -- see economy.py's RESOURCE_REGEN_RATE docstring.
        economy.regenerate_resources(self.world)

        intents: dict = {}

        # Pass 1: perceive + decide (or reuse cached intent), for every agent.
        for agent_id, agent in self.agents.items():
            perception = self._build_perception(agent)
            if self._should_think(agent_id, perception):
                intent = agent.decider.decide(agent_id, perception)
                self._cached_intent[agent_id] = intent
                self._last_thought_tick[agent_id] = self.world.tick
            else:
                intent = self._cached_intent.get(agent_id) or _IDLE_INTENT
            intents[agent_id] = intent

        # Pass 2: execute all intents. Separated from decide (see module
        # docstring) so no agent's action this tick can be perceived by
        # another agent's decision this same tick.
        for agent_id, intent in intents.items():
            actions.execute(self.agents[agent_id], intent, self.world, self.agents)

        # Clock-driven housekeeping: tally any proposals whose voting
        # window just closed. Runs after agent actions so a vote cast
        # earlier THIS tick is still counted before tallying.
        governance.tick(self.world, self.agents)

        # If a wealth_tax rule is active and this tick is a collection
        # tick, collect and redistribute. Runs after proposal tallying
        # (so a tax enacted THIS tick could in principle start applying
        # as soon as its period next lands) and before passive demurrage
        # (so an agent isn't simultaneously hit by both in a way that's
        # hard to attribute -- ordering here just keeps the two distinct
        # in the event log; see governance.py's apply_wealth_tax_if_due
        # docstring for the full reasoning on why this redistributes
        # rather than just collecting).
        governance.apply_wealth_tax_if_due(self.world, self.agents)

        # Pull every agent's reputation a small step back toward
        # neutral. See REPUTATION_DECAY_RATE's docstring above for why
        # this exists -- without it, reputation is a one-way ratchet
        # that hits 0.0/1.0 and never recovers.
        for agent in self.agents.values():
            if agent.reputation > REPUTATION_NEUTRAL:
                agent.reputation = max(REPUTATION_NEUTRAL, agent.reputation - REPUTATION_DECAY_RATE)
            elif agent.reputation < REPUTATION_NEUTRAL:
                agent.reputation = min(REPUTATION_NEUTRAL, agent.reputation + REPUTATION_DECAY_RATE)

        # Apply mild passive wealth decay (demurrage) -- see economy.py's
        # DEMURRAGE_RATE docstring for why this exists: unlike reputation,
        # money has no natural "neutral" to decay toward, so this is a
        # flat percentage shrink on every balance rather than a pull
        # toward a baseline. Routed into world.treasury, not destroyed
        # (see apply_demurrage's docstring -- an earlier version
        # destroyed it outright, which a money-supply check caught as a
        # real bug). Deliberately the LAST piece of tick housekeeping,
        # after governance (a wealth tax enacted THIS tick should apply
        # before ALSO taking demurrage, not the reverse, so an agent
        # isn't double-charged in a way that's hard to reason about) and
        # after reputation decay (no ordering dependency between the two,
        # but keeping all "passive decay" steps together keeps this
        # section readable as one unit).
        economy.apply_demurrage(self.world, self.agents)

        # Periodically empty the treasury back out to the population --
        # this is what closes the loop demurrage opens. Independent of
        # (and on a different cadence from) any active wealth_tax rule's
        # own collect/redistribute cycle; see
        # economy.redistribute_treasury_if_due's docstring.
        economy.redistribute_treasury_if_due(self.world, self.agents)

        # --- chaos.py hooks (added when political/economic chaos and
        # cross-sector intersections were introduced) ---
        #
        # Market shocks first: independent of everything else, can run
        # any time. Placed here (not earlier, alongside resource
        # regeneration at the top of step()) so a shock's effect is
        # visible starting NEXT tick's work actions, not retroactively
        # altering what already happened this tick.
        chaos.apply_market_shocks_if_triggered(self.world, self.rng)

        # Corruption runs AFTER wealth_tax/demurrage/redistribution have
        # all settled this tick's treasury -- an embezzler skims from
        # the REAL, final treasury balance, not a stale pre-tax one
        # that's about to be redistributed out from under them anyway.
        chaos.apply_corruption_if_opportunity(self.world, self.agents, self.rng)

        # Bank-run and unrest crisis state both read reputation/wealth
        # AFTER this tick's reputation decay and demurrage have already
        # applied, so the crisis check reflects where the town actually
        # ended up this tick, not a transient mid-tick value.
        chaos.update_bank_run_state(self.world, self.agents)
        chaos.update_unrest_state(self.world, self.agents, self.rng)

        # Speculation buzz reads THIS tick's gossip events (already in
        # world.event_log from Pass 2's actions.execute calls above),
        # so it can run any time after that -- grouped here with the
        # other chaos housekeeping for readability.
        self._current_buzz = chaos.update_speculation_buzz(self.world, self.agents)

        # Faction formation reads THIS tick's vote_cast events, which
        # governance.tick() (above) may have just resolved into a
        # proposal_closed -- but vote_cast events themselves were
        # logged during Pass 2 (actions.execute), before governance.tick
        # ever ran, so faction updates work correctly regardless of
        # whether they're placed before or after governance.tick in this
        # function. Placed here, with the rest of chaos housekeeping,
        # for readability.
        chaos.update_factions(self.world, self.agents)

        self.world.tick += 1

    def _should_think(self, agent_id: str, perception: Perception) -> bool:
        """The sparse-thinking gate. An agent thinks this tick if ANY of:
          - it has never thought yet,
          - another agent is present at its location (a social interrupt),
          - it has a pending trade offer or an open proposal to vote on,
          - its periodic reflection interval has elapsed.

        Everything else (most ticks, for most agents) skips decide()
        entirely and replays the cached intent. This single method is
        what Phase 2 tunes hardest -- making it stricter (fewer interrupts)
        directly cuts LLM call volume, the dominant cost/rate-limit factor
        once Decider is LLM-backed.
        """
        last = self._last_thought_tick[agent_id]
        if last < 0:
            return True
        if perception.location_agents:
            return True
        if perception.pending_trade_offers:
            return True
        # Open proposals are only an interrupt if this agent HASN'T voted
        # on at least one of them yet. Without this check, every agent
        # re-thinks every tick for the entire 10-tick voting window even
        # though re-voting changes nothing once a vote is cast -- this
        # was a real bug (caught by inspecting `vote_cast` counts after
        # the first full run: 504 votes against only 5 proposals, ~16x
        # too many). The fix matters for more than cleanliness: once
        # Decider is LLM-backed, an unnecessary interrupt is a real,
        # billed API call. governance.py's `cast_vote` already silently
        # overwrites a repeat vote, so the *correctness* was never at
        # risk -- this is purely a cost/call-volume fix.
        unvoted = [p for p in perception.open_proposals
                   if agent_id not in p.get("votes", {})]
        if unvoted:
            return True
        if self.world.tick - last >= PASSIVE_THINK_INTERVAL:
            return True
        return False

    def _build_perception(self, agent: Agent) -> Perception:
        """Assemble the flat, serializable snapshot passed to decide().
        See decision.py's Perception docstring for why flatness matters.
        """
        loc = self.world.get_location(agent.location)
        others_here = [aid for aid, a in self.agents.items()
                        if a.location == agent.location and aid != agent.agent_id]
        recent = [m.as_text() for m in agent.memory.recent(5)]
        relationships = {oid: agent.relationship_with(oid) for oid in others_here}

        return Perception(
            self_id=agent.agent_id,
            self_money=agent.money,
            self_inventory=dict(agent.inventory),
            self_location=agent.location,
            self_reputation=agent.reputation,
            location_agents=others_here,
            location_resources=dict(loc.resources),
            active_rules=dict(self.world.active_rules),
            recent_memories=recent,
            relationships=relationships,
            pending_trade_offers=economy.offers_for(agent.agent_id),
            open_proposals=governance.open_proposals_snapshot(),
            tick=self.world.tick,
            self_industriousness=agent.persona.industriousness,
            self_generosity=agent.persona.generosity,
            self_sociability=agent.persona.sociability,
            self_rule_respect=agent.persona.rule_respect,
            self_risk_tolerance=agent.persona.risk_tolerance,
            active_crises=set(self.world.active_crises),
            self_faction=self.world.factions.get(agent.agent_id),
            speculation_buzz=dict(self._current_buzz),
            faction_lean=chaos.get_faction_lean(self.world, agent.agent_id),
            enacted_proposals=governance.enacted_proposals_snapshot(),
        )
