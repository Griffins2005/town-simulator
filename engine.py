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
from decision import DecisionDraft, Intent, Perception
import actions
import analytics
import chaos
import decision_record
import economy
import geography
import governance
import faith
import inventions
import town_factory
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
        self.last_decision_records: list[dict] = []
        governance.ensure_office(self.world, self.agents)

    def step(self) -> None:
        """Advance the simulation by exactly one tick."""
        # Regenerate location resources BEFORE agents act this tick, so
        # a `work` action this tick sees the freshly-regenerated amount
        # rather than lagging a full tick behind. Added after the
        # 1000-tick stress test showed farm food draining to 0 and
        # staying there -- see economy.py's RESOURCE_REGEN_RATE docstring.
        economy.regenerate_resources(self.world)

        intents: dict = {}
        drafts: dict[str, DecisionDraft] = {}
        perceptions: dict = {}
        reused: dict[str, bool] = {}

        # Pass 1: perceive + decide (or reuse cached intent), for every agent.
        for agent_id, agent in self.agents.items():
            perception = self._build_perception(agent)
            perceptions[agent_id] = perception
            if self._should_think(agent_id, perception):
                if hasattr(agent.decider, "deliberate"):
                    draft = agent.decider.deliberate(agent_id, perception)
                else:
                    draft = DecisionDraft(intent=agent.decider.decide(agent_id, perception))
                self._cached_intent[agent_id] = draft
                self._last_thought_tick[agent_id] = self.world.tick
                reused[agent_id] = False
            else:
                cached = self._cached_intent.get(agent_id)
                if isinstance(cached, DecisionDraft):
                    draft = cached
                else:
                    draft = DecisionDraft(intent=cached or _IDLE_INTENT)
                reused[agent_id] = True
            drafts[agent_id] = draft
            intents[agent_id] = draft.intent

        # Pass 2: execute all intents, then write the six-stage record.
        records = []
        for agent_id, intent in intents.items():
            result = actions.execute(self.agents[agent_id], intent, self.world, self.agents)
            rec = decision_record.assemble(
                agent_id, self.world.tick, perceptions[agent_id],
                drafts[agent_id], result, reused[agent_id],
            )
            intel = rec.intelligence or {}
            if intel.get("source") == "fallback" and not reused[agent_id]:
                self.world.log_event(
                    "llm_fallback",
                    agent=agent_id,
                    reason=intel.get("fallback_reason"),
                    model_proposed=(intel.get("model_proposed") or {}).get("action"),
                    engine_action=intel.get("engine_action"),
                    accepted=intel.get("engine_accepted"),
                )
            records.append(rec.to_dict())
        self.last_decision_records = records

        # Clock-driven housekeeping: tally any proposals whose voting
        # window just closed. Runs after agent actions so a vote cast
        # earlier THIS tick is still counted before tallying.
        governance.tick(self.world, self.agents)
        governance.restore_expired_suspensions(self.world, self.agents)
        self._welcome_replacements()

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
        chaos.apply_market_shocks_if_triggered(self.world, self.rng, self.agents)

        # Corruption runs AFTER wealth_tax/demurrage/redistribution have
        # all settled this tick's treasury -- an embezzler skims from
        # the REAL, final treasury balance, not a stale pre-tax one
        # that's about to be redistributed out from under them anyway.
        chaos.apply_corruption_if_opportunity(self.world, self.agents, self.rng)

        # Religion clock: open a session, warm or cool piety, maybe
        # receive a convert. After corruption so a same-tick scandal
        # can cool that congregation; after governance.tick so a failed
        # festival vote or a welcome is already in the log.
        faith.tick(self.world, self.agents, self.rng)

        # Bank-run and unrest crisis state both read reputation/wealth
        # AFTER this tick's reputation decay and demurrage have already
        # applied, so the crisis check reflects where the town actually
        # ended up this tick, not a transient mid-tick value.
        chaos.update_bank_run_state(self.world, self.agents)
        chaos.update_unrest_state(self.world, self.agents, self.rng)
        chaos.tick_crises(self.world, self.agents, self.rng)
        economy.apply_crisis_pressure(self.world, self.agents, self.rng)
        governance.review_office(self.world, self.agents)

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
        chaos.update_factions(self.world, self.agents, self.rng)

        # Gossip-storm / notice-board update. Reads THIS tick's gossip
        # events (already in the log) and may emit influence_campaign
        # events that recorder/live_server pick up the same tick.
        chaos.update_influence_campaigns(self.world, self.agents)

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
        if perception.active_crises:
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
        if unvoted and perception.can_vote:
            return True
        if perception.lobby_targets and (
            perception.is_leader
            or getattr(perception, "is_faith_leader", False)
            or any(prop.get("proposed_by") == agent_id for prop in perception.open_proposals)
        ):
            return True
        if getattr(perception, "is_faith_leader", False) and getattr(perception, "worship_now", False):
            return True
        if getattr(perception, "office_vacant", False) and perception.can_vote:
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
        open_props = governance.open_proposals_snapshot(self.world, self.agents)
        leader = governance.town_leader(self.agents, self.world)
        notorious = [
            {"id": a.agent_id, "name": a.persona.name, "reputation": round(a.reputation, 3)}
            for a in sorted(self.agents.values(), key=lambda x: x.reputation)
            if not a.expelled and a.reputation < 0.28 and a.agent_id != agent.agent_id
        ][:4]
        newcomers = [
            {"id": a.agent_id, "name": a.persona.name}
            for a in self.agents.values()
            if not a.expelled and not a.voting_rights
        ]
        proposer_faith = None
        if open_props:
            sponsor = self.agents.get(open_props[0].get("proposed_by"))
            if sponsor:
                proposer_faith = getattr(sponsor.persona, "faith", None)
        own_faith = getattr(agent.persona, "faith", "unaffiliated")
        session_faith = faith.session_at(agent.location, self.world)
        faith_lead = faith.congregation_leader(self.agents, own_faith)
        congregation_leads = faith.leaders_public(self.agents)

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
            open_proposals=open_props,
            tick=self.world.tick,
            self_industriousness=agent.persona.industriousness,
            self_generosity=agent.persona.generosity,
            self_sociability=agent.persona.sociability,
            self_rule_respect=agent.persona.rule_respect,
            self_risk_tolerance=agent.persona.risk_tolerance,
            active_crises=set(self.world.active_crises),
            crisis_intensity=dict(self.world.crisis_intensity),
            self_faction=self.world.factions.get(agent.agent_id),
            self_faction_name=(
                self.world.faction_names.get(self.world.factions[agent.agent_id])
                if agent.agent_id in self.world.factions else None
            ),
            nearby_names={aid: self.agents[aid].persona.name for aid in others_here},
            public_headlines=analytics.build_public_headlines(self.world, self.agents),
            notice_board=list(self.world.notice_board),
            speculation_buzz=dict(self._current_buzz),
            faction_lean=chaos.get_faction_lean(self.world, agent.agent_id),
            enacted_proposals=governance.enacted_proposals_snapshot(),
            observed_problems=inventions.observed_problems(self.world, self.agents),
            invention_catalog=inventions.catalog_public(),
            known_inventions=list(self.world.inventions),
            trade_fairness_bonus=inventions.trade_fairness_bonus(self.world),
            can_vote=agent.can_vote(self.world.tick),
            expelled=agent.expelled,
            is_leader=bool(leader and leader.agent_id == agent.agent_id),
            town_leader_id=leader.agent_id if leader else None,
            town_leader_name=leader.persona.name if leader else None,
            notorious=notorious,
            lobby_targets=self._lobby_targets_for(agent, open_props, leader, faith_lead),
            newcomers=newcomers,
            self_faith=own_faith,
            self_faith_name=faith.faith_name(own_faith),
            self_piety=getattr(agent.persona, "piety", 0.2),
            self_wanderlust=getattr(agent.persona, "wanderlust", 0.35),
            nearby_faiths={aid: getattr(self.agents[aid].persona, "faith", "unaffiliated")
                           for aid in others_here},
            proposer_faith=proposer_faith,
            street_neighbors=geography.neighbors(agent.location, self.world),
            reachable=geography.reachable(agent.location, self.world),
            space=geography.space(agent.location),
            blocked_streets=sorted(geography.blocked_kinds(self.world)),
            worship_now=faith.worship_now(own_faith, self.world),
            faith_home=faith.faith_home(own_faith),
            congregation_size=len(faith.congregation(self.agents, own_faith)),
            congregation_here=len(faith.present_of_faith(
                self.agents, agent.location, own_faith)),
            session_faith=session_faith,
            session_present=len(faith.present_of_faith(
                self.agents, agent.location, session_faith)) if session_faith else 0,
            faith_census=faith.census(self.agents),
            festival_faith=faith.festival_faith(self.world, expire=False),
            water_blessing=bool(self.world.active_rules.get("water_blessing")),
            is_faith_leader=bool(faith_lead and faith_lead.agent_id == agent.agent_id),
            faith_leader_id=faith_lead.agent_id if faith_lead else None,
            faith_leader_name=faith_lead.persona.name if faith_lead else None,
            congregation_leaders=congregation_leads,
            self_livelihood=getattr(agent.persona, "livelihood", "laborer"),
            self_livelihood_name=economy.livelihood_name(getattr(agent.persona, "livelihood", None)),
            self_solvency=getattr(agent, "solvency", "ok"),
            can_vote_civic=agent.can_vote_on(self.world.tick, "curfew"),
            can_vote_economy=agent.can_vote_on(self.world.tick, "wealth_tax"),
            crisis_exposure={
                tag: round(economy.exposure(agent, tag), 2)
                for tag in (self.world.active_crises or set())
            },
            office_vacant=leader is None,
            succession_candidates=governance.succession_candidates(self.agents, self.world),
            impeach_justified=governance.impeach_justified(self.world, leader),
            leader_solvency=getattr(leader, "solvency", "ok") if leader else None,
        )

    def _welcome_replacements(self) -> None:
        """After an expulsion, seat a newcomer (franchise pending a welcome vote)."""
        for pending in governance.take_pending_welcomes():
            newcomer = town_factory.spawn_newcomer(
                self.rng, self.agents, self.world, replacing=pending.get("replacing"),
            )
            if newcomer is None:
                continue
            self.agents[newcomer.agent_id] = newcomer
            self._last_thought_tick[newcomer.agent_id] = -1

    def _lobby_targets_for(self, agent: Agent, open_props: list, leader, faith_lead=None) -> list:
        """Public roll of voters a political actor might still need to convince."""
        sponsor = any(prop.get("proposed_by") == agent.agent_id for prop in open_props)
        is_lead = bool(leader and leader.agent_id == agent.agent_id)
        is_faith_lead = bool(faith_lead and faith_lead.agent_id == agent.agent_id)
        rite_open = any(p.get("rule_type") in ("festival", "water_blessing") for p in open_props)
        deadlock = any(prop.get("deadlock") for prop in open_props)
        if not (sponsor or is_lead or (is_faith_lead and rite_open)
                or (agent.official_track_record and (deadlock or self.world.active_crises))):
            return []
        targets = []
        for prop in open_props:
            faith_rite = is_faith_lead and prop.get("rule_type") in ("festival", "water_blessing")
            if not (prop.get("deadlock") or self.world.active_crises
                    or prop.get("proposed_by") == agent.agent_id or faith_rite):
                continue
            lean = "yes" if prop.get("proposed_by") == agent.agent_id else (
                prop.get("votes", {}).get(agent.agent_id) or "yes"
            )
            for other in self.agents.values():
                if other.agent_id == agent.agent_id or not other.can_vote(self.world.tick):
                    continue
                last = (prop.get("votes") or {}).get(other.agent_id)
                if last == lean:
                    continue
                targets.append({
                    "id": other.agent_id,
                    "name": other.persona.name,
                    "location": other.location,
                    "last_vote": last,
                    "proposal_id": prop["proposal_id"],
                    "lean": lean,
                })
                if len(targets) >= 6:
                    return targets
        return targets
