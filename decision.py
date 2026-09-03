"""
decision.py -- The swappable "brain" interface.

THIS IS THE SEAM. Everything else in the engine is built so that this file
is the only thing that needs to change when you move from Phase 1
(deterministic, free, rule-based agents) to Phase 2 (LLM-driven agents on
a free-tier provider).

Contract:
    A `Decider.decide(agent, perception) -> Intent` is the entire interface.
    - `perception` is a read-only snapshot of what this agent can currently
      sense (its own state, nearby agents, active rules, recent memories).
      It is assembled by engine.py, NOT by the Decider -- the Decider never
      reaches into World/Agent directly, so a future LLM-based Decider only
      ever needs the same flat, serializable `Perception` object as input.
      That flatness is exactly what will let Phase 2 turn `perception` into
      a prompt: a dict in, a dict out, with no hidden state in between.
    - `Intent` is a small structured result: an action name plus args. The
      Decider does NOT execute the action -- it only declares intent.
      actions.py validates and executes. This separation is what makes a
      malformed or "hallucinated" decision (inevitable once an LLM is in
      the loop) fail safely: an illegal Intent gets rejected by actions.py
      and logged, rather than corrupting state.

Why a rule-based stub now, not a mock LLM call: a mocked LLM call that
returns canned text would give false confidence -- it'd "work" without
exercising the real failure modes (malformed output, missing fields,
illegal actions) that Phase 2 must handle. A simple but genuinely
*reasoning* rule-based Decider exercises the full engine honestly: the
interrupt logic, the action validators, the governance and economy
mutations all run for real. When Phase 2 swaps in an LLM-backed Decider,
nothing downstream needs to change because the contract was exercised
honestly from day one.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class Perception:
    """Read-only snapshot of what one agent can sense this tick.

    Deliberately flat and JSON-serializable -- every field here should be
    able to drop directly into an LLM prompt template in Phase 2 with no
    further transformation. If you find yourself wanting to add something
    here that *isn't* flat/serializable, that's a signal it belongs in
    engine.py's perception-building logic instead, reduced to a flat fact
    first.
    """

    self_id: str
    self_money: float
    self_inventory: dict[str, float]
    self_location: str
    self_reputation: float
    location_agents: list[str]          # other agent_ids at the same location
    location_resources: dict[str, float]  # resources available to work/gather here
    active_rules: dict[str, object]
    recent_memories: list[str]          # pre-rendered text, see MemoryEntry.as_text
    relationships: dict[str, float]     # this agent's opinions of nearby agents
    pending_trade_offers: list[dict]    # offers addressed to this agent, if any
    open_proposals: list[dict]          # active governance proposals this agent can vote on
    tick: int
    # Self-knowledge: dispositional traits, in [0, 1]. NOTE on a design
    # correction -- these were originally left out of Perception on the
    # theory that "traits aren't sensory input." That was an
    # overcorrection: traits are knowledge the agent has of ITSELF (closer
    # in kind to self_id than to "what's visible in the room"), and a
    # rule-based Decider literally cannot act in-character without them.
    # They stay in Perception, not pulled from Agent directly, preserving
    # the actual invariant that matters: Decider never reaches past
    # Perception into live World/Agent state.
    self_industriousness: float = 0.5
    self_generosity: float = 0.5
    self_sociability: float = 0.5
    self_rule_respect: float = 0.5
    self_risk_tolerance: float = 0.5
    # Added alongside chaos.py: the set of currently active town-wide
    # crisis tags (e.g. {"bank_run"}, {"unrest"}), this agent's own
    # faction_id (None if not yet aligned with anyone), and the current
    # speculation-buzz snapshot (agent_id -> buzz score, see
    # chaos.update_speculation_buzz). These are the actual hooks that
    # let a Decider's behavior change in response to political/economic
    # chaos rather than the chaos existing purely as world-state nobody
    # reacts to -- see decision.py's CHAOS_INTEGRATION note on
    # RuleBasedDecider for where these get read.
    active_crises: set[str] = field(default_factory=set)
    crisis_intensity: dict[str, float] = field(default_factory=dict)
    self_faction: str | None = None
    speculation_buzz: dict[str, float] = field(default_factory=dict)
    # The agent's faction's recent yes-rate across its members' last
    # votes (0.5 if no faction or no voting history yet) -- the minimal
    # summary a Decider needs to let faction membership correlate
    # FUTURE votes, without needing full cross-agent vote history in
    # every Perception. See chaos.py's update_factions for how
    # factions form, and RuleBasedDecider._vote for how this gets used.
    faction_lean: float = 0.5
    # List of {"proposal_id": int, "rule_type": str} for every currently
    # enacted (not yet repealed) rule -- the minimal information an
    # agent needs to target a "repeal" proposal at a SPECIFIC enacted
    # rule, without exposing governance.py's internal
    # _enacted_keys_by_proposal index (which tracks raw active_rules
    # KEYS, an implementation detail Perception has no business leaking).
    enacted_proposals: list[dict] = field(default_factory=list)
    # Display name for self_faction (e.g. "Amber Guild"), separate from
    # the stable faction_id. None if the agent has no faction yet.
    self_faction_name: str | None = None
    # Nearby agent_id -> persona name, so dialogue can use names.
    nearby_names: dict[str, str] = field(default_factory=dict)
    # Recent public events an agent can gossip about, each
    # {"about": agent_id|None, "text": str}.
    public_headlines: list[dict] = field(default_factory=list)
    # Latest tavern notice-board slips (shared public channel).
    notice_board: list[dict] = field(default_factory=list)
    observed_problems: list[str] = field(default_factory=list)
    invention_catalog: list[dict] = field(default_factory=list)
    known_inventions: list[dict] = field(default_factory=list)
    trade_fairness_bonus: float = 0.0
    can_vote: bool = True
    expelled: bool = False
    is_leader: bool = False
    town_leader_id: str | None = None
    town_leader_name: str | None = None
    notorious: list = field(default_factory=list)
    lobby_targets: list = field(default_factory=list)
    newcomers: list = field(default_factory=list)


@dataclass
class Intent:
    """A declared action, not yet validated or executed.

    Attributes:
        action: one of the legal action names defined in actions.py's
            ACTION_REGISTRY (e.g. "move", "work", "trade_offer", "speak",
            "vote", "gossip", "idle"). An unrecognized action name is
            treated as "idle" by the engine and logged as a malformed
            intent -- this is the specific failure mode Phase 2 must
            handle gracefully, and it's already handled here in Phase 1
            so the path is exercised before it's load-bearing.
        args: action-specific keyword arguments, e.g. {"destination":
            "market"} for "move". Validated by actions.py, not here --
            decision.py's job is only to propose, never to validate.
        say: optional natural-language utterance, for "speak"/"gossip"
            actions or just flavor. Stored separately from `action`
            because dialogue and mechanical action are independent --
            an agent can speak WHILE moving, in principle.
    """

    action: str
    args: dict = field(default_factory=dict)
    say: str | None = None


@dataclass
class DecisionDraft:
    """Decider output before validation: the intent plus what was weighed."""

    intent: Intent
    retrieved_memories: list = field(default_factory=list)
    considered_actions: list = field(default_factory=list)


class Decider(Protocol):
    """Interface every brain implementation must satisfy."""

    def decide(self, agent_id: str, perception: Perception) -> Intent:
        """Given `agent_id` and a read-only `Perception` snapshot of
        what that agent currently senses, return the Intent the agent
        wants to act on this tick. Implementations must not mutate
        `perception` or reach into any live World/Agent state -- see
        this module's docstring for why that boundary matters.
        """
        ...


class RuleBasedDecider:
    """Phase 1 brain: simple, legible, trait-driven heuristics.

    This is not meant to produce deep or surprising behavior -- it's meant
    to be a honest, fully-deterministic-given-a-seed stand-in that exercises
    every part of the engine (movement, work, trade, voting, rule
    compliance/violation, gossip) so the mechanics can be validated before
    any LLM cost or nondeterminism enters the picture.

    Each branch below is intentionally simple and commented with WHY that
    heuristic was chosen, so it's clear which of these are "real" modeling
    decisions worth keeping even after Phase 2, versus placeholder logic.
    """

    def __init__(self, rng: random.Random | None = None) -> None:
        """
        Args:
            rng: an injected random.Random instance (rather than calling
                the global `random` module directly), so simulations can
                be made fully reproducible by seeding once at the top
                level -- essential for the "research artifact" use case,
                where you want to replay a run exactly. Defaults to a
                fresh, unseeded Random if not provided.
        """
        self.rng = rng or random.Random()

    def decide(self, agent_id: str, perception: Perception) -> Intent:
        return self.deliberate(agent_id, perception).intent

    def deliberate(self, agent_id: str, perception: Perception) -> DecisionDraft:
        """Priority chain plus a forensic draft of what was weighed."""
        p = perception
        considered: list[dict] = []
        memories = list(p.recent_memories)

        def take(intent: Intent, reason: str, weight: float) -> DecisionDraft:
            considered.append({"action": intent.action, "reason": reason, "weight": round(weight, 2)})
            return DecisionDraft(intent=intent, retrieved_memories=memories,
                                 considered_actions=list(considered))

        # Priority 1: respond to a pending trade offer if one exists.
        # Rationale: unresolved offers shouldn't sit forever; an agent
        # addresses its "inbox" before doing anything discretionary. This
        # mirrors a real heuristic humans use (resolve direct asks first).
        if p.pending_trade_offers:
            offer = p.pending_trade_offers[0]
            return take(self._respond_to_trade(agent_id, p, offer),
                        "resolve a pending trade offer first", 0.92)

        # Outcasts slink to the tavern and sour the room -- they cannot
        # vote, but gossip is how notoriety and welcome talk stay alive.
        if p.expelled:
            if p.self_location != "tavern" and self.rng.random() < 0.7:
                return take(Intent(action="move", args={"destination": "tavern"}),
                            "expelled — the tavern is all that's left", 0.84)
            if p.location_agents:
                other = self.rng.choice(p.location_agents)
                return take(Intent(
                    action="gossip",
                    args={"about": other, "tone": "accuse"},
                    say=f"this town threw me out and still wants {p.nearby_names.get(other, other)} to smile about it",
                ), "expelled — bitter gossip", 0.7)

        # Priority 2: vote on an open proposal this agent hasn't voted on
        # yet. Franchise-only -- a suspended or expelled resident is not
        # a hidden extra ballot. Filtered to UNVOTED proposals.
        unvoted = [prop for prop in p.open_proposals if agent_id not in prop.get("votes", {})]
        if p.can_vote and unvoted:
            return take(self._vote(agent_id, p, unvoted[0]),
                        "an open proposal still needs this vote", 0.88)

        lobby_draft = self._maybe_lobby(agent_id, p, take)
        if lobby_draft is not None:
            return lobby_draft

        sanction_draft = self._maybe_sanction(agent_id, p, take)
        if sanction_draft is not None:
            return sanction_draft

        # Priority 3: occasionally propose a rule -- either a NEW one
        # (curfew/wealth_tax) or, if something is already enacted, a
        # REPEAL of it. Both share the same gating (town_hall presence,
        # no open proposal already pending, low base probability) since
        # both are "I am about to spend political capital" actions.
        #
        # Choice between proposing-new and proposing-repeal: an agent
        # with LOW rule_respect and something already enacted leans
        # toward repeal (chafing under existing rules, plausible
        # grudge-like behavior) rather than always defaulting to
        # proposing something new -- this is what actually exercises
        # the repeal pipeline; without this branch, "repeal" would be a
        # fully-built but never-used pipeline, the same mistake Phase 1
        # made with curfew/wealth_tax before RuleBasedDecider was fixed
        # to actually propose them.
        if (p.self_location == "town_hall"
                and not p.open_proposals
                and self.rng.random() < p.self_rule_respect * 0.15):
            propose_tax = p.self_money < 10.0
            if propose_tax:
                return take(Intent(action="propose_rule", args={
                    "rule_type": "wealth_tax",
                    "rule_args": {"rate": 0.15, "threshold": 15.0, "period": 20},
                }), "propose a wealth tax from the hall", 0.7)
            return take(Intent(action="propose_rule", args={
                "rule_type": "curfew",
                "rule_args": {"after_tick_of_day": 18, "period": 24},
            }), "propose a curfew from the hall", 0.68)

        if (p.self_location == "town_hall"
                and not p.open_proposals
                and p.enacted_proposals
                and self.rng.random() < (1.0 - p.self_rule_respect) * 0.12):
            target = self.rng.choice(p.enacted_proposals)
            return take(Intent(action="propose_rule", args={
                "rule_type": "repeal",
                "rule_args": {"target_proposal_id": target["proposal_id"]},
            }), "propose repeal of an enacted rule", 0.64)

        crisis_draft = self._survive_crisis(agent_id, p, take)
        if crisis_draft is not None:
            return crisis_draft

        # Priority 3.5: wanderlust. Independent of everything below, an
        # agent has a flat per-tick chance to just move on regardless of
        # social/trade opportunities present. This is the actual fix for
        # a real bug the stress test surfaced: with circulation as the
        # LOWEST priority (below), an agent at market that always has
        # someone to talk to or trade with would essentially NEVER reach
        # the move branch -- speak/trade opportunities are self-renewing
        # as long as agents keep arriving, so "go elsewhere" never won a
        # priority contest it was always going to lose. The result was a
        # one-way feedback loop: market accumulates agents -> more
        # reasons to stay -> more agents arrive. location_entropy
        # collapsed to ~0.17 by tick 1000 in that run.
        #
        # WANDERLUST_CHANCE is checked BEFORE social/trade priorities
        # (not folded into the "else nothing else applies" tail) so it
        # can interrupt an otherwise-sticky agent. It's deliberately
        # small and not trait-weighted yet -- this is a blunt fix for a
        # structural priority-ordering problem, not a personality trait;
        # Phase 2's LLM agents should make this choice via actual
        # reasoning about competing needs, not a coin flip at all.
        WANDERLUST_CHANCE = 0.12
        if self.rng.random() < WANDERLUST_CHANCE:
            dest = self._next_stop(p.self_location)
            return take(Intent(action="move", args={"destination": dest}),
                        "wander to keep circulating", 0.55)

        # Priority 4: if other agents are present, maybe socialize
        # (gossip or speak) based on sociability trait -- now read from
        # real Perception data, not a hardcoded constant.
        if p.location_agents and self.rng.random() < p.self_sociability:
            return take(self._socialize(agent_id, p), "talk while others are here", 0.6)

        # Priority 5: initiate a trade if standing at the market with a
        # surplus to offer. "Surplus" is defined relative to a fixed
        # comfort threshold rather than by inspecting anyone else's
        # inventory -- an agent has no legitimate way to see another
        # agent's private holdings (see module docstring on Perception),
        # so it offers speculatively, the way a real market stall doesn't
        # know who's buying before someone walks up. Gated by generosity:
        # more generous agents trade more readily / ask for less in return.
        FOOD_COMFORT_THRESHOLD = 2.0
        food_held = p.self_inventory.get("food", 0.0)
        if (p.self_location == "market" and p.location_agents
                and food_held > FOOD_COMFORT_THRESHOLD):
            return take(self._initiate_trade(agent_id, p, food_held),
                        "offer surplus food at market", 0.58)

        # Priority 6: economic behavior -- work if resources are available
        # here and the agent leans industrious.
        if p.location_resources and self.rng.random() < p.self_industriousness:
            return take(Intent(action="work", args={}), "work available resources", 0.5)

        # Priority 6.5: propose or adopt an invention. The agent only
        # NAMES a catalog kind; inventions.py decides whether knowledge,
        # materials, and location make construction legal.
        match = next((item for item in p.invention_catalog
                      if item.get("problem") in p.observed_problems), None)
        if (match and p.self_location in ("workshop", "farm", "market", "town_hall")
                and p.self_industriousness >= 0.4
                and p.self_inventory.get("food", 0) >= 1.0
                and p.self_money >= 2.0
                and self.rng.random() < p.self_industriousness * 0.45):
            considered.append({"action": "invent", "reason": f"address {match['problem']}", "weight": 0.52})
            return take(Intent(action="invent", args={"invention_kind": match["kind"]}),
                        f"propose {match['name']} to address {match['problem']}", 0.52)
        adoptable = [inv for inv in p.known_inventions if agent_id not in inv.get("adopters", [])]
        if adoptable and self.rng.random() < 0.18:
            inv = adoptable[0]
            return take(Intent(action="adopt_invention", args={"invention_id": inv["id"]}),
                        f"adopt {inv.get('name')}", 0.42)

        # Priority 7: circulate. Rather than a one-way trip to "market"
        # that never returns (the original version's bug -- the whole
        # town piled into one room and stayed), agents cycle through the
        # fixed location loop. This is a deliberately crude stand-in for
        # "go where my day takes me" -- the kind of judgment Phase 2's
        # LLM should own outright -- but it's enough to keep the town's
        # population spatially distributed, which both governance
        # (location-gated proposing) and economy (market needs people
        # WITHOUT surplus arriving too, to be worth trading with) depend on.
        dest = self._next_stop(p.self_location)
        return take(Intent(action="move", args={"destination": dest}),
                    "nothing else applied — circulate", 0.2)

    # -- helpers -----------------------------------------------------
    # These read trait-ish info off the perception object's relationships/
    # reputation rather than the Agent directly, on purpose: the Decider
    # must only ever see what's in `Perception`, never the live Agent/World,
    # to keep the seam honest for Phase 2.

    _CIRCUIT = ["farm", "workshop", "market", "tavern", "town_hall"]

    def _next_stop(self, current: str) -> str:
        """Fixed circulation order. A real agent's reason to be somewhere
        is a research question for Phase 2 (an LLM agent could reason
        "I'm low on food, I should go to the farm" using self_inventory
        from Perception); this just guarantees the town doesn't collapse
        into a single room, which is a prerequisite for governance and
        economy to have anyone to act on.
        """
        if current not in self._CIRCUIT:
            return self._CIRCUIT[0]
        idx = self._CIRCUIT.index(current)
        return self._CIRCUIT[(idx + 1) % len(self._CIRCUIT)]

    def _survive_crisis(self, agent_id: str, p: Perception, take):
        """When the town is in a crisis, residents fight it — they do not idle.

        Famine: go work the farm, invent a pump, or beg/gossip for food.
        Unrest: go to the hall, tax the rich, or rally neighbors.
        Bank run: hoard, refuse strangers, talk people down at the tavern.
        Intensity scales how urgently they abandon the usual day.
        """
        if not p.active_crises:
            return None
        intensity = p.crisis_intensity or {}
        mag = max((intensity.get(tag, 0.5) for tag in p.active_crises), default=0.5)
        if self.rng.random() > min(0.92, 0.45 + mag * 0.55):
            return None
        if "famine" in p.active_crises or (
            "farm" in (p.location_resources or {}) and (p.location_resources or {}).get("food", 1) < 2
        ):
            food = p.self_inventory.get("food", 0.0)
            if p.self_location != "farm" and food < 2.5:
                return take(Intent(action="move", args={"destination": "farm"}),
                            "famine — run to the farm", 0.94)
            if p.self_location == "farm" and p.location_resources:
                return take(Intent(action="work", args={}),
                            "famine — harvest what is left", 0.93)
            pump = next((item for item in p.invention_catalog
                         if item.get("kind") == "water_pump"), None)
            if pump and p.self_location in ("workshop", "farm") and p.self_money >= 2:
                return take(Intent(action="invent", args={"invention_kind": "water_pump"}),
                            "famine — build a water pump", 0.9)
            if p.location_agents:
                other = self.rng.choice(p.location_agents)
                return take(Intent(action="gossip", args={
                    "about": other, "to": other,
                    "say": "the stores are failing — we have to work the farm",
                }), "famine — warn a neighbor", 0.82)

        if "unrest" in p.active_crises:
            if p.self_location != "town_hall" and self.rng.random() < 0.7:
                return take(Intent(action="move", args={"destination": "town_hall"}),
                            "unrest — go demand a vote", 0.9)
            if p.self_location == "town_hall" and not p.open_proposals:
                return take(Intent(action="propose_rule", args={
                    "rule_type": "wealth_tax",
                    "rule_args": {"rate": 0.2, "threshold": 12.0, "period": 15},
                }), "unrest — tax the rich before the town splits", 0.88)
            if p.location_agents and (p.is_leader or p.self_sociability > 0.55):
                other = self.rng.choice(p.location_agents)
                return take(Intent(action="speak", args={"to": other},
                                   say="walk with me to the hall — I need your vote before this splits us"),
                            "unrest — the leader works the room", 0.82)
            if p.location_agents:
                other = self.rng.choice(p.location_agents)
                return take(Intent(action="speak", args={"to": other},
                                   say="we either pass a law tonight or this gets worse"),
                            "unrest — rally whoever is here", 0.8)

        if "bank_run" in p.active_crises:
            if p.self_location != "tavern" and self.rng.random() < 0.55:
                return take(Intent(action="move", args={"destination": "tavern"}),
                            "bank run — find people before trust dies", 0.86)
            if p.location_agents:
                other = self.rng.choice(p.location_agents)
                line = ("hold your coin — I will not trade in this panic"
                        if p.self_risk_tolerance < 0.45
                        else "if we stop trading the town starves anyway")
                return take(Intent(action="speak", args={"to": other}, say=line),
                            "bank run — talk someone through the panic", 0.8)
        return None

    # Reference price: what a "fair" price-per-unit-of-food looks like,
    # used by the RECEIVING side of a trade to judge an offer. This must
    # match the CENTER of the range _initiate_trade's price_per_unit can
    # produce (1.5 to 3.0), so a generous seller's price reads as a good
    # deal and a stingy seller's price reads as a bad one -- without this
    # reference, acceptance can't respond to price at all. This was the
    # actual root cause behind a wealth-concentration artifact found by
    # the 1000-tick stress test: low-generosity (high-price) agents were
    # accumulating money fastest, because `_respond_to_trade` previously
    # ignored the offer's price entirely and accepted/rejected on a flat
    # trait-weighted coin flip. A real market needs price discipline on
    # the buying side, or sellers have no reason not to charge the
    # maximum -- this constant is the minimal version of that discipline.
    FAIR_PRICE_PER_UNIT = 2.25  # midpoint of _initiate_trade's [1.5, 3.0] range

    def _respond_to_trade(self, agent_id: str, p: Perception, offer: dict) -> Intent:
        """Decide whether to accept or reject a pending trade offer
        addressed to this agent, based on price fairness relative to
        FAIR_PRICE_PER_UNIT (adjusted by speculation buzz about the
        OFFERING agent, see CHAOS_INTEGRATION below), with tolerance
        widened by self_generosity -- and suppressed altogether during
        a bank-run crisis.

        Returns a "trade_accept" or "trade_reject" Intent for
        `offer["offer_id"]`.
        """
        # CHAOS_INTEGRATION: bank run (norms/reputation -> economy).
        # When town-wide trust has collapsed (chaos.update_bank_run_state
        # has set BANK_RUN_TAG), agents become reluctant to trade at all
        # -- a real, if crude, model of panic: when you don't trust
        # ANYONE's reputation, you stop transacting even with someone
        # who's never personally wronged you. Reject outright with high
        # probability rather than evaluating price at all; a small
        # chance of accepting survives so the bank run doesn't become an
        # absolute, mechanical freeze (real panics have holdouts).
        if "bank_run" in p.active_crises and self.rng.random() < 0.85:
            return Intent(action="trade_reject", args={"offer_id": offer["offer_id"]})

        # Evaluate price fairness directly, rather than ignoring the
        # offer's content. `offer` shape: {"from": ..., "give": {item:
        # qty}, "want": {item: qty}} from this RECEIVER's point of view
        # (the receiver would give `want` and get `give` -- see
        # economy.py's resolve_offer, where `actor` is the receiver).
        give = offer.get("give", {})
        want = offer.get("want", {})
        food_offered = give.get("food", 0.0)
        money_asked = want.get("money", 0.0)

        if food_offered <= 0:
            # Nothing of substance being offered to receiver (e.g. a
            # pure money-for-money or malformed offer) -- reject rather
            # than risk an unintended transfer. Fails safe.
            return Intent(action="trade_reject", args={"offer_id": offer["offer_id"]})

        price_per_unit = money_asked / food_offered
        # How much worse than fair this offer is, as a ratio. >1 means
        # overpriced; <1 means a good deal. Generosity widens how much
        # overpricing a receiver will still tolerate -- a generous agent
        # gives the seller more benefit of the doubt -- but no amount of
        # generosity makes an arbitrarily extortionate price acceptable,
        # which is exactly the missing discipline the stress test exposed.
        price_ratio = price_per_unit / self.FAIR_PRICE_PER_UNIT

        # CHAOS_INTEGRATION: speculation (rumor -> perceived fair price).
        # If the OFFERING agent has been the subject of recent gossip
        # (high buzz, see chaos.update_speculation_buzz), this receiver's
        # judgment of "is this a fair price" gets distorted in either
        # direction depending on sign -- modeled simply here as buzz
        # making a price seem WORSE than it is (rumor breeds suspicion of
        # a deal, even a fair one), capped by
        # SPECULATION_MAX_PRICE_DISTORTION so a single rumor can't make
        # every price look infinitely bad.
        offering_agent = offer.get("from")
        buzz = p.speculation_buzz.get(offering_agent, 0.0) if offering_agent else 0.0
        distortion = min(buzz, 0.6)  # mirrors chaos.SPECULATION_MAX_PRICE_DISTORTION
        price_ratio *= (1.0 + distortion)

        tolerance = 1.0 + (p.self_generosity * 0.5) - p.trade_fairness_bonus
        if price_ratio <= tolerance:
            return Intent(action="trade_accept", args={"offer_id": offer["offer_id"]})
        return Intent(action="trade_reject", args={"offer_id": offer["offer_id"]})

    def _initiate_trade(self, agent_id: str, p: Perception, food_held: float) -> Intent:
        """Construct a speculative trade_offer of surplus food (above
        FOOD_COMFORT_THRESHOLD) to a randomly chosen agent at the same
        location, priced inversely to this agent's generosity (more
        generous sellers charge less per unit).

        Args:
            agent_id: this agent's id (used only for signature symmetry
                with the other helper methods; not otherwise read here).
            p: this agent's current Perception, used to read
                self_generosity and the list of other agents present.
            food_held: this agent's current food inventory, used to
                compute the offerable surplus.

        Returns a "trade_offer" Intent.
        """
        target = self.rng.choice(p.location_agents)
        # Offer a portion of surplus food for a modest amount of money.
        # More generous agents ask for less money per unit of food --
        # a concrete, legible way for `generosity` to actually show up
        # in the ledger rather than just flavoring dialogue.
        FOOD_COMFORT_THRESHOLD = 2.0
        surplus = food_held - FOOD_COMFORT_THRESHOLD
        offer_amount = round(min(surplus, 2.0), 1)
        price_per_unit = 3.0 - (p.self_generosity * 1.5)  # range ~[1.5, 3.0]
        ask_price = round(offer_amount * price_per_unit, 2)
        return Intent(
            action="trade_offer",
            args={"to": target, "give": {"food": offer_amount}, "want": {"money": ask_price}},
            say=f"Selling {offer_amount} food for {ask_price}.",
        )

    def _vote(self, agent_id: str, p: Perception, proposal: dict) -> Intent:
        """Cast a yes/no vote on `proposal`. Blends this agent's own
        self_rule_respect disposition with its faction's recent voting
        tendency (CHAOS_INTEGRATION below) -- a faction member doesn't
        vote purely independently once they've actually joined one.

        Returns a "vote" Intent for `proposal["proposal_id"]`.
        """
        # Vote yes more often when this agent's own rule_respect is high
        # -- a legible, trait-grounded baseline. This is still a crude
        # stand-in for real political reasoning (Phase 2's LLM should
        # weigh the SPECIFIC proposal's content, who proposed it,
        # self-interest, etc.) but at least now it's the agent's own
        # disposition driving the vote, not an unexplained constant.
        base_yes_probability = p.self_rule_respect

        # CHAOS_INTEGRATION: factions (repeated voting alignment ->
        # correlated future votes). If this agent has joined a faction
        # (self_faction is set), blend in the faction's recent yes-rate
        # (faction_lean) at FACTION_VOTE_CORRELATION weight -- this is
        # the actual mechanism that turns "agents who happened to agree
        # a few times" into "a voting bloc that moves together going
        # forward." An agent with no faction yet takes the `else` branch
        # below and votes purely on its own disposition, with no
        # correlation pressure applied at all -- the blend only ever
        # runs for agents who have actually joined a faction.
        if p.self_faction:
            from chaos import FACTION_VOTE_CORRELATION
            yes_probability = (
                (1 - FACTION_VOTE_CORRELATION) * base_yes_probability
                + FACTION_VOTE_CORRELATION * p.faction_lean
            )
        else:
            yes_probability = base_yes_probability

        target = (proposal.get("rule_args") or {}).get("target_agent")
        if target == agent_id:
            yes_probability = 0.05
        elif proposal.get("rule_type") == "expel" and any(n.get("id") == target for n in p.notorious):
            yes_probability = min(0.95, yes_probability + 0.28)
        elif proposal.get("rule_type") == "welcome" and any(n.get("id") == target for n in p.newcomers):
            yes_probability = min(0.95, yes_probability + 0.2)
        elif proposal.get("emergency") or (proposal.get("vote_rules") or {}).get("emergency"):
            yes_probability = min(0.9, yes_probability + 0.12)
        if p.is_leader and proposal.get("proposed_by") == agent_id:
            yes_probability = min(0.95, yes_probability + 0.15)

        vote = "yes" if self.rng.random() < yes_probability else "no"
        return Intent(action="vote", args={"proposal_id": proposal["proposal_id"], "choice": vote})

    def _maybe_lobby(self, agent_id: str, p: Perception, take):
        """Leader, sponsor, or official works the room one name at a time."""
        if p.expelled or not p.lobby_targets:
            return None
        deadlock = any(prop.get("deadlock") for prop in p.open_proposals)
        sponsor = any(prop.get("proposed_by") == agent_id for prop in p.open_proposals)
        if not (p.is_leader or sponsor or deadlock or p.active_crises):
            return None
        if not (p.is_leader or sponsor) and not deadlock:
            return None
        here = [t for t in p.lobby_targets if t.get("id") in p.location_agents]
        pick = here[0] if here else p.lobby_targets[0]
        if pick.get("id") not in p.location_agents:
            dest = pick.get("location") or "town_hall"
            if dest != p.self_location:
                return take(Intent(action="move", args={"destination": dest}),
                            "lobby — find the next vote", 0.8)
        name = pick.get("name") or pick.get("id")
        lean = pick.get("lean") or "yes"
        line = (f"{name}, I need your {lean} on this — the hall is split"
                if deadlock else
                f"{name}, walk this through with me. Vote {lean}.")
        return take(Intent(
            action="lobby",
            args={"to": pick["id"], "proposal_id": pick["proposal_id"], "lean": lean},
            say=line,
        ), "lobby one member at a time", 0.83)

    def _maybe_sanction(self, agent_id: str, p: Perception, take):
        """Notorious names get a suspend or expel motion; newcomers get a welcome."""
        if p.self_location != "town_hall" or p.open_proposals or not p.can_vote:
            return None
        if p.newcomers and self.rng.random() < 0.45:
            guest = p.newcomers[0]
            return take(Intent(action="propose_rule", args={
                "rule_type": "welcome",
                "rule_args": {"target_agent": guest["id"]},
            }), f"welcome {guest.get('name') or guest['id']} onto the roll", 0.72)
        if not p.notorious:
            return None
        mark = p.notorious[0]
        if mark.get("id") == agent_id:
            return None
        civic = p.is_leader or p.self_rule_respect > 0.4
        if not civic and self.rng.random() > 0.35:
            return None
        if mark.get("reputation", 1) < 0.18 and (p.is_leader or p.self_rule_respect > 0.5):
            return take(Intent(action="propose_rule", args={
                "rule_type": "expel",
                "rule_args": {"target_agent": mark["id"]},
            }), f"expel notorious {mark.get('name') or mark['id']}", 0.76)
        return take(Intent(action="propose_rule", args={
            "rule_type": "suspend_vote",
            "rule_args": {"target_agent": mark["id"]},
        }), f"suspend the vote of {mark.get('name') or mark['id']}", 0.7)

    def _socialize(self, agent_id: str, p: Perception) -> Intent:
        """Gossip with a point: scandal, expulsion talk, welcome, or a headline."""
        other = self.rng.choice(p.location_agents)
        other_name = p.nearby_names.get(other, other)
        headlines = [h for h in p.public_headlines if h.get("about") and h.get("about") != agent_id]

        if p.newcomers and self.rng.random() < 0.35:
            guest = self.rng.choice(p.newcomers)
            return Intent(
                action="gossip",
                args={"about": guest["id"], "tone": "welcome"},
                say=f"have you met {guest.get('name') or guest['id']}? the town could use a new face",
            )
        if p.notorious and self.rng.random() < 0.22:
            mark = self.rng.choice(p.notorious)
            mark_name = mark.get("name") or mark["id"]
            if mark.get("reputation", 1) < 0.18:
                return Intent(
                    action="gossip",
                    args={"about": mark["id"], "tone": "expel"},
                    say=f"{mark_name} has become a stain — I say we expel them before the next vote",
                )
            return Intent(
                action="gossip",
                args={"about": mark["id"], "tone": "scandal"},
                say=f"did you hear the scandal around {mark_name}? I would not trust their ballot",
            )
        if p.town_leader_name and p.active_crises and self.rng.random() < 0.35:
            return Intent(
                action="speak",
                args={"to": other},
                say=f"{p.town_leader_name} is working the hall one vote at a time. Stay close.",
            )
        if headlines and self.rng.random() < 0.55:
            headline = self.rng.choice(headlines)
            return Intent(
                action="gossip",
                args={"about": headline["about"]},
                say=headline.get("text") or f"Did you hear about {other_name}?",
            )
        if self.rng.random() < 0.3:
            return Intent(
                action="gossip",
                args={"about": other, "tone": "chat"},
                say=f"keep an eye on {other_name} — the tavern is writing a story about them",
            )
        return Intent(action="speak", args={"to": other},
                      say=f"{other_name}, tell me who you are backing before the window closes.")
