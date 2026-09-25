"""
economy.py -- Ledger, production/scarcity, and trade resolution.

Design principle: this is where "economy" stops being a word agents say
and starts being numbers that move. Three deliberately small mechanisms,
chosen to be the *minimum* that makes scarcity and trade bite without
modeling a full input-output economy (which would be a distraction from
the actual research question -- how norms/governance/economy *emerge*,
not how to build SimEconomy):

    1. Work: converts location resources (finite, depleting) into
       inventory, at a rate that means not everyone can have everything.
    2. Trade offers: a two-step propose/resolve flow (never an instant
       atomic swap), so that "this trade is pending" is itself state other
       agents can perceive, gossip about, and react to.
    3. The ledger: a simple dict of agent_id -> money, with all transfers
       going through `_transfer` so an audit trail (world.log_event) is
       never skipped.

INVARIANT: this module is the only place `agent.money` and
`agent.inventory` are mutated (besides direct initialization in main.py).
"""

from __future__ import annotations

import copy
import itertools

from agent import Agent
from memory import MemoryEntry
from world import World


# Module-level counter for offer IDs. A simple itertools.count is enough
# here -- offers are ephemeral (created and resolved within a few ticks),
# so we don't need anything fancier like a UUID for collision safety in a
# single-process simulation.
_offer_ids = itertools.count(1)

# Pending trade offers, keyed by offer_id. Owned by this module rather
# than by World, since trade-offer bookkeeping is purely an economy
# concern -- World stays focused on places/clock/rules (see world.py's
# docstring). engine.py never touches this dict directly.
_pending_offers: dict[int, dict] = {}

# How much of a location's resource pool one `work` action extracts.
# Named constant, not a magic number, so the "how scarce is scarcity"
# knob is easy to find and tune during playtesting.
WORK_YIELD = 1.0
DEFAULT_RESOURCE_KIND = "food"

# Regeneration rate: how much a location's resource pool replenishes per
# tick, up to its cap. Added after the 1000-tick stress test showed the
# farm's food pool draining to 0.0 by tick ~100 and staying there for the
# remaining 900 ticks -- a real finding, not a code bug: the original
# model had extraction with no renewal, making the "economy" a one-time
# liquidation rather than a sustainable system. A real farm produces food
# every season; this regen rate is the minimum fix for that. Tuned low
# (slower than WORK_YIELD) so scarcity still bites -- the goal is a
# steady state where work matters, not removing scarcity altogether.
RESOURCE_REGEN_RATE = 0.3
RESOURCE_CAP = 40.0

# Livelihoods are how a crisis chooses its victims. Farmers take flood
# and famine; traders take a bank run; laborers take unrest. Bankruptcy
# is insolvency on this ledger — not a lost seat in the hall.
LIVELIHOODS = {
    "farmer": "Farmer",
    "trader": "Trader",
    "laborer": "Laborer",
}
ECONOMIC_RULES = ("wealth_tax", "water_blessing")
BANKRUPT_MONEY = 0.8
BANKRUPT_FOOD = 0.45
STRAINED_MONEY = 4.0


def assign_livelihood(index: int, faith_id: str | None = None) -> str:
    if faith_id == "old_ways":
        return "farmer"
    if faith_id == "hall_creed":
        return ("trader", "laborer")[index % 2]
    return ("trader", "laborer", "farmer")[index % 3]


def livelihood_name(livelihood: str | None) -> str:
    return LIVELIHOODS.get(livelihood or "laborer", "Laborer")


def exposure(agent: Agent, tag: str) -> float:
    """How hard this crisis hits this person, in [0, 1]."""
    job = getattr(agent.persona, "livelihood", "laborer")
    here = agent.location
    if tag == "flood":
        hit = 1.0 if job == "farmer" else 0.25 if job == "trader" else 0.35
        if here in ("farm", "park"):
            hit = max(hit, 0.75)
        return hit
    if tag == "famine":
        hit = 1.0 if job == "farmer" else 0.55 if job == "laborer" else 0.4
        if here == "farm":
            hit = max(hit, 0.7)
        return hit
    if tag == "bank_run":
        hit = 1.0 if job == "trader" else 0.25 if job == "farmer" else 0.45
        if here == "bank" or agent.money >= 12:
            hit = max(hit, 0.8)
        return hit
    if tag == "unrest":
        hit = 1.0 if job == "laborer" else 0.55 if job == "trader" else 0.3
        if here in ("market", "tavern", "park", "town_hall"):
            hit = max(hit, 0.65)
        return hit
    if tag == "drought":
        hit = 1.0 if job == "farmer" else 0.35 if job == "laborer" else 0.3
        if here in ("farm", "chapel"):
            hit = max(hit, 0.75)
        return hit
    if tag == "pollution":
        hit = 1.0 if job == "laborer" else 0.4 if job == "trader" else 0.25
        if here == "workshop":
            hit = max(hit, 0.9)
        if here in ("market", "homes"):
            hit = max(hit, 0.45)
        return hit
    return 0.2


def work_yield_now(world: World, actor: Agent) -> float:
    """Harvest shrinks in a flood or famine, especially for farmers."""
    yield_amt = WORK_YIELD
    mag = float((world.crisis_intensity or {}).get("flood", 0) or 0)
    if mag and actor.location == "farm":
        yield_amt *= max(0.15, 1.0 - 0.55 * mag)
    mag = float((world.crisis_intensity or {}).get("famine", 0) or 0)
    if mag and actor.location == "farm":
        yield_amt *= max(0.2, 1.0 - 0.4 * mag)
    mag = float((world.crisis_intensity or {}).get("drought", 0) or 0)
    if mag and actor.location == "farm":
        yield_amt *= max(0.12, 1.0 - 0.6 * mag)
    mag = float((world.crisis_intensity or {}).get("pollution", 0) or 0)
    if mag and actor.location == "workshop":
        yield_amt *= max(0.2, 1.0 - 0.5 * mag)
    return round(yield_amt, 3)

# Demurrage rate: a small percentage of EVERY agent's money is taxed
# away each tick, regardless of governance, and routed into the town
# treasury (see apply_demurrage below -- an earlier version destroyed
# this amount outright with no beneficiary, which a system-wide money
# supply check caught as a real bug: total money shrank monotonically
# toward zero with no source ever replenishing it). This is framed as
# "demurrage" (a real economic concept: decaying currency, used
# historically and in some local-currency systems specifically to
# discourage hoarding) rather than as taxation proper -- it runs
# unconditionally, with no vote required, and is deliberately mild.
# Added after stress-testing showed that even with FAIR-priced trades
# (see FAIR_PRICE_PER_UNIT in decision.py), money has no decay analogous
# to reputation's pull-to-neutral, so pure compounding luck over long
# runs produces persistent, growing inequality (Gini trending toward
# ~0.7 by tick 1000) with nothing pushing back. The heavier, OPTIONAL
# lever the town can choose to layer on top is the governance-enacted
# wealth tax (see governance.py's "wealth_tax" rule type).
DEMURRAGE_RATE = 0.0015  # ~0.15% of holdings per tick


def apply_demurrage(world, agents: dict) -> None:
    """Called once per tick by engine.py. Shrinks every agent's money by
    DEMURRAGE_RATE -- but routes the collected amount into world.treasury
    rather than destroying it.

    IMPORTANT CORRECTION: an earlier version of this function destroyed
    the collected amount outright (no `world` parameter, no treasury
    credit). Checking total system money supply (sum of all agents' money
    + treasury) over a 1000-tick run caught this: it shrank monotonically
    from ~118 to ~30 with no floor, because `work` only ever produces
    FOOD (see economy.py's `work`), never money, so money had exactly one
    exit (demurrage) and no entry point once initial endowments were
    spent down. A currency that only ever leaves circulation isn't
    modeling demurrage -- real decaying-currency systems work because
    the decayed value gets reinjected (historically: into public works,
    here: into the treasury for eventual redistribution). Routing into
    the treasury, which periodic redistribution (see
    `redistribute_treasury_if_due` below) eventually returns to the
    population, closes that loop and keeps total system money conserved.
    """
    for agent in agents.values():
        if agent.money > 0:
            levy = round(agent.money * DEMURRAGE_RATE, 4)
            agent.money = round(agent.money - levy, 4)
            world.treasury += levy


# How often (in ticks) the treasury redistributes its balance evenly
# across all agents, INDEPENDENT of whether a wealth_tax rule is active.
# This is what gives demurrage's collected funds somewhere to go even
# when the town hasn't voted in an active wealth_tax -- without this,
# demurrage proceeds would simply accumulate in the treasury forever,
# which is just relocating the "money disappears" problem rather than
# fixing it (see apply_demurrage's docstring for the original finding).
BASELINE_REDISTRIBUTION_PERIOD = 30


def redistribute_treasury_if_due(world, agents: dict) -> None:
    """Called once per tick by engine.py. Independent of governance's
    wealth_tax (which collects AND redistributes on its own cadence --
    see governance.apply_wealth_tax_if_due), this periodically empties
    whatever has accumulated in the treasury (from demurrage, mainly)
    back out to the population evenly. Kept as a SEPARATE, slower cadence
    from wealth_tax specifically so the two remain conceptually distinct
    even though they share a treasury: demurrage+baseline redistribution
    is unconditional background plumbing; wealth_tax is the thing the
    town actually has to vote for.
    """
    if world.tick % BASELINE_REDISTRIBUTION_PERIOD != 0:
        return
    if world.treasury <= 0 or not agents:
        return
    share = world.treasury / len(agents)
    for agent in agents.values():
        agent.money = round(agent.money + share, 4)
    world.treasury = 0.0


def regenerate_resources(world) -> None:
    """Called once per tick by engine.py, BEFORE agent actions execute
    (see engine.py's step() ordering). Replenishes each location's
    resource pools toward RESOURCE_CAP. Kept as a simple linear regrowth
    rather than a logistic/carrying-capacity curve -- the latter is more
    realistic but is exactly the kind of refinement to add only if a
    future stress test shows linear regen producing its own degenerate
    behavior (e.g. oscillation). Start simple, earn complexity with
    evidence, same principle as the rest of this codebase.
    """
    for loc in world.locations.values():
        current = loc.resources.get(DEFAULT_RESOURCE_KIND, 0.0)
        if current < RESOURCE_CAP:
            import inventions
            bonus = inventions.farm_regen_bonus(world) if loc.name == "farm" else 0.0
            if loc.name == "farm":
                import faith
                bonus += faith.water_blessing_bonus(world)
                drought = float((world.crisis_intensity or {}).get("drought", 0) or 0)
                if drought:
                    bonus -= RESOURCE_REGEN_RATE * (0.85 * drought)
            loc.resources[DEFAULT_RESOURCE_KIND] = min(
                RESOURCE_CAP, max(0.0, current + RESOURCE_REGEN_RATE + bonus)
            )


def work(actor: Agent, world: World):
    """Execute a `work` action: extract WORK_YIELD of DEFAULT_RESOURCE_KIND
    from the actor's current location into the actor's inventory.

    Args:
        actor: the agent performing the action.
        world: the simulation world (used to look up the actor's current
            Location and its resource pool).

    Returns:
        actions.ActionResult: success with the amount gathered, or
        failure if the location has insufficient resources remaining.
        (Return type is left unannotated due to the local import below
        avoiding a circular dependency with actions.py.)
    """
    from actions import ActionResult  # local import: economy.py is imported BY
    # actions.py, so importing ActionResult at module level would create a
    # cycle. Importing inside the function avoids that without restructuring
    # the module graph. This is the one deliberate exception to "imports at
    # the top" in this codebase, and it's confined to this single spot.

    loc = world.get_location(actor.location)
    gained = work_yield_now(world, actor)
    available = loc.resources.get(DEFAULT_RESOURCE_KIND, 0.0)
    if available < gained or gained <= 0:
        return ActionResult(False, f"no {DEFAULT_RESOURCE_KIND} left to gather at {actor.location}")

    loc.resources[DEFAULT_RESOURCE_KIND] = available - gained
    actor.inventory[DEFAULT_RESOURCE_KIND] = actor.inventory.get(DEFAULT_RESOURCE_KIND, 0.0) + gained
    actor.memory.add(MemoryEntry(world.tick, "worked", None,
                                 {"location": actor.location, "gained": gained}))
    world.log_event("work", agent=actor.agent_id, location=actor.location, gained=gained)
    return ActionResult(True, f"gathered {gained} {DEFAULT_RESOURCE_KIND}")


def create_offer(actor: Agent, target: Agent, args: dict):
    """Execute a `trade_offer` action: validate that `actor` actually
    holds what they're proposing to give, then register a pending offer
    addressed to `target` for later accept/reject resolution.

    Args:
        actor: the agent proposing the trade (the one initiating
            `trade_offer`).
        target: the agent the offer is addressed to (must later call
            `trade_accept`/`trade_reject` to resolve it).
        args: dict with keys "give" (what `actor` offers, e.g.
            {"food": 2}) and "want" (what `actor` requests in return,
            e.g. {"money": 5}). Both default to {} if omitted.

    Returns:
        actions.ActionResult: success with the new offer_id, or failure
        if `actor` doesn't actually hold the goods/money being offered.
    """
    from actions import ActionResult

    give = args.get("give", {})   # what actor offers, e.g. {"food": 2}
    want = args.get("want", {})   # what actor wants in return, e.g. {"money": 5}

    # Validate the actor actually HAS what they're offering. This is the
    # check that prevents a malformed/hallucinated Intent (Phase 2) from
    # creating an offer for goods that don't exist -- caught here, before
    # it ever becomes a pending offer another agent could "accept" into
    # a state-corrupting transfer.
    for item, qty in give.items():
        if item == "money":
            if actor.money < qty:
                return ActionResult(False, f"cannot offer {qty} money, only have {actor.money}")
        elif actor.inventory.get(item, 0.0) < qty:
            return ActionResult(False, f"cannot offer {qty} {item}, only have {actor.inventory.get(item, 0.0)}")

    offer_id = next(_offer_ids)
    _pending_offers[offer_id] = {
        "offer_id": offer_id,
        "from": actor.agent_id,
        "to": target.agent_id,
        "give": give,
        "want": want,
    }
    target.memory.add(MemoryEntry(0, "received_trade_offer", actor.agent_id,
                                   {"offer_id": offer_id, "give": give, "want": want}))
    return ActionResult(True, f"offer {offer_id} created")


def resolve_offer(actor: Agent, offer_id, accept: bool, world: World, agents: dict):
    """Execute a `trade_accept` or `trade_reject` action: resolve a
    pending offer addressed to `actor`.

    On accept, re-validates both parties still hold what's required
    (state may have shifted since the offer was created -- see inline
    comment below), then performs the transfer, updates relationships
    and reputation, and records a memory for both parties. On reject,
    simply removes the offer and notifies the proposer.

    Args:
        actor: the agent resolving the offer (must be the offer's `to`).
        offer_id: the id returned by `create_offer`.
        accept: True to accept and execute the trade, False to reject.
        world: the simulation world (used for tick-stamping memories and
            event logging).
        agents: the full agent registry, used to look up the proposer.

    Returns:
        actions.ActionResult: success or a specific failure reason
        (no such offer, not addressed to this agent, proposer no longer
        exists, or insufficient funds/goods at resolution time).
    """
    from actions import ActionResult

    offer = _pending_offers.pop(offer_id, None) if offer_id is not None else None
    if offer is None:
        return ActionResult(False, f"no pending offer '{offer_id}'")
    if offer["to"] != actor.agent_id:
        # Someone tried to accept/reject an offer not addressed to them.
        # Put it back -- this isn't actor's offer to resolve.
        _pending_offers[offer_id] = offer
        return ActionResult(False, "offer not addressed to this agent")

    proposer = agents.get(offer["from"])
    if proposer is None:
        return ActionResult(False, "proposing agent no longer exists")

    if not accept:
        proposer.memory.add(MemoryEntry(world.tick, "trade_rejected", actor.agent_id, {"offer_id": offer_id}))
        world.log_event("trade_rejected", offer_id=offer_id, by=actor.agent_id)
        return ActionResult(True, "rejected")

    # Re-validate both sides have what's required at resolution time --
    # state may have changed between offer creation and acceptance (e.g.
    # the proposer already spent the money on something else). Re-checking
    # here, not just at creation, is what prevents a stale offer from
    # creating money/goods out of nothing.
    give, want = offer["give"], offer["want"]
    if not _has_sufficient(proposer, give) or not _has_sufficient(actor, want):
        world.log_event("trade_failed_insufficient_funds", offer_id=offer_id)
        return ActionResult(False, "one party no longer has sufficient funds/goods")

    _transfer(proposer, actor, give)
    _transfer(actor, proposer, want)

    TRADE_RELATIONSHIP_DELTA = 0.05
    proposer.adjust_relationship(actor.agent_id, TRADE_RELATIONSHIP_DELTA)
    actor.adjust_relationship(proposer.agent_id, TRADE_RELATIONSHIP_DELTA)

    # A completed trade is also a small PUBLIC reputation signal for both
    # parties -- distinct from the private bilateral relationship bump
    # above. This is what closes the gap between agent.py's documented
    # intent ("reputation mutated by actions.py/governance.py") and
    # actual behavior: prior to this, `reputation` was set once at
    # construction and never touched again, which silently broke any
    # downstream norm logic that reads it. Kept small and symmetric --
    # honoring a trade is mildly reputation-positive for both sides, not
    # just the seller, since reliably paying up is also norm-following.
    TRADE_REPUTATION_DELTA = 0.01
    REPUTATION_CAP = 1.0
    proposer.reputation = min(REPUTATION_CAP, proposer.reputation + TRADE_REPUTATION_DELTA)
    actor.reputation = min(REPUTATION_CAP, actor.reputation + TRADE_REPUTATION_DELTA)

    for a in (proposer, actor):
        other_id = actor.agent_id if a is proposer else proposer.agent_id
        a.memory.add(MemoryEntry(world.tick, "trade_completed", other_id,
                                  {"offer_id": offer_id, "give": give, "want": want}))
    world.log_event("trade_completed", offer_id=offer_id, from_=proposer.agent_id, to=actor.agent_id)
    return ActionResult(True, "trade completed")


def _has_sufficient(agent: Agent, items: dict) -> bool:
    """Check whether `agent` currently holds at least the quantities
    specified in `items` (a dict of item_name -> quantity, where "money"
    is checked against `agent.money` and anything else against
    `agent.inventory`). Used by `resolve_offer` to re-validate both
    parties at resolution time, not just at offer creation.
    """
    for item, qty in items.items():
        held = agent.money if item == "money" else agent.inventory.get(item, 0.0)
        if held < qty:
            return False
    return True


def _transfer(sender: Agent, receiver: Agent, items: dict) -> None:
    """Move `items` from sender to receiver. The only function that
    actually mutates `.money` / `.inventory` -- every trade path funnels
    through here so there is exactly one place to audit for ledger bugs.
    """
    for item, qty in items.items():
        if item == "money":
            sender.money -= qty
            receiver.money += qty
        else:
            sender.inventory[item] = sender.inventory.get(item, 0.0) - qty
            receiver.inventory[item] = receiver.inventory.get(item, 0.0) + qty


def offers_for(agent_id: str) -> list:
    """Return pending offers addressed to `agent_id`. Used by engine.py
    when building that agent's Perception.
    """
    return [o for o in _pending_offers.values() if o["to"] == agent_id]


def reset_offers() -> None:
    """Clear all pending offers. Exists mainly for test isolation, since
    `_pending_offers` is module-level state shared across a process --
    without this, running multiple simulations in one Python process
    (e.g. in a test suite) would leak offers between runs.
    """
    _pending_offers.clear()


def export_state() -> dict:
    used = list(_pending_offers)
    return {
        "pending_offers": copy.deepcopy(_pending_offers),
        "next_id": max(used, default=0) + 1,
    }


def import_state(blob: dict) -> None:
    global _offer_ids
    reset_offers()
    _pending_offers.update(copy.deepcopy(blob.get("pending_offers") or {}))
    _offer_ids = itertools.count(int(blob.get("next_id") or 1))


def apply_crisis_pressure(world: World, agents: dict[str, Agent], rng) -> None:
    """Clock: ruin the people a crisis actually lands on, then mark solvency.

    Flood and famine take farmers first. A bank run takes traders and
    large balances. Unrest takes laborers in the street. Bankruptcy is
    recorded here; the hall does not take their vote for it.
    """
    intensities = world.crisis_intensity or {}
    if intensities:
        for agent in agents.values():
            money_cut = 0.0
            food_cut = 0.0
            for tag, mag in intensities.items():
                exp = exposure(agent, tag)
                if exp <= 0.05:
                    continue
                mag = float(mag or 0)
                if tag == "flood":
                    money_cut += agent.money * 0.05 * mag * exp
                    food_cut += 0.18 * mag * exp
                elif tag == "famine":
                    money_cut += agent.money * 0.03 * mag * exp
                    food_cut += 0.22 * mag * exp
                elif tag == "bank_run":
                    money_cut += agent.money * 0.045 * mag * exp
                elif tag == "unrest":
                    money_cut += agent.money * 0.035 * mag * exp
                    if rng.random() < 0.15 * mag * exp:
                        agent.reputation = max(0.0, agent.reputation - 0.02 * mag)
                elif tag == "drought":
                    money_cut += agent.money * 0.025 * mag * exp
                    food_cut += 0.2 * mag * exp
                elif tag == "pollution":
                    money_cut += agent.money * 0.03 * mag * exp
                    if rng.random() < 0.2 * mag * exp:
                        agent.reputation = max(0.0, agent.reputation - 0.025 * mag)
            if money_cut:
                agent.money = round(max(0.0, agent.money - money_cut), 2)
            if food_cut:
                food = agent.inventory.get("food", 0.0)
                agent.inventory["food"] = round(max(0.0, food - food_cut), 2)
    _update_solvency(world, agents)


def _update_solvency(world: World, agents: dict[str, Agent]) -> None:
    for agent in agents.values():
        food = float(agent.inventory.get("food", 0.0))
        money = float(agent.money)
        prev = getattr(agent, "solvency", "ok") or "ok"
        if money < BANKRUPT_MONEY and food < BANKRUPT_FOOD:
            nxt = "bankrupt"
        elif money < STRAINED_MONEY or food < 0.8:
            nxt = "strained"
        else:
            nxt = "ok"
        if prev == "bankrupt" and nxt != "ok" and money < 3.0:
            nxt = "bankrupt" if money < 2.0 else "strained"
        if nxt == prev:
            continue
        agent.solvency = nxt
        job = getattr(agent.persona, "livelihood", "laborer")
        if nxt == "bankrupt":
            agent.reputation = max(0.05, agent.reputation - 0.08)
            agent.memory.add(MemoryEntry(world.tick, "went_bankrupt", None, {"livelihood": job}))
            world.log_event("bankrupt", agent=agent.agent_id, name=agent.persona.name,
                            livelihood=job, money=round(money, 2))
            world.notice_board.append({
                "tick": world.tick, "from": "market", "about": agent.agent_id,
                "text": f"{agent.persona.name} is bankrupt — still seated on both ballots",
            })
            del world.notice_board[:-8]
        elif nxt == "strained" and prev == "ok":
            agent.memory.add(MemoryEntry(world.tick, "going_bankrupt", None, {"livelihood": job}))
            world.log_event("going_bankrupt", agent=agent.agent_id, name=agent.persona.name,
                            livelihood=job, money=round(money, 2))
        elif prev in ("bankrupt", "strained") and nxt == "ok":
            agent.memory.add(MemoryEntry(world.tick, "recovered", None, {"from": prev}))
            world.log_event("recovered", agent=agent.agent_id, name=agent.persona.name,
                            from_=prev)


def snapshot(world: World, agents: dict[str, Agent]) -> dict:
    """Frame payload: who is ruined, who still holds each ballot."""
    by_job = {k: 0 for k in LIVELIHOODS}
    by_solvency = {"ok": 0, "strained": 0, "bankrupt": 0}
    ruined = []
    for agent in agents.values():
        job = getattr(agent.persona, "livelihood", "laborer")
        by_job[job] = by_job.get(job, 0) + 1
        sol = getattr(agent, "solvency", "ok") or "ok"
        by_solvency[sol] = by_solvency.get(sol, 0) + 1
        if sol in ("strained", "bankrupt"):
            ruined.append({
                "id": agent.agent_id,
                "name": agent.persona.name,
                "livelihood": job,
                "solvency": sol,
                "money": round(agent.money, 2),
                "can_vote_civic": agent.can_vote_on(world.tick, "curfew"),
                "can_vote_economy": agent.can_vote_on(world.tick, "wealth_tax"),
            })
    return {
        "livelihoods": by_job,
        "names": dict(LIVELIHOODS),
        "solvency": by_solvency,
        "ruined": ruined,
        "economic_rules": list(ECONOMIC_RULES),
    }
