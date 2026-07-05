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
            loc.resources[DEFAULT_RESOURCE_KIND] = min(RESOURCE_CAP, current + RESOURCE_REGEN_RATE)


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
    available = loc.resources.get(DEFAULT_RESOURCE_KIND, 0.0)
    if available < WORK_YIELD:
        return ActionResult(False, f"no {DEFAULT_RESOURCE_KIND} left to gather at {actor.location}")

    loc.resources[DEFAULT_RESOURCE_KIND] = available - WORK_YIELD
    actor.inventory[DEFAULT_RESOURCE_KIND] = actor.inventory.get(DEFAULT_RESOURCE_KIND, 0.0) + WORK_YIELD
    world.log_event("work", agent=actor.agent_id, location=actor.location, gained=WORK_YIELD)
    return ActionResult(True, f"gathered {WORK_YIELD} {DEFAULT_RESOURCE_KIND}")


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
