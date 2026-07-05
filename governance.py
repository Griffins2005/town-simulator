"""
governance.py -- Proposals, voting, and rule enforcement.

Design principle: a "law" that doesn't constrain any actual action is
flavor text, not governance. So this module has two halves that must stay
connected:

    1. The democratic process: propose -> vote -> tally -> (if passed)
       write into world.active_rules. This half is intentionally simple
       (majority vote, fixed voting window) -- the interesting research
       question is what agents DO with governance, not exotic voting
       theory, so the mechanism itself stays legible.
    2. The enforcement hooks: small functions like `is_movement_blocked`
       that OTHER modules (actions.py) call to check active_rules before
       allowing an action. This is what closes the loop -- without these
       hooks, active_rules would just be a dict nothing reads, and
       "governance" would be exactly the theater problem flagged earlier.

Three rule types are implemented end-to-end:
    - `curfew`: blocks movement after a given tick-of-day (enforced via
      `is_movement_blocked`, called from actions.py).
    - `wealth_tax`: periodic collection above a threshold into the
      treasury, then redistribution (enforced via
      `apply_wealth_tax_if_due`, called from engine.py).
    - `repeal`: removes a previously-enacted rule's `active_rules` keys
      (enforced in `_finalize`, using `_enacted_keys_by_proposal` to
      know exactly which keys a given proposal_id wrote).

Adding another rule type means: a new branch in `propose`/`_finalize`,
and (for rules that constrain agent actions) a new enforcement hook
called from the relevant action handler or engine housekeeping. The
propose -> vote -> tally -> enact pipeline itself does not change.
"""

from __future__ import annotations

import itertools

from agent import Agent
from memory import MemoryEntry
from world import World

_proposal_ids = itertools.count(1)

# Open proposals currently accepting votes, keyed by proposal_id.
# Module-level for the same reason as economy.py's _pending_offers:
# proposal bookkeeping is purely a governance concern, kept out of World.
_open_proposals: dict[int, dict] = {}

VOTING_WINDOW_TICKS = 10  # how many ticks a proposal stays open for voting
PASS_THRESHOLD = 0.5      # fraction of votes cast (not of all agents) needed to pass

_SUPPORTED_RULE_TYPES = {"curfew", "wealth_tax", "repeal"}

# Maps proposal_id -> the set of world.active_rules keys that proposal
# wrote, so a later "repeal" proposal can remove exactly those keys
# without needing to know the rule's internal shape. Added alongside
# chaos.py's repeal mechanic -- world.active_rules itself stays a flat
# dict (no breaking change to is_movement_blocked / apply_wealth_tax_if_due,
# which read flat keys directly), this is purely an out-of-band index
# recording "which keys did THIS proposal add."
_enacted_keys_by_proposal: dict[int, set] = {}

# Maps proposal_id -> rule_type, for exactly the proposals present in
# _enacted_keys_by_proposal (kept in lockstep with it -- entries are
# added together in _finalize and removed together when a repeal
# resolves). Exists purely so enacted_proposals_snapshot() can tell a
# Decider what KIND of rule each repealable proposal_id enacted (e.g.
# "curfew" vs "wealth_tax") without exposing the raw active_rules keys.
_rule_type_by_proposal: dict[int, str] = {}


def propose(actor: Agent, args: dict, world: World):
    """Execute a `propose_rule` action: open a new proposal of the given
    `rule_type` for a fixed VOTING_WINDOW_TICKS-tick voting period.

    Args:
        actor: the agent making the proposal.
        args: dict with keys "rule_type" (must be in
            `_SUPPORTED_RULE_TYPES`: "curfew", "wealth_tax", or "repeal")
            and "rule_args" (rule-specific parameters). For "curfew":
            {"after_tick_of_day": int, "period": int}. For "wealth_tax":
            {"rate": float, "threshold": float, "period": int}. For
            "repeal": {"target_proposal_id": int} -- the proposal_id of
            a PREVIOUSLY ENACTED rule to remove. Repeal deliberately
            reuses this exact same propose/vote/enact pipeline (see
            chaos.py's module docstring for why) rather than being a
            separate mechanism -- a law and its repeal are the same
            kind of collective decision, just with opposite effect.
        world: the simulation world (used for tick-stamping the
            proposal's open/close window and event logging).

    Returns:
        actions.ActionResult: success with the new proposal_id, or
        failure if `rule_type` isn't supported, or (for "repeal")
        the target proposal was never enacted / already repealed.
    """
    from actions import ActionResult

    rule_type = args.get("rule_type")
    if rule_type not in _SUPPORTED_RULE_TYPES:
        return ActionResult(False, f"unsupported rule_type '{rule_type}'")

    if rule_type == "repeal":
        target_id = args.get("rule_args", {}).get("target_proposal_id")
        if target_id not in _enacted_keys_by_proposal:
            return ActionResult(False, f"proposal '{target_id}' has no active enacted rule to repeal")

    proposal_id = next(_proposal_ids)
    _open_proposals[proposal_id] = {
        "proposal_id": proposal_id,
        "proposed_by": actor.agent_id,
        "rule_type": rule_type,
        "rule_args": args.get("rule_args", {}),
        "opens_at": world.tick,
        "closes_at": world.tick + VOTING_WINDOW_TICKS,
        "votes": {},  # agent_id -> "yes" | "no"
    }
    world.log_event("rule_proposed", proposal_id=proposal_id, by=actor.agent_id, rule_type=rule_type)
    return ActionResult(True, f"proposal {proposal_id} opened")


def cast_vote(actor: Agent, proposal_id, choice: str, world: World):
    """Execute a `vote` action: record `actor`'s vote on an open
    proposal. A repeat vote from the same agent silently overwrites
    their prior choice (see README's "known limitations" for why this
    is currently harmless in practice).

    Args:
        actor: the agent casting the vote.
        proposal_id: the id of the proposal being voted on.
        choice: must be "yes" or "no".
        world: the simulation world (used to check the voting window
            hasn't closed, and for event logging).

    Returns:
        actions.ActionResult: success, or failure if the proposal
        doesn't exist, its voting window has closed, or `choice` is
        neither "yes" nor "no".
    """
    from actions import ActionResult

    proposal = _open_proposals.get(proposal_id)
    if proposal is None:
        return ActionResult(False, f"no open proposal '{proposal_id}'")
    if world.tick > proposal["closes_at"]:
        return ActionResult(False, "voting window has closed")
    if choice not in ("yes", "no"):
        return ActionResult(False, f"invalid vote choice '{choice}'")

    proposal["votes"][actor.agent_id] = choice
    world.log_event("vote_cast", proposal_id=proposal_id, by=actor.agent_id, choice=choice)
    return ActionResult(True, "vote recorded")


def tick(world: World, agents: dict) -> None:
    """Called once per engine tick (see engine.py). Closes any proposals
    whose voting window has elapsed, tallies votes, and -- if passed --
    writes the rule into world.active_rules.

    This is the one piece of governance.py NOT triggered by an agent
    Intent -- it's clock-driven housekeeping, which is why it lives here
    as a plain function the engine calls directly, rather than being
    routed through actions.py like agent-initiated behavior.
    """
    closed_ids = [pid for pid, p in _open_proposals.items() if world.tick >= p["closes_at"]]
    for pid in closed_ids:
        proposal = _open_proposals.pop(pid)
        _finalize(proposal, world, agents)


def _finalize(proposal: dict, world: World, agents: dict) -> None:
    """Tally a closed proposal's votes and, if it passed
    (yes-fraction-of-votes-cast >= PASS_THRESHOLD), write the
    corresponding rule into `world.active_rules`. Records a memory of
    the outcome for every agent who voted, and logs the result
    regardless of outcome.

    Called only by `tick` (above), once a proposal's voting window has
    elapsed. Not part of the public propose/vote/execute action surface
    -- this is the clock-driven resolution step, not something an agent
    can trigger directly.
    """
    votes = proposal["votes"]
    yes = sum(1 for v in votes.values() if v == "yes")
    total = len(votes)
    passed = total > 0 and (yes / total) >= PASS_THRESHOLD

    world.log_event("proposal_closed", proposal_id=proposal["proposal_id"],
                     passed=passed, yes=yes, total=total)

    # Every participant gets a memory of the outcome -- this is what lets
    # a rule-based or LLM agent's future rule_respect / compliance behavior
    # be grounded in "I remember voting for/against this," not just an
    # abstract fact about the world.
    for agent_id, choice in votes.items():
        agent = agents.get(agent_id)
        if agent:
            agent.memory.add(MemoryEntry(world.tick, "proposal_resolved", None,
                                          {"proposal_id": proposal["proposal_id"],
                                           "my_vote": choice, "passed": passed}))

    if not passed:
        return

    # A passed proposal makes its proposer more of a political "official"
    # -- a track record other modules (chaos.py's corruption mechanism)
    # read to bias outcomes toward agents who've actually gotten things
    # enacted, not just anyone who happens to be standing in town_hall.
    # Repeals count too -- successfully overturning a law is itself a
    # real political act, arguably MORE so than passing a routine one.
    proposer = agents.get(proposal["proposed_by"])
    if proposer:
        proposer.official_track_record += 1

    rule_type = proposal["rule_type"]
    rule_args = proposal["rule_args"]
    enacted_keys: set = set()
    if rule_type == "curfew":
        world.active_rules["curfew_after_tick_of_day"] = rule_args.get("after_tick_of_day", 20)
        world.active_rules["curfew_period"] = rule_args.get("period", 24)
        enacted_keys = {"curfew_after_tick_of_day", "curfew_period"}
    elif rule_type == "wealth_tax":
        # A periodic progressive-ish levy: every `period` ticks, agents
        # above `threshold` money have `rate` of their EXCESS over the
        # threshold collected into world.treasury, then the treasury is
        # split evenly across all agents. This is deliberately the
        # "heavier, active" counterpart to economy.py's passive
        # demurrage -- demurrage runs unconditionally and gently;
        # wealth_tax only runs if the town's agents vote it in, can be
        # voted back out via a "repeal" proposal (see chaos.py), and
        # actually redistributes rather than just shrinking balances
        # into nothing. This is what gives "governance" a real economic
        # lever, not just a curfew toy.
        world.active_rules["wealth_tax_rate"] = rule_args.get("rate", 0.1)
        world.active_rules["wealth_tax_threshold"] = rule_args.get("threshold", 15.0)
        world.active_rules["wealth_tax_period"] = rule_args.get("period", 20)
        enacted_keys = {"wealth_tax_rate", "wealth_tax_threshold", "wealth_tax_period"}
    elif rule_type == "repeal":
        # Remove exactly the keys the target proposal originally wrote,
        # without needing to know that rule's internal shape -- this is
        # what _enacted_keys_by_proposal exists for. A repeal of a
        # repeal is meaningless (repeals don't write any active_rules
        # keys themselves), so repeals are never themselves repealable
        # -- _enacted_keys_by_proposal simply never gets an entry for
        # rule_type=="repeal" (see below), so propose()'s validation
        # already rejects "repeal a repeal" as "no active enacted rule
        # to repeal."
        target_id = rule_args.get("target_proposal_id")
        target_keys = _enacted_keys_by_proposal.pop(target_id, set())
        _rule_type_by_proposal.pop(target_id, None)
        for key in target_keys:
            world.active_rules.pop(key, None)
        world.log_event("rule_repealed", target_proposal_id=target_id, removed_keys=sorted(target_keys))
        return  # no new active_rules keys to record for THIS proposal

    if enacted_keys:
        _enacted_keys_by_proposal[proposal["proposal_id"]] = enacted_keys
        _rule_type_by_proposal[proposal["proposal_id"]] = rule_type
    world.log_event("rule_enacted", rule_type=rule_type, rule_args=rule_args)


def is_movement_blocked(world: World, actor: Agent) -> bool:
    """Enforcement hook called by actions.py's `_move` handler.

    Returns True if an active curfew rule currently blocks movement for
    `actor`. This is the concrete mechanism that gives a passed "curfew"
    proposal real teeth: actions.py checks this BEFORE allowing a move,
    so a law that passed actually constrains behavior rather than being
    a fact nobody enforces.

    Deliberately not agent-specific yet (no exemptions, e.g. for an
    elected "sheriff") -- that's a natural Phase 2 extension once the
    base mechanism is validated.
    """
    after = world.active_rules.get("curfew_after_tick_of_day")
    period = world.active_rules.get("curfew_period")
    if after is None or period is None:
        return False
    tick_of_day = world.tick % period
    return tick_of_day >= after


def apply_wealth_tax_if_due(world: World, agents: dict) -> None:
    """Called once per tick by engine.py. If a wealth_tax rule is active
    AND this tick lands on its collection period, collect `rate` of each
    agent's money above `threshold` into world.treasury, then
    immediately redistribute the ENTIRE treasury (including any prior
    balance) evenly across all agents.

    Collect-then-redistribute-immediately (rather than letting the
    treasury accumulate indefinitely) is a deliberate choice: an
    ever-growing unspent treasury would just be a second place wealth
    "disappears" to, which doesn't actually counteract concentration --
    it would just relocate it. Immediate even redistribution is what
    makes this a real countervailing force against the compounding-luck
    concentration the 1000-tick stress test surfaced (see economy.py's
    DEMURRAGE_RATE docstring for the original finding).

    This intentionally does NOT touch reputation -- being taxed is a
    structural/economic event in this model, not a moral one. Whether
    paying tax should affect how OTHER agents perceive you (e.g. resentment
    toward high earners, or approval of compliance) is a real and
    interesting modeling question, but one to introduce deliberately
    later, not as a side effect of this function.
    """
    rate = world.active_rules.get("wealth_tax_rate")
    threshold = world.active_rules.get("wealth_tax_threshold")
    period = world.active_rules.get("wealth_tax_period")
    if rate is None or threshold is None or period is None:
        return
    if period <= 0 or world.tick % period != 0:
        return

    collected = 0.0
    for agent in agents.values():
        if agent.money > threshold:
            excess = agent.money - threshold
            levy = excess * rate
            agent.money = round(agent.money - levy, 4)
            collected += levy

    world.treasury += collected
    if world.treasury <= 0 or not agents:
        return

    share = world.treasury / len(agents)
    for agent in agents.values():
        agent.money = round(agent.money + share, 4)
    world.log_event("wealth_tax_applied", collected=round(collected, 2),
                     redistributed_per_agent=round(share, 4))
    world.treasury = 0.0


def open_proposals_snapshot() -> list:
    """Read-only list of currently open proposals, for engine.py to embed
    into agents' Perception objects.
    """
    return [dict(p) for p in _open_proposals.values()]


def enacted_proposals_snapshot() -> list:
    """Read-only list of {"proposal_id": int, "rule_type": str} for every
    currently enacted (not yet repealed) rule -- for engine.py to embed
    into agents' Perception objects, so an agent can target a "repeal"
    proposal at a specific enacted rule. Deliberately exposes only
    proposal_id and rule_type, not the raw active_rules keys each
    proposal wrote (that's _enacted_keys_by_proposal's job internally;
    Perception has no business knowing about active_rules' key shape).
    """
    result = []
    for proposal_id in _enacted_keys_by_proposal:
        rule_type = _rule_type_by_proposal.get(proposal_id, "unknown")
        result.append({"proposal_id": proposal_id, "rule_type": rule_type})
    return result


def reset() -> None:
    """Clear all open proposals AND the enacted-keys/rule-type-by-proposal
    indexes. Test/run isolation, same rationale as economy.reset_offers()
    -- without clearing these too, a second simulation run in the same
    process could see proposal_ids from the FIRST run as still
    "repealable," since itertools.count and these dicts aren't otherwise
    reset together.
    """
    _open_proposals.clear()
    _enacted_keys_by_proposal.clear()
    _rule_type_by_proposal.clear()
