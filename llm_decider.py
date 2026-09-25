"""
llm_decider.py -- Phase 2: an LLM-backed Decider using Groq's free tier.

THIS FILE IS THE ONLY THING THAT CHANGES to go from Phase 1 (rule-based,
free, deterministic) to Phase 2 (LLM-driven, free-tier-rate-limited,
genuinely reasoning). Nothing in world.py, agent.py, actions.py,
economy.py, governance.py, or engine.py needs to change -- LLMDecider
implements the exact same `Decider` protocol (`decide(agent_id,
perception) -> Intent`) that RuleBasedDecider does. This is the payoff of
having built decision.py's Perception/Intent contract as the seam from
day one.

IMPORTANT, STATED PLAINLY: this module was written and reasoned through
carefully, including verifying the rate limiter's timing against the
real clock and confirming the installed `groq` SDK's call shape matches
what's used below. It has NOT been exercised against a live Groq API
call from within the environment this was built in, because that
environment's network egress does not include api.groq.com. Treat the
prompt-construction and JSON-parsing logic as carefully-reasoned-but-
unverified-against-the-live-API until you've run it once yourself with a
real GROQ_API_KEY and confirmed the schema round-trips as expected.

Setup:
    python3 -m pip install -r requirements.txt
    export GROQ_API_KEY=your_key_here

The JSON schema and prompt cover the same action set as Phase 1's
RuleBasedDecider, including governance `repeal` (via flat
`rule_target_proposal_id`) and the Phase 4 `Perception` fields an
agent needs to react to chaos (crises, factions, enacted rules, trade
rumors). See `_INTENT_SCHEMA`, `_SYSTEM_PROMPT`, and `_build_user_prompt`.
"""

from __future__ import annotations

import json
import os

from decision import Intent, Perception
from rate_limiter import RECOMMENDED_RPM_SAFETY_MARGIN, TokenBucketRateLimiter


def require_groq() -> None:
    """Fail clearly when `groq` is missing from *this* interpreter.

    `pip install groq` often lands in a different Python than `python3`
    (Anaconda vs Xcode vs a project venv). The message prints
    `sys.executable` so the next install command hits the same binary.
    """
    try:
        import groq  # noqa: F401
    except ModuleNotFoundError as exc:
        import sys
        raise ModuleNotFoundError(
            "groq is not installed for this Python:\n"
            f"  {sys.executable}\n"
            "Install it into that interpreter (plain `pip` may target another):\n"
            f"  {sys.executable} -m pip install -r requirements.txt"
        ) from exc

# Must match main_llm.py's BUILD_VERSION -- main_llm.py checks this at
# startup and warns loudly if they differ, which catches the specific
# failure mode of someone updating one file but not the other (or
# running a stale copy of one of the two) without needing a manual
# grep to notice.
BUILD_VERSION = "2026-07-05-v7-repeal-spec-sync"

# CORRECTED, based on a live API error (June 2026): the original choice
# here was "llama-3.3-70b-versatile", which I asserted supported
# structured-output JSON schema mode based on a general feature
# announcement. A live run returned a 400: "This model does not support
# response format `json_schema`." That was wrong, and the live error is
# the more reliable source. Groq's strict structured-output mode
# (response_format.json_schema.strict=true, which this codebase uses --
# see _INTENT_SCHEMA below) is only supported on a specific, smaller set
# of models; "openai/gpt-oss-120b" is confirmed (across Groq's own docs
# and multiple independent integration docs) to be one of them. If this
# model is later deprecated or Groq's supported-model list changes,
# check https://console.groq.com/docs/structured-outputs#supported-models
# before assuming a new model works -- verify against the live API, the
# same way this correction was made, rather than from a general feature
# description alone.
DEFAULT_MODEL = "openai/gpt-oss-120b"
CHEAP_MODEL = "llama-3.1-8b-instant"
MODEL = os.environ.get("TOWNSIM_LLM_MODEL") or os.environ.get("GROQ_MODEL") or DEFAULT_MODEL


def resolve_model(name: str | None = None) -> str:
    return (name or os.environ.get("TOWNSIM_LLM_MODEL") or os.environ.get("GROQ_MODEL") or DEFAULT_MODEL).strip()


# JSON Schema for Intent, used with Groq's structured-output mode (see
# https://console.groq.com/docs/structured-outputs). This guarantees the
# model's response parses as valid JSON matching this exact shape --
# eliminating the "regex prose out of a chat response" failure mode that
# sinks most LLM-agent prototypes. `action` is constrained to the exact
# set actions.py's _REGISTRY recognizes (kept here as a literal list
# rather than importing actions.py, to avoid this module depending on
# the whole action-execution stack just to know action NAMES).
_VALID_ACTIONS = [
    "move", "work", "trade_offer", "trade_accept", "trade_reject", "trade_counter",
    "bid", "speak", "gossip", "propose_rule", "vote", "lobby", "invent", "adopt_invention",
    "worship", "convert", "idle",
]

_INTENT_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": _VALID_ACTIONS},
        # FLAT fields, not a nested "args" object. This is a real
        # correction: a live run hit "additionalProperties:false must
        # be set on every object" because Groq's strict mode requires
        # that constraint on EVERY object in the schema, including
        # nested ones -- and the original `args: {"type": "object"}`
        # had no declared properties at all, let alone the constraint.
        # Worse, `args` needed to hold structurally different shapes
        # per action (move needs a destination, trade_offer needs an
        # item+amount pair in each direction) AND trade items use a
        # dynamic key (the item name, e.g. "food" or "money") that
        # strict mode's "every property must be named in advance" rule
        # cannot represent at all as a nested dict. The fix: every
        # field any action could need is hoisted to a flat, named,
        # nullable property here. The model fills in only the fields
        # relevant to its chosen action and leaves the rest null;
        # `_parse_intent` (below) reconstructs the nested dict shape
        # actions.py/economy.py already expect, so NOTHING downstream
        # of decision.py's Intent needed to change.
        "destination": {"type": ["string", "null"], "description": "for move"},
        "to": {"type": ["string", "null"], "description": "agent_id for trade_offer/trade_accept/trade_reject/speak/gossip"},
        "offer_id": {"type": ["integer", "null"], "description": "for trade_accept/trade_reject/trade_counter"},
        "auction_id": {"type": ["integer", "null"], "description": "for bid"},
        "bid_amount": {"type": ["number", "null"], "description": "sealed bid amount"},
        "give_item": {"type": ["string", "null"], "description": "item name for trade_offer, e.g. food"},
        "give_amount": {"type": ["number", "null"], "description": "quantity of give_item for trade_offer"},
        "want_item": {"type": ["string", "null"], "description": "item name wanted in return, e.g. money"},
        "want_amount": {"type": ["number", "null"], "description": "quantity of want_item for trade_offer"},
        "about": {"type": ["string", "null"], "description": "agent_id for gossip"},
        "rule_type": {"type": ["string", "null"], "enum": ["curfew", "wealth_tax", "repeal", "expel", "suspend_vote", "welcome", "festival", "water_blessing", "impeach", "elect", None], "description": "for propose_rule"},
        "convert_faith": {"type": ["string", "null"], "enum": ["vale_covenant", "hall_creed", "old_ways", None], "description": "for convert: the congregation receiving you"},
        "rule_after_tick_of_day": {"type": ["integer", "null"], "description": "for propose_rule curfew"},
        "rule_period": {"type": ["integer", "null"], "description": "for propose_rule curfew or wealth_tax"},
        "rule_tax_rate": {"type": ["number", "null"], "description": "for propose_rule wealth_tax, 0-1"},
        "rule_tax_threshold": {"type": ["number", "null"], "description": "for propose_rule wealth_tax"},
        "rule_target_proposal_id": {"type": ["integer", "null"], "description": "for propose_rule repeal: proposal_id of the enacted rule to remove"},
        "rule_target_agent_id": {"type": ["string", "null"], "description": "for propose_rule expel/suspend_vote/welcome/impeach/elect"},
        "proposal_id": {"type": ["integer", "null"], "description": "for vote or lobby"},
        "vote_choice": {"type": ["string", "null"], "enum": ["yes", "no", None], "description": "for vote, or lobby lean"},
        "invention_kind": {"type": ["string", "null"], "description": "catalog kind for invent, e.g. water_pump"},
        "invention_id": {"type": ["integer", "null"], "description": "for adopt_invention"},
        "say": {"type": ["string", "null"]},
        "reasoning": {
            "type": "string",
            "description": "One short sentence: why this action, in character.",
        },
    },
    "required": [
        "action", "destination", "to", "offer_id", "auction_id", "bid_amount",
        "give_item", "give_amount",
        "want_item", "want_amount", "about", "rule_type", "rule_after_tick_of_day",
        "rule_period", "rule_tax_rate", "rule_tax_threshold", "rule_target_proposal_id",
        "rule_target_agent_id", "proposal_id", "vote_choice", "invention_kind", "invention_id",
        "convert_faith", "say", "reasoning",
    ],
    "additionalProperties": False,
}

_SYSTEM_PROMPT = """You are role-playing one resident of a small simulated town. \
You will be given your current situation (Perception) and must choose ONE action \
to take this tick, responding ONLY with JSON matching the required schema.

The schema is FLAT: every possible field is listed, but only some apply to any \
given action. Set every field that doesn't apply to your chosen action to null.

Valid actions and which fields each one uses (all others should be null):
- move: destination
- work: (no fields needed)
- trade_offer: to, give_item, give_amount, want_item, want_amount
- trade_accept / trade_reject: offer_id
- trade_counter: offer_id, want_item=money, want_amount as the new price. Bargain, do not invent goods.
- bid: auction_id, bid_amount. Sealed. First bid sticks. Food lot is second-price; public work is lowest bid. You must be at the market and hold the coins.
- speak: to, optionally say
- gossip: about, optionally to, optionally say (make it specific: scandal, expulsion, welcome)
- propose_rule: rule_type ("curfew", "wealth_tax", "repeal", "expel", "suspend_vote",
  "welcome", "festival", "water_blessing", "impeach", or "elect").
  curfew: rule_after_tick_of_day + rule_period; wealth_tax: rule_tax_rate +
  rule_tax_threshold + rule_period; repeal: rule_target_proposal_id;
  expel/suspend_vote/welcome/impeach/elect: rule_target_agent_id; festival and
  water_blessing need no extra fields (the engine binds festival to YOUR faith).
  impeach only during a crisis, if the sitting lead is ruined, or if their
  reputation has collapsed. elect only when the chair is vacant; you cannot
  elect a bankrupt resident. A bankrupt sitting leader steps down without a vote.
- vote: proposal_id, vote_choice ("yes" or "no") — only if you have voting rights
- lobby: to (one agent here), proposal_id, vote_choice as the lean you want from them.
  Use this on deadlock, as town leader, or in a crisis. Convince people one by one.
- invent: invention_kind (must be a catalog kind from Perception, e.g. water_pump)
- adopt_invention: invention_id
- worship: no extra fields. Only at your faith home while the service is in session
  (or during your congregation's festival).
- convert: convert_faith. Only if you are unaffiliated or your piety has fallen
  below 0.22, only at a living service with at least two members present.
- idle: (no fields needed)

Stay in character based on your traits and recent memories. Be concise. \
Do NOT choose idle unless every other action is impossible. Idle is a last resort. \
You walk ONE street per tick. Name a destination; the engine walks the next \
open street toward it. You cannot teleport. A flood can close the bridge and \
greenways — then those places drop off your reachable list.
Religion is a civic institution, not doctrine you invent. You cannot author \
beliefs, rites, or scripture. Worship, convert, festival, and water_blessing \
are the only faith actions; the engine validates them. \
If the town is in a crisis (famine, unrest, bank_run, flood, drought, pollution), you MUST act: move to the \
farm and work, go to town_hall or the park and propose or vote, leave flooded \
ground, invent a catalog tool, speak or gossip to organize neighbors. Fight \
for the town the way a frightened person would. \
Respond with ONLY the JSON object, no other text."""


def _build_user_prompt(perception):
    """Render Perception to a flat text block for the prompt. Every
    Perception field is already flat/serializable (see decision.py's
    docstring on why) so this is a direct, unsurprising mapping -- no
    transformation logic that could silently drop information the model
    needs to reason about its situation.
    """
    p = perception
    lines = [
        f"You are agent {p.self_id}, currently at '{p.self_location}', tick {p.tick}.",
        f"Money: {p.self_money:.2f}. Inventory: {p.self_inventory}.",
        f"Your reputation in town: {p.self_reputation:.2f} (0=poor, 1=excellent).",
        f"Your traits (0-1 scale): industriousness={p.self_industriousness:.2f}, "
        f"generosity={p.self_generosity:.2f}, sociability={p.self_sociability:.2f}, "
        f"rule_respect={p.self_rule_respect:.2f}, risk_tolerance={p.self_risk_tolerance:.2f}.",
        f"Your livelihood: {getattr(p, 'self_livelihood_name', 'Laborer')} "
        f"(solvency {getattr(p, 'self_solvency', 'ok')}). "
        f"Flood and famine hit farmers hardest; a bank run hits traders; unrest hits laborers. "
        f"Bankruptcy does not take your vote — civic ballot {getattr(p, 'can_vote_civic', True)}, "
        f"economy ballot {getattr(p, 'can_vote_economy', True)}.",
        f"Crisis exposure this tick: {getattr(p, 'crisis_exposure', {})}.",
        f"Your faith: {getattr(p, 'self_faith_name', 'Unaffiliated')} "
        f"(piety {getattr(p, 'self_piety', 0):.2f}). Home: {getattr(p, 'faith_home', None)}. "
        f"Worship in session for you: {getattr(p, 'worship_now', False)}. "
        f"Congregation size {getattr(p, 'congregation_size', 0)}, "
        f"{getattr(p, 'congregation_here', 0)} of yours here.",
        f"Service in this room: {getattr(p, 'session_faith', None)} "
        f"({getattr(p, 'session_present', 0)} members present). "
        f"Census: {getattr(p, 'faith_census', {})}. "
        f"Festival: {getattr(p, 'festival_faith', None)}. "
        f"Water blessing enacted: {getattr(p, 'water_blessing', False)}.",
        f"Other agents here: {p.location_agents}.",
        f"Resources available to work here: {p.location_resources}.",
        f"Streets you can walk this tick (one hop): {getattr(p, 'street_neighbors', [])}.",
        f"Places still reachable: {getattr(p, 'reachable', [])}.",
        f"This place's character (1.0 is a street corner): {getattr(p, 'space', {})}.",
        f"Closed street kinds: {getattr(p, 'blocked_streets', [])}.",
        f"Currently active town rules: {p.active_rules}.",
    ]
    if p.active_crises:
        levels = getattr(p, "crisis_intensity", {}) or {}
        bits = [f"{c} (intensity {levels.get(c, 0.5):.2f})" for c in sorted(p.active_crises)]
        lines.append("CRISIS — the town is in danger: " + ", ".join(bits) + ".")
        lines.append("You must act this tick. Do not idle. Work, move to help, vote, invent, or organize.")
    if p.self_faction:
        faction_label = p.self_faction_name or p.self_faction
        lines.append(f"Your faction: {faction_label} (recent yes-rate: {p.faction_lean:.2f}).")
    if p.public_headlines:
        lines.append("Recent town headlines:")
        lines.extend(f"  - {h.get('text')}" for h in p.public_headlines)
    if getattr(p, "observed_problems", None):
        lines.append(f"Problems you could invent toward: {p.observed_problems}.")
    if getattr(p, "invention_catalog", None):
        lines.append("Invention catalog (only these kinds are valid): "
                     + ", ".join(f"{i['kind']} ({i['name']})" for i in p.invention_catalog) + ".")
    if getattr(p, "known_inventions", None):
        lines.append("Existing inventions: "
                     + ", ".join(f"#{i['id']} {i['name']}" for i in p.known_inventions) + ".")
    if p.notice_board:
        lines.append("Tavern notice board: " + ", ".join(
            slip.get("text", "") for slip in p.notice_board if slip.get("text")
        ) + ".")
    if p.speculation_buzz:
        lines.append(f"Trade rumors you have heard: {p.speculation_buzz}.")
    if p.enacted_proposals:
        lines.append(f"Enacted rules you could propose to repeal: {p.enacted_proposals}.")
    if p.relationships:
        lines.append(f"Your opinions of agents here (-1 to 1): {p.relationships}.")
    if p.recent_memories:
        lines.append("Recent things you remember:")
        lines.extend(f"  - {m}" for m in p.recent_memories)
    if p.pending_trade_offers:
        lines.append(f"Trade offers waiting for your response: {p.pending_trade_offers}.")
        lines.append("You may trade_counter with a new money price (Nash split), or accept, or reject. Two counters max.")
    net = getattr(p, "network", None) or {}
    if net:
        lines.append(
            f"Your social-graph position: degree {net.get('degree', 0)}, "
            f"betweenness {net.get('betweenness', 0)}, "
            f"clustering {net.get('clustering', 0)}, broker={net.get('broker')}."
        )
    town_net = getattr(p, "network_town", None) or {}
    if town_net:
        lines.append(f"Town network density {town_net.get('density')}, components {town_net.get('components')}.")
    if getattr(p, "open_auctions", None):
        lines.append(f"Open market auctions (bid only at market): {p.open_auctions}.")
    if getattr(p, "pivotal_votes", None):
        lines.append(f"Your ballot is PIVOTAL on proposals {p.pivotal_votes} — you can flip pass/fail.")
    if p.open_proposals:
        lines.append(f"Open governance proposals (quorum/pass/deadlock annotated): {p.open_proposals}.")
    if getattr(p, "can_vote", True) is False:
        lines.append("Your voting rights are suspended or you are expelled. You cannot vote or propose.")
    if getattr(p, "office_vacant", False) or not getattr(p, "town_leader_id", None):
        lines.append("The town-leader chair is VACANT. Propose elect naming rule_target_agent_id from succession candidates. A bankrupt name cannot take the chair.")
        cands = getattr(p, "succession_candidates", None) or []
        if cands:
            lines.append("Succession candidates: " + ", ".join(
                f"{c.get('name')} ({c.get('id')})" for c in cands
            ) + ".")
    elif getattr(p, "is_leader", False):
        lines.append(f"You are the town leader ({p.town_leader_name}). If a vote is deadlocked or the town is in crisis, lobby members one by one.")
        if getattr(p, "self_solvency", "ok") == "bankrupt":
            lines.append("You are bankrupt — the engine will sit you down; you do not keep the chair.")
    elif getattr(p, "town_leader_name", None):
        lines.append(f"Town leader: {p.town_leader_name} (solvency {getattr(p, 'leader_solvency', 'ok')}).")
        if getattr(p, "impeach_justified", False):
            lines.append("Impeach is in play: a crisis, a ruined lead, or collapsed reputation. Propose impeach with rule_target_agent_id set to the sitting leader.")
    if getattr(p, "is_faith_leader", False):
        lines.append(
            f"You lead the {getattr(p, 'self_faith_name', 'congregation')}. "
            "During service, go to your faith home and worship. "
            "You may propose festival or water_blessing; you cannot invent doctrine."
        )
    elif getattr(p, "faith_leader_name", None):
        lines.append(f"Your congregation's leader: {p.faith_leader_name}.")
    if getattr(p, "congregation_leaders", None):
        bits = [f"{row.get('faith_name')}: {row.get('name')}" for row in p.congregation_leaders]
        if bits:
            lines.append("Congregation leaders: " + "; ".join(bits) + ".")
    if getattr(p, "notorious", None):
        lines.append(f"Notorious residents (expel or suspend_vote): {p.notorious}.")
    if getattr(p, "newcomers", None):
        lines.append(f"New faces waiting to be welcomed onto the roll: {p.newcomers}.")
    if getattr(p, "lobby_targets", None):
        lines.append(f"People you could still lobby: {p.lobby_targets}.")
    lines.append("\nChoose your action for this tick. Respond with ONLY the JSON object.")
    return "\n".join(lines)


class LLMDecider:
    """Drop-in replacement for RuleBasedDecider, implementing the same
    `Decider` protocol. One shared rate limiter is used across ALL
    LLMDecider instances in a process (see `_shared_limiter` below)
    because Groq's RPM cap applies at the organization/API-key level,
    not per agent -- giving each agent its own limiter would let 16
    agents each independently believe they have 24 RPM, instantly
    blowing the real shared budget by 16x.
    """

    # Exposed as a class attribute (not just the module-level constant
    # above) specifically so main_llm.py can read LLMDecider.BUILD_VERSION
    # and cross-check it against its own BUILD_VERSION at startup.
    BUILD_VERSION = BUILD_VERSION

    # Class-level (shared across every LLMDecider instance), not
    # instance-level, for the reason above: this MUST be one bucket per
    # process, mirroring Groq's actual per-organization limit.
    _shared_limiter = None

    def __init__(self, api_key=None, rpm=RECOMMENDED_RPM_SAFETY_MARGIN, verbose=True,
                 use_strict_schema=False, model=None):
        """
        Args:
            api_key: Groq API key. If omitted, read from the
                GROQ_API_KEY environment variable. Raises ValueError if
                neither is provided -- fails at construction time, not
                on the first `decide()` call, so a missing key is caught
                immediately rather than after the simulation is already
                running.
            rpm: requests/minute budget for the SHARED rate limiter (see
                class docstring -- this is shared across every
                LLMDecider instance in the process, not per-instance).
                Only takes effect the first time any LLMDecider is
                constructed; later instances reuse the existing shared
                limiter and ignore their own `rpm` argument.
            verbose: if True, print rate-limit waits, the model's chosen
                action and reasoning, and any parse/API errors to
                stdout. Useful for watching a run; set False for quiet
                batch use.
            use_strict_schema: if True, attempt Groq's strict
                json_schema mode first (with a fallback to json_object
                on failure, as before). Defaults to FALSE as of this
                version -- a real correction, not the original design.
                TWO consecutive live runs against `openai/gpt-oss-120b`
                failed json_validate_failed on 100% of calls (not the
                ~10% intermittent rate originally assumed from a single
                Groq community report), and a broader search turned up
                multiple corroborating, currently-open bug reports
                (Groq's own forum AND a December-2025 LangChain GitHub
                issue) describing this exact model silently ignoring
                strict mode, returning blank completions when uncertain,
                or failing validation outright on Groq's hosting. Given
                that weight of evidence, defaulting to the strict path
                and treating failure as the exception was the wrong
                default. json_object mode is far more broadly reliable
                in practice; `_parse_intent`'s existing defensive
                validation (unknown action -> idle, malformed args ->
                coerced, invalid JSON -> idle) carries the
                schema-conformance burden that strict mode would
                otherwise guarantee. Set this to True only to
                deliberately re-test strict mode (e.g. if Groq ships a
                fix) -- the fallback machinery below still exists and
                still engages correctly if you do.
        """
        api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "No Groq API key found. Pass api_key=... or set the "
                "GROQ_API_KEY environment variable. Get a free key at "
                "https://console.groq.com/keys"
            )
        # Imported here, not at module level, so rule-based entry
        # points do not require the groq package.
        require_groq()
        from groq import Groq
        self.client = Groq(api_key=api_key)
        self.verbose = verbose
        self.model = resolve_model(model)

        if LLMDecider._shared_limiter is None:
            LLMDecider._shared_limiter = TokenBucketRateLimiter(max_per_minute=rpm)

        # Tracks whether THIS instance has discovered that `MODEL`
        # rejects strict json_schema mode (a real 400 error, confirmed
        # against the live API -- see MODEL's docstring above for the
        # history). Once discovered, every subsequent call uses the
        # more broadly-supported json_object fallback instead, rather
        # than re-attempting and re-failing json_schema mode on every
        # single tick. Per-instance (not class-level like the rate
        # limiter) because different LLMDecider instances could in
        # principle be configured with different models in the future.
        #
        # Initialized to the OPPOSITE of `use_strict_schema`: if strict
        # mode wasn't explicitly requested, start in json_object mode
        # from the very first call rather than wastefully attempting
        # and failing json_schema first (see use_strict_schema's
        # docstring above for why this is the new default).
        self._use_json_object_fallback = not use_strict_schema
        self._rules = None
        self._last_source = "llm"
        self._last_model_intent = None
        self._last_fallback_reason = None

    def _rule_fallback(self, agent_id, perception, why: str):
        """When the model idles or fails, still act like a resident."""
        from decision import RuleBasedDecider
        if self._rules is None:
            self._rules = RuleBasedDecider()
        if self.verbose:
            print(f"  [llm fallback] {agent_id}: {why} — rule-based action instead")
        self._last_source = "fallback"
        self._last_fallback_reason = why
        return self._rules.decide(agent_id, perception)

    def _after_llm(self, agent_id, perception, intent):
        from decision_record import intent_as_dict
        self._last_model_intent = intent_as_dict(intent)
        if intent.action != "idle":
            self._last_source = "llm"
            self._last_fallback_reason = None
            return intent
        return self._rule_fallback(agent_id, perception, "model returned idle")

    def deliberate(self, agent_id, perception):
        """Same as decide(), plus a forensic draft for the six-stage record."""
        from decision import DecisionDraft
        self._last_source = "llm"
        self._last_model_intent = None
        self._last_fallback_reason = None
        intent = self.decide(agent_id, perception)
        considered = [{
            "action": intent.action,
            "reason": intent.say or self._last_fallback_reason or "llm choice",
            "weight": 1.0,
        }]
        if self._last_model_intent and self._last_source == "fallback":
            considered.insert(0, {
                "action": (self._last_model_intent or {}).get("action") or "idle",
                "reason": "model proposed — engine used the rule fallback",
                "weight": 0.4,
            })
        return DecisionDraft(
            intent=intent,
            retrieved_memories=list(perception.recent_memories),
            considered_actions=considered,
            source=self._last_source,
            model_intent=self._last_model_intent,
            fallback_reason=self._last_fallback_reason,
        )

    def decide(self, agent_id, perception):
        """Implements the Decider protocol (see decision.Decider).
        Blocks on the shared rate limiter, then calls Groq's chat
        completions API with structured-output JSON schema mode (or, if
        this model has already been found not to support it -- see
        `_use_json_object_fallback` -- the more broadly-supported
        json_object mode instead), constraining the response to the
        Intent shape as closely as the active mode allows. On any
        unrecoverable failure (rate limit, network error, malformed
        response), falls back to an "idle" Intent rather than raising --
        see this module's docstring and actions.py's fail-safe
        philosophy for why.
        """
        waited = LLMDecider._shared_limiter.wait_for_token()
        if self.verbose and waited > 0.5:
            print(f"  [rate limit] {agent_id} waited {waited:.1f}s for an LLM call slot")

        response_format = self._json_object_format() if self._use_json_object_fallback \
            else self._json_schema_format()

        try:
            extra = {"reasoning_effort": "low"} if "gpt-oss" in self.model else {}
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": _build_user_prompt(perception)},
                ],
                response_format=response_format,
                temperature=0.7,
                max_completion_tokens=500,
                **extra,
            )
            raw = response.choices[0].message.content
            return self._after_llm(agent_id, perception, self._parse_intent(agent_id, raw))
        except Exception as exc:
            if not self._use_json_object_fallback and self._is_unsupported_json_schema_error(exc):
                # This is the exact failure mode a real run against
                # llama-3.3-70b-versatile produced: a 400 saying the
                # model doesn't support response_format json_schema.
                # Rather than idle forever on every future call too,
                # downgrade ONCE to json_object mode and retry this
                # same tick -- json_object mode is supported far more
                # broadly (see this module's MODEL docstring) and still
                # gets us JSON, just without the strict schema guarantee,
                # so _parse_intent's existing defensive parsing carries
                # the rest of the safety burden from here on.
                if self.verbose:
                    print(f"  [llm fallback] {agent_id}: model rejected json_schema mode "
                          f"-- switching to json_object mode for future calls")
                self._use_json_object_fallback = True
                return self.decide(agent_id, perception)

            if self._is_json_validate_failed_error(exc) and response_format.get("type") == "json_schema":
                # DIFFERENT failure mode from the one above, confirmed
                # against a real run plus Groq's own community forum:
                # even on models that genuinely support strict mode
                # (openai/gpt-oss-120b is on the supported list), Groq
                # has an acknowledged, intermittent (~10% per their bug
                # tracker) reliability gap where constrained decoding
                # itself fails validation -- typically because the
                # generation got cut off, often missing just one field.
                # This is NOT the same as "model doesn't support
                # json_schema at all" (handled above) -- it's a
                # per-request flake on an otherwise-working setup, so
                # the right response is a ONE-TIME retry in json_object
                # mode for just this call, not a permanent instance-wide
                # downgrade like the branch above. If json_object mode
                # ALSO fails (the nested exception handling here doesn't
                # recurse further), this falls through to idle below
                # rather than retrying indefinitely.
                #
                # This branch only matters now if use_strict_schema=True
                # was explicitly passed (see __init__'s docstring -- the
                # default changed to json_object mode after two
                # consecutive live runs failed 100% of strict-mode calls,
                # not the ~10% originally assumed from a single Groq
                # community report). Kept as a real fallback rather than
                # removed, since Groq may fix this and someone may want
                # to re-test with use_strict_schema=True.
                if self.verbose:
                    print(f"  [llm retry] {agent_id}: strict mode validation failed "
                          f"-- retrying this call in json_object mode")
                try:
                    extra = {"reasoning_effort": "low"} if "gpt-oss" in self.model else {}
                    retry_response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": _SYSTEM_PROMPT},
                            {"role": "user", "content": _build_user_prompt(perception)},
                        ],
                        response_format=self._json_object_format(),
                        temperature=0.7,
                        max_completion_tokens=500,
                        **extra,
                    )
                    raw = retry_response.choices[0].message.content
                    return self._after_llm(agent_id, perception, self._parse_intent(agent_id, raw))
                except Exception as retry_exc:
                    if self.verbose:
                        print(f"  [llm error] {agent_id}: retry also failed "
                              f"({retry_exc!r}) — using rule-based action")
                    return self._rule_fallback(agent_id, perception, "retry failed")

            # Fail safe: never crash the town. If the model cannot
            # answer, a rule-based survival choice still runs so a
            # crisis is not met with a row of idle agents.
            if self.verbose:
                print(f"  [llm error] {agent_id}: {exc!r} — using rule-based action")
            return self._rule_fallback(agent_id, perception, "api error")

    @staticmethod
    def _json_schema_format() -> dict:
        """The strict structured-output request format: guarantees
        schema-conformant output, but only on models that support it
        (see MODEL's docstring).
        """
        return {
            "type": "json_schema",
            "json_schema": {"name": "intent", "schema": _INTENT_SCHEMA, "strict": True},
        }

    @staticmethod
    def _json_object_format() -> dict:
        """The broadly-supported fallback request format: asks for valid
        JSON, with no schema guarantee. `_SYSTEM_PROMPT` already
        instructs the model to respond with JSON matching the documented
        shape, which satisfies json_object mode's requirement that the
        word "JSON" appear in the prompt. `_parse_intent`'s existing
        defensive validation (unknown action -> idle, non-dict args ->
        coerced, invalid JSON -> idle) carries the schema-conformance
        burden that strict mode would otherwise guarantee.
        """
        return {"type": "json_object"}

    @staticmethod
    def _is_unsupported_json_schema_error(exc: Exception) -> bool:
        """Detect the specific 400 error Groq returns when a model
        doesn't support response_format json_schema, distinguishing it
        from other failures (rate limits, network errors, genuinely
        malformed requests) that should NOT trigger a mode downgrade --
        only this exact, confirmed failure mode should.
        """
        message = str(exc).lower()
        return "response format" in message and "json_schema" in message

    @staticmethod
    def _is_json_validate_failed_error(exc: Exception) -> bool:
        """Detect Groq's `json_validate_failed` error code -- a DIFFERENT
        failure from `_is_unsupported_json_schema_error` above. This one
        fires even on models that genuinely support strict mode: Groq's
        own community forum (and a real run against this codebase)
        confirm it's an intermittent reliability gap in their
        constrained decoding itself (reported around 10% of requests on
        otherwise-working setups), not a hard incompatibility. Detected
        separately so it triggers a one-time per-call retry rather than
        the permanent instance-wide mode downgrade the other error
        triggers -- conflating the two would either retry forever on a
        truly unsupported model, or permanently abandon strict mode
        after a single transient Groq-side flake.
        """
        message = str(exc).lower()
        return "json_validate_failed" in message or "failed to validate json" in message

    def _parse_intent(self, agent_id, raw):
        """Parse the model's JSON response into an Intent. Even with
        strict JSON-schema mode (which should make this close to
        guaranteed-valid), this stays defensive: a free-tier model on a
        cost-saving inference stack is still a place malformed output
        can slip through, and the contract established in actions.py
        (illegal/malformed Intents fail safe, never crash) should hold
        here too, not just at execution time.

        Reconstructs the nested `args` dict that actions.py/economy.py
        expect (e.g. {"give": {"food": 2}, "want": {"money": 5}} for a
        trade_offer) from the FLAT fields the schema actually uses (see
        _INTENT_SCHEMA's docstring for why flat, not nested) -- this is
        the one place that translation happens, so nothing downstream
        of `Intent` needed to change when the schema was flattened.
        """
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError) as exc:
            if self.verbose:
                print(f"  [llm parse error] {agent_id}: invalid JSON ({exc}) -- raw: {raw!r}")
            return Intent(action="idle", args={}, say=None)

        action = data.get("action", "idle")
        if action not in _VALID_ACTIONS:
            if self.verbose:
                print(f"  [llm parse error] {agent_id}: unknown action {action!r} -- idling")
            action = "idle"

        args = self._build_args(action, data)

        if self.verbose:
            reasoning = data.get("reasoning", "")
            print(f"  [llm] {agent_id}: {action} ({reasoning})")

        return Intent(action=action, args=args, say=data.get("say"))

    @staticmethod
    def _build_args(action: str, data: dict) -> dict:
        """Reconstruct the nested `args` dict actions.py/economy.py
        expect, from the flat schema fields in `data`. Only includes
        keys relevant to `action` -- e.g. a "move" Intent's args will
        contain only "destination", never the trade or voting fields,
        even though `data` itself has all fields present (as null, per
        the schema). Defensively uses `.get()` throughout since `data`
        may not be a dict at all if json_object fallback mode produced
        something unexpected (see `_json_object_format`'s docstring).

        Unknown/"idle" actions get {} -- there's nothing to reconstruct.
        """
        if not isinstance(data, dict):
            return {}

        if action == "move":
            return {"destination": data.get("destination")}
        if action == "trade_offer":
            give_item = data.get("give_item")
            want_item = data.get("want_item")
            args = {"to": data.get("to")}
            if give_item:
                args["give"] = {give_item: data.get("give_amount") or 0}
            if want_item:
                args["want"] = {want_item: data.get("want_amount") or 0}
            return args
        if action in ("trade_accept", "trade_reject"):
            return {"offer_id": data.get("offer_id")}
        if action == "trade_counter":
            args = {"offer_id": data.get("offer_id")}
            if data.get("want_item"):
                args["want"] = {data.get("want_item"): data.get("want_amount") or 0}
            return args
        if action == "bid":
            return {"auction_id": data.get("auction_id"), "amount": data.get("bid_amount")}
        if action == "speak":
            return {"to": data.get("to")}
        if action == "gossip":
            args = {"about": data.get("about")}
            if data.get("to"):
                args["to"] = data.get("to")
            return args
        if action == "propose_rule":
            rule_type = data.get("rule_type")
            rule_args = {}
            if rule_type == "curfew":
                rule_args = {
                    "after_tick_of_day": data.get("rule_after_tick_of_day"),
                    "period": data.get("rule_period"),
                }
            elif rule_type == "wealth_tax":
                rule_args = {
                    "rate": data.get("rule_tax_rate"),
                    "threshold": data.get("rule_tax_threshold"),
                    "period": data.get("rule_period"),
                }
            elif rule_type == "repeal":
                rule_args = {
                    "target_proposal_id": data.get("rule_target_proposal_id"),
                }
            elif rule_type in ("expel", "suspend_vote", "welcome", "impeach", "elect"):
                rule_args = {
                    "target_agent": data.get("rule_target_agent_id"),
                }
            elif rule_type in ("festival", "water_blessing"):
                rule_args = {}
            return {"rule_type": rule_type, "rule_args": rule_args}
        if action == "convert":
            return {"faith": data.get("convert_faith")}
        if action == "vote":
            return {"proposal_id": data.get("proposal_id"), "choice": data.get("vote_choice")}
        if action == "lobby":
            return {
                "to": data.get("to"),
                "proposal_id": data.get("proposal_id"),
                "lean": data.get("vote_choice"),
            }
        if action == "invent":
            return {"invention_kind": data.get("invention_kind")}
        if action == "adopt_invention":
            return {"invention_id": data.get("invention_id")}
        # work, idle, or anything unrecognized: no args needed.
        return {}
