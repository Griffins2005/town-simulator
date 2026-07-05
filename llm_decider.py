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
    pip install groq
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
MODEL = "openai/gpt-oss-120b"


# JSON Schema for Intent, used with Groq's structured-output mode (see
# https://console.groq.com/docs/structured-outputs). This guarantees the
# model's response parses as valid JSON matching this exact shape --
# eliminating the "regex prose out of a chat response" failure mode that
# sinks most LLM-agent prototypes. `action` is constrained to the exact
# set actions.py's _REGISTRY recognizes (kept here as a literal list
# rather than importing actions.py, to avoid this module depending on
# the whole action-execution stack just to know action NAMES).
_VALID_ACTIONS = [
    "move", "work", "trade_offer", "trade_accept", "trade_reject",
    "speak", "gossip", "propose_rule", "vote", "idle",
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
        "offer_id": {"type": ["integer", "null"], "description": "for trade_accept/trade_reject"},
        "give_item": {"type": ["string", "null"], "description": "item name for trade_offer, e.g. food"},
        "give_amount": {"type": ["number", "null"], "description": "quantity of give_item for trade_offer"},
        "want_item": {"type": ["string", "null"], "description": "item name wanted in return, e.g. money"},
        "want_amount": {"type": ["number", "null"], "description": "quantity of want_item for trade_offer"},
        "about": {"type": ["string", "null"], "description": "agent_id for gossip"},
        "rule_type": {"type": ["string", "null"], "enum": ["curfew", "wealth_tax", "repeal", None], "description": "for propose_rule"},
        "rule_after_tick_of_day": {"type": ["integer", "null"], "description": "for propose_rule curfew"},
        "rule_period": {"type": ["integer", "null"], "description": "for propose_rule curfew or wealth_tax"},
        "rule_tax_rate": {"type": ["number", "null"], "description": "for propose_rule wealth_tax, 0-1"},
        "rule_tax_threshold": {"type": ["number", "null"], "description": "for propose_rule wealth_tax"},
        "rule_target_proposal_id": {"type": ["integer", "null"], "description": "for propose_rule repeal: proposal_id of the enacted rule to remove"},
        "proposal_id": {"type": ["integer", "null"], "description": "for vote"},
        "vote_choice": {"type": ["string", "null"], "enum": ["yes", "no", None], "description": "for vote"},
        "say": {"type": ["string", "null"]},
        "reasoning": {
            "type": "string",
            "description": "One short sentence: why this action, in character.",
        },
    },
    "required": [
        "action", "destination", "to", "offer_id", "give_item", "give_amount",
        "want_item", "want_amount", "about", "rule_type", "rule_after_tick_of_day",
        "rule_period", "rule_tax_rate", "rule_tax_threshold", "rule_target_proposal_id",
        "proposal_id", "vote_choice", "say", "reasoning",
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
- speak: to, optionally say
- gossip: about, optionally to, optionally say
- propose_rule: rule_type ("curfew", "wealth_tax", or "repeal"), and for curfew:
  rule_after_tick_of_day + rule_period; for wealth_tax:
  rule_tax_rate + rule_tax_threshold + rule_period; for repeal:
  rule_target_proposal_id (must match one of enacted_proposals in Perception)
- vote: proposal_id, vote_choice ("yes" or "no")
- idle: (no fields needed)

Stay in character based on your traits and recent memories. Be concise. \
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
        f"Other agents here: {p.location_agents}.",
        f"Resources available to work here: {p.location_resources}.",
        f"Currently active town rules: {p.active_rules}.",
    ]
    if p.active_crises:
        lines.append(f"Active town crises: {sorted(p.active_crises)}.")
    if p.self_faction:
        lines.append(f"Your faction: {p.self_faction} (recent yes-rate: {p.faction_lean:.2f}).")
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
    if p.open_proposals:
        lines.append(f"Open governance proposals you can vote on: {p.open_proposals}.")
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
                 use_strict_schema=False):
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
        # Imported here, not at module level, so the rest of this
        # codebase (and anything that imports decision.py, which every
        # module does) doesn't require the `groq` package installed
        # just to run Phase 1's RuleBasedDecider path.
        from groq import Groq
        self.client = Groq(api_key=api_key)
        self.verbose = verbose

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
            response = self.client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": _build_user_prompt(perception)},
                ],
                response_format=response_format,
                temperature=0.7,
                # openai/gpt-oss-120b is a reasoning model that, by
                # default, spends tokens on hidden chain-of-thought
                # (Groq's docs: included in the response's `reasoning`
                # field, controllable via reasoning_effort) BEFORE
                # producing the final JSON. Under a fixed token budget,
                # that hidden reasoning competes with the actual answer
                # for the remaining tokens -- a real, plausible
                # contributor to the empty/truncated completions seen in
                # live runs. 'low' minimizes that overhead; this
                # decision doesn't need deep multi-step reasoning, just
                # a same-tick choice from a fixed action menu.
                reasoning_effort="low",
                # Raised from an initial 300 after a real run's flat,
                # nullable-field-heavy schema (16 fields + reasoning,
                # most of them "null" for any given action) combined
                # with a token-budget-limited generation to plausibly
                # produce truncated, validation-failing JSON -- see the
                # json_validate_failed handling below for the broader
                # context. More headroom costs a little latency but
                # removes one concrete way to truncate mid-object.
                max_completion_tokens=500,
            )
            raw = response.choices[0].message.content
            return self._parse_intent(agent_id, raw)
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
                    retry_response = self.client.chat.completions.create(
                        model=MODEL,
                        messages=[
                            {"role": "system", "content": _SYSTEM_PROMPT},
                            {"role": "user", "content": _build_user_prompt(perception)},
                        ],
                        response_format=self._json_object_format(),
                        temperature=0.7,
                        reasoning_effort="low",
                        max_completion_tokens=500,
                    )
                    raw = retry_response.choices[0].message.content
                    return self._parse_intent(agent_id, raw)
                except Exception as retry_exc:
                    if self.verbose:
                        print(f"  [llm error] {agent_id}: retry also failed "
                              f"({retry_exc!r}) -- falling back to idle")
                    return Intent(action="idle", args={}, say=None)

            # Fail safe, exactly like actions.py does for illegal
            # Intents: a network error, a 429 that slipped through, a
            # malformed response -- none of it should crash the
            # simulation. The agent just idles this tick and we log why,
            # the same philosophy as actions.py's ActionResult(False, ...).
            if self.verbose:
                print(f"  [llm error] {agent_id}: {exc!r} -- falling back to idle")
            return Intent(action="idle", args={}, say=None)

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
            return {"rule_type": rule_type, "rule_args": rule_args}
        if action == "vote":
            return {"proposal_id": data.get("proposal_id"), "choice": data.get("vote_choice")}
        # work, idle, or anything unrecognized: no args needed.
        return {}
