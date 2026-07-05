# townsim — a turn-based agent-society simulator

Agents that trade, vote, gossip, form reputations, embezzle, panic, and
form political factions in a small simulated town. Phase 1 is a free,
deterministic, rule-based engine; Phase 2 swaps in real LLM reasoning
(Groq, free tier) behind the exact same interface; Phase 3 records a run
to a portable trace file for an interactive browser-based replay (map
view, live charts, click-to-inspect agent details); Phase 4 adds
political/economic chaos (corruption, market shocks, bank runs, unrest,
factions, rule repeal) and wires it into real cross-sector feedback
loops with the existing economy/politics/norms systems; Phase 5 adds a
true live web visualization -- a local server streams the simulation to
a browser in real time, tick by tick, while it's actually running.

## File inventory (18 source files + this README + requirements.txt)

Phase 1 -- core engine, zero external dependencies:
- `world.py`, `agent.py`, `memory.py` -- state primitives
- `decision.py` -- the Perception/Intent contract + RuleBasedDecider
- `actions.py` -- the only code that mutates state
- `economy.py`, `governance.py` -- ledger/trade and proposals/voting
- `engine.py` -- the step() loop and per-tick housekeeping
- `town_factory.py` -- shared town construction
- `main.py` -- short demo (60 ticks)
- `stress_test.py` -- long-run harness (1000 ticks, time-series metrics)

Phase 2 -- LLM-backed agents, requires `groq`:
- `rate_limiter.py` -- token-bucket limiter for Groq's free-tier RPM cap
- `llm_decider.py` -- LLMDecider (same Decider protocol, real LLM calls)
- `main_llm.py` -- entry point mixing LLM-backed and rule-based agents

Phase 3 -- recording for visualization, zero external dependencies:
- `recorder.py` -- wraps an Engine, captures a per-tick trace to JSON
- `record_demo.py` -- entry point: runs a town, saves `trace.json`

Phase 4 -- chaos and cross-sector intersections, zero external dependencies:
- `chaos.py` -- corruption, market shocks, bank runs, speculation,
  unrest, faction formation, and the wiring connecting them to each
  other and to the existing economy/politics/norms systems

Phase 5 -- true live visualization, zero external dependencies (stdlib only):
- `live_server.py` -- local web server: runs the simulation in a
  background thread, streams one frame per tick to any connected
  browser over Server-Sent Events

If any of these 18 `.py` files are missing from your copy, something
didn't extract correctly -- every file listed above is required for at
least one of the five entry points (`main.py`, `stress_test.py`,
`main_llm.py`, `record_demo.py`, `live_server.py`) to run.

## Run it

    python3 main.py            # short demo: 60 ticks, full trace
    python3 stress_test.py     # long-run harness: 1000 ticks, time-series metrics
    python3 record_demo.py     # records 200 ticks to trace.json for visualization
    python3 live_server.py     # true live web visualization at http://localhost:8765/

Fully reproducible (`SEED = 7` in all five entry points) -- same seed,
same trace, every time. This includes `chaos.py`'s randomized events
(market shocks, corruption) as of the fix described in "Phase 5"
below -- verified directly by diffing two independent runs.

## Why no LLM yet

Phase 1's entire job is to validate the *mechanics* (norms, governance,
economy, social activity) using a cheap, deterministic stand-in brain,
so that when Phase 2 swaps in a free-tier LLM, you're debugging the LLM
integration in isolation -- not simultaneously debugging "is the ledger
right" and "is the API call right." See `decision.py`'s module docstring
for the full reasoning.

## Architecture

```
world.py       -- state container: locations, clock, active_rules, treasury
agent.py       -- persona/traits, inventory, relationships, reputation
memory.py      -- append-only per-agent event log (structured, not prose)
decision.py    -- THE SEAM: Perception -> Intent. Swap RuleBasedDecider
                  for an LLM-backed Decider here in Phase 2; nothing else changes.
actions.py     -- the ONLY code that mutates state. Validates + executes
                  every Intent, gives governance real enforcement teeth.
economy.py     -- ledger, scarcity-based work w/ regen, two-step trade
                  offer/resolve, demurrage, treasury redistribution
governance.py  -- propose -> vote -> tally -> enact, with THREE rule
                  types (curfew, wealth_tax, repeal) proven end-to-end
engine.py      -- the step() loop, sparse-thinking interrupt logic, and
                  per-tick housekeeping (regen, decay, demurrage, tax,
                  corruption, market shocks, crisis tracking, factions);
                  takes an injected, seeded `rng` for reproducible chaos
town_factory.py -- shared town/agent construction, used by every entry point
main.py        -- short runnable demo + aggregate stats
stress_test.py -- long-run harness with time-series metrics (Gini,
                  reputation spread, location entropy, active crises,
                  faction count) for catching degenerate equilibria
                  before they're expensive to find
recorder.py    -- wraps an Engine from the OUTSIDE (same seam discipline
                  as decision.py/llm_decider.py); captures a per-tick
                  trace to a JSON file for the browser-based visualizer
record_demo.py -- entry point: builds a town, records 200 ticks, saves trace.json
chaos.py       -- corruption, market shocks, bank runs, speculation,
                  unrest, faction formation -- the cross-sector feedback
                  loops connecting economy/politics/norms to each other
live_server.py -- local web server (stdlib only): runs the simulation
                  live in a background thread, streams it to a browser
                  over Server-Sent Events as it advances
```

## The one invariant that matters most

State (`agent.money`, `agent.inventory`, `agent.location`,
`world.active_rules`, etc.) is only ever mutated inside `actions.py`
(or `economy.py`/`governance.py`, which `actions.py` delegates to).
Every other module reads state; only those write it. This is what
keeps a malformed or hallucinated LLM decision (inevitable in Phase 2)
from corrupting the simulation -- it just fails one action cleanly.

## What's validated by the current run (60 ticks, 16 agents)

- **Economy**: real trades, real money/goods moving through a
  re-validated ledger (stale offers correctly fail rather than
  fabricating value), AND a verified conservation invariant: total
  system money (agent balances + treasury) stays exactly constant --
  nothing is silently created or destroyed.
- **Governance**: three rule types (`curfew`, `wealth_tax`, `repeal`)
  share the same propose/vote/tally/enact pipeline. `curfew` and
  `wealth_tax` get proposed, voted on, and -- when passed -- actually
  take effect (curfew blocks movement; wealth_tax collects and
  redistributes through the treasury). `repeal` removes a previously
  enacted rule's keys. Multiple rules can be active simultaneously,
  and enacted rules can later be voted back out via repeal.
- **Norms**: reputation moves with trade compliance and rule violation,
  and decays back toward neutral over time rather than ratcheting to a
  permanent 0.0/1.0 extreme.
- **Social activity**: gossip propagates reputation-relevant information
  to agents who never directly witnessed the underlying event.
- **Spatial distribution**: agents circulate across all four locations
  for the full length of a 1000-tick run rather than collapsing into
  one room.

## Stress-test findings (1000+ ticks) and what they led to

A short demo can't distinguish "the mechanics work" from "the mechanics
work for a while and then degenerate." Running `stress_test.py` for
1000 ticks surfaced four real problems, each found from a concrete
number going somewhere it shouldn't, not from guessing:

1. **Farm resource permanently depleted.** `food` at the farm hit 0.0 by
   tick ~100 and stayed there for 900 ticks -- the original model had
   extraction with no renewal. **Fix:** `economy.regenerate_resources`,
   a slow per-tick regrowth toward a cap, called every tick before
   agent actions.

2. **Reputation hit hard 0.0/1.0 caps and never recovered**, by tick
   ~450, because the only reputation mutations (trade completion:
   +0.01, curfew violation: -0.02) had no opposing force. **Fix:**
   `engine.py` decays every agent's reputation a small step toward 0.5
   each tick.

3. **Wealth concentrated under a price-blind trade mechanism.**
   `_respond_to_trade` originally ignored the offer's actual price and
   accepted/rejected on a flat trait-weighted coin flip, so
   low-generosity (high-price) sellers had no market discipline and
   accumulated money fastest. **Fix:** `decision.py`'s
   `_respond_to_trade` now compares the offered price-per-unit against
   a fair reference price; confirmed by re-checking the wealth/trait
   correlation before and after (it flipped from "low generosity wins"
   to no clear correlation). Even with this fixed, **wealth still
   concentrated from pure compounding variance** over long runs (Gini
   trending toward ~0.7) -- a real, separate phenomenon (akin to
   power-law wealth concentration from compounding luck in unregulated
   markets), addressed deliberately rather than engineered away: see
   `economy.DEMURRAGE_RATE` (mild, automatic, no vote needed) and
   `governance`'s `wealth_tax` rule type (heavier, requires the town to
   vote it in) for the two countervailing levers now available.

4. **The market became a one-way absorbing trap.** Circulation
   (`move`) was the LOWEST-priority action in `RuleBasedDecider`, so an
   agent that always had someone to talk to or trade with at market
   essentially never reached the branch that would move it elsewhere --
   `location_entropy` collapsed to ~0.17 by tick 1000. **Fix:** a flat
   `WANDERLUST_CHANCE` check in `decision.py`, evaluated BEFORE
   social/trade priorities, gives circulation an independent chance to
   win each tick regardless of what else is available.

5. **(Caught and fixed during the wealth-tax work, not by the stress
   test itself.)** The first version of demurrage destroyed the
   collected money outright with no beneficiary. Explicitly checking
   total system money supply (agent balances + treasury) over 1000
   ticks caught it shrinking monotonically toward zero with no source
   ever replenishing it. **Fix:** demurrage now routes into
   `world.treasury`, which periodically redistributes back to the
   population (`economy.redistribute_treasury_if_due`), closing the
   loop. Re-verified: total system money now holds exactly constant
   across a full run.

The throughline: every fix here came from a specific number that didn't
match what the underlying model should produce (a depleted-forever
resource, a capped-forever reputation, a wealth/trait correlation
running backwards, a collapsing entropy metric, a shrinking money
supply) -- not from aesthetic judgment about what "looked right."
`stress_test.py` is built to keep surfacing exactly this kind of thing;
re-run it after any future change to `decision.py`, `economy.py`, or
`governance.py`.

## Known limitations (intentional, for now)

- `RuleBasedDecider`'s trade pricing and voting are intentionally crude
  (a few trait-weighted heuristics, now price-aware on the accept/reject
  side). This is correct for Phase 1's job -- exercising the pipes --
  and is exactly the layer Phase 2's LLM should replace with real
  reasoning over `Perception.recent_memories`, `relationships`, etc.
- A vote, once cast, can be silently overwritten by a repeat vote from
  the same agent (governance.py's `votes` dict just keys by agent_id).
  Harmless under sparse-thinking (an agent won't re-vote on something
  it's already voted on -- see engine.py's `_should_think`), but worth
  knowing if you add a Decider that might deliberately reconsider.
- Rule repeal exists (`governance.py`'s `"repeal"` rule_type, wired
  through `RuleBasedDecider` and stress-tested), but a short 60-tick
  demo may not show one -- repeal proposals only arise when something
  is already enacted and the proposing agent has low `rule_respect`.
- `WANDERLUST_CHANCE` is a flat constant, not trait-weighted. It's a
  blunt fix for a priority-ordering bug, not a personality trait --
  Phase 2's LLM agents should decide whether to move on through actual
  reasoning about competing needs, not a coin flip.

## Phase 2: LLM-backed agents (Groq, free tier)

```
rate_limiter.py -- token-bucket limiter for the shared Groq RPM budget
llm_decider.py  -- LLMDecider: same Decider protocol, backed by Groq's
                   structured-output (JSON schema) mode
main_llm.py     -- entry point: a SMALL number of LLM-backed agents
                   mixed into an otherwise rule-based town
```

### Setup

    pip install -r requirements.txt        # only needed for Phase 2 (groq)
    export GROQ_API_KEY=your_key_here       # free, no credit card: console.groq.com/keys
    python3 main_llm.py

Important: `export` only applies to the current shell session and
anything launched from it. If you set the key in one terminal window
and run the script from another, or in a fresh session, the key won't
be there. `main_llm.py` now checks the key against a real (free) Groq
call before doing anything else and will say clearly if it's missing
or invalid, rather than failing silently tick after tick.

**Check the version banner.** The very first line `main_llm.py` prints
is `townsim build: <date>-<tag>`. If a run's behavior doesn't match
what this README describes, check that line first -- it's the fastest
way to confirm you're running the files actually being discussed here,
not a stale local copy from an earlier extraction (this exact problem
caused several rounds of confusing, seemingly-unfixable errors before
the banner existed; see "A sixth run" below). If `main_llm.py` and
`llm_decider.py` ever have mismatched versions -- e.g. you replaced one
file but not the other -- the script will say so explicitly and refuse
to continue, rather than silently misbehaving.

By default, `LLMDecider` uses Groq's `json_object` mode, not strict
`json_schema` mode -- see "A fifth real run" below for why. Pass
`LLMDecider(use_strict_schema=True)` to opt back into strict mode if
you want to test whether Groq's reliability for this model has
improved.

### The rate limit that drives every design choice here

Groq's free tier caps chat-completion models at roughly **30 requests/
minute** (confirmed via Groq's own rate-limit docs and independent
trackers, June 2026), shared at the organization level -- not per agent,
not per API key. `rate_limiter.py` enforces a real wall-clock budget
(default 24 RPM, a safety margin below the published 30) shared across
EVERY `LLMDecider` instance via a class-level limiter, so 16 agents each
holding their own `LLMDecider` correctly draw from one bucket rather than
each believing it has the full 24 RPM to itself. **Confirmed against a
real run**: rate-limit waits of 2.0-2.4s per call appeared in actual
output, matching the expected pacing at ~24-30 RPM.

Practical consequence: `main_llm.py` defaults to only 3 LLM-backed
agents (mixed into a town of 16, the rest on the free `RuleBasedDecider`)
and 30 ticks, not the Phase 1 demo's 16/60. Scale up once you've
confirmed it works -- the rate limiter will simply make a larger run
take longer in wall-clock time, not fail.

### A real bug, found by a real run, and the fix

The first version of this file specified `model = "llama-3.3-70b-
versatile"` and asserted (based on a general feature-announcement
search, not the live API) that it supported Groq's strict
structured-output JSON schema mode. **That was wrong.** A live run
against the real API returned, on every single LLM-backed agent, every
tick:

    BadRequestError("Error code: 400 - {'error': {'message': 'This model
    does not support response format `json_schema`. ...'}}")

The fail-safe design caught this exactly as intended -- every failed
call fell back to `idle`, nothing crashed, the rest of the town (gossip,
trade, governance, the rule-based agents) kept running normally for the
full 30 ticks. But the LLM-backed agents themselves never got to think.

**The fix, in two parts:**
1. Switched `MODEL` to `"openai/gpt-oss-120b"`, which Groq's docs and
   multiple independent integration references confirm DOES support
   `strict: true` json_schema mode.
2. Added a same-instance fallback: if a model rejects json_schema mode
   with this specific error, `LLMDecider` detects it
   (`_is_unsupported_json_schema_error`), switches to the more broadly
   supported `json_object` mode for that instance going forward, and
   retries once rather than idling every subsequent call. This was
   verified with a mocked test reproducing the EXACT error message from
   the real run, confirming the fallback triggers correctly and that
   unrelated errors (rate limits, network failures, other 400s) do NOT
   trigger it. The lesson generalizes: Groq's list of models supporting
   strict structured outputs is small and changes over time -- if you
   swap `MODEL` again, verify against a live call before trusting it,
   the same way this correction was made.

### A second real run, two more findings

A follow-up run (after rotating the API key) revealed two more things,
neither caused by the model fix above:

1. **The run used a stale/invalid key partway through** -- the log
   showed `json_schema` 400s for the first several ticks (meaning an
   old copy of the code, or an old key, was still active), then flipped
   to `AuthenticationError 401 Invalid API Key` from tick 13 onward.
   16+ agent-ticks of rate-limited waiting (2+ seconds each) were spent
   failing on a dead key before the run was stopped manually. **Fix:**
   `main_llm.py` now makes one cheap `client.models.list()` call up
   front, before building the town or starting the simulation, to
   confirm the key actually works. A bad key now fails once, in under a
   second, with a clear message pointing at the most common cause
   (`export` only applies to the current shell session -- a key set in
   a previous terminal window doesn't carry over). Verified with a
   mocked 401 response.
2. **Ctrl+C during a rate-limited wait produced a raw traceback**
   instead of exiting cleanly -- a user stopping a slow, free-tier-
   throttled run partway through is normal and expected, not an error
   condition. **Fix:** the main loop now catches `KeyboardInterrupt`
   and prints partial results (whatever state the LLM-backed agents
   reached) instead of crashing. Verified with a mocked mid-run
   interrupt.

### A third real run, the actual schema bug

With the key working and the pre-flight check passing, a third run hit
a NEW error on every call:

    BadRequestError("Error code: 400 - {'error': {'message': "invalid
    JSON schema for response_format: 'intent': /properties/args:
    `additionalProperties:false` must be set on every object"}})

This was a real bug in `_INTENT_SCHEMA`, not a Groq limitation. Groq's
strict structured-output mode requires `additionalProperties: false`
on EVERY object in the schema, not just the top level -- and the
original schema's `"args": {"type": "object"}` declared no properties
and no constraint at all, which strict mode rejects outright. Worse,
`args` genuinely needed to hold structurally different shapes per
action (a `destination` string for `move`, an item+amount pair in each
direction for `trade_offer`), and trade items use a dynamic key (the
item name -- "food", "money") that strict mode's "every property must
be named in advance" rule cannot represent as a nested dict at all.

**The fix:** the schema was flattened. Every field any action could
need (`destination`, `to`, `give_item`, `give_amount`, `want_item`,
`want_amount`, `rule_type`, `vote_choice`, etc.) is now a top-level,
nullable property, with every property required (satisfying strict
mode's "all fields must be required" constraint) and
`additionalProperties: false` only needing to be declared once, since
there are no longer any nested objects. The model fills in only the
fields relevant to its chosen action and leaves the rest null.
`LLMDecider._build_args` (new) reconstructs the nested dict shape
`actions.py`/`economy.py` already expect from these flat fields, so
**nothing downstream of `Intent` needed to change** -- the
Perception/Intent seam held exactly as designed.

Verified, concretely:
- The schema was checked programmatically: `additionalProperties:
  false` is set, every property is in `required`, and zero nested
  `"type": "object"` schemas remain (the exact category of thing that
  broke last time).
- `_build_args` was tested against a realistic, schema-conformant
  response for all 10 action types, confirming correct reconstruction.
- A full mocked `decide()` call was traced all the way through to a
  REAL `actions.execute()` call against the real engine -- both a
  rejected case (insufficient food) and an accepted case (a genuine
  pending offer appearing in `economy._pending_offers`) were confirmed,
  proving the reconstructed args are truly compatible with the
  validated Phase 1 engine, not just structurally similar to it.
- The json_schema-unsupported-model fallback (from the first finding)
  was re-verified against the new schema shape, confirming it still
  works correctly.

### A fourth real run: an intermittent Groq-side reliability gap

With the schema fixed, a fourth run produced yet another distinct
error on every call:

    BadRequestError("Error code: 400 - {'error': {'message': "Failed
    to validate JSON. Please adjust your prompt. See
    'failed_generation' for more details.", 'code':
    'json_validate_failed', 'failed_generation': ''}})

This one is NOT a bug in this codebase's schema -- it's a documented,
acknowledged reliability gap in Groq's own constrained-decoding
implementation. A search turned up Groq's own community forum
reporting this exact error code on `openai/gpt-oss-120b` and
`gpt-oss-20b` at roughly a 10% failure rate, even though both models
are on Groq's supported-models list for strict structured outputs --
and a separate Groq feature-request thread notes the typical pattern
is the generation getting cut off with just one field missing, which
points at token-budget truncation interacting badly with a verbose,
many-field flat schema.

**Two changes, addressing two different things:**
1. Raised `max_completion_tokens` from 300 to 500 -- cheap, real
   mitigation for the truncation half of the problem, given the
   schema's 16 always-present fields (most `null` for any single
   action) plus a reasoning sentence add real token overhead before
   the model even reaches the part that matters.
2. Added a SECOND, DISTINCT fallback: when this specific error occurs,
   retry just that one call in `json_object` mode, then go straight
   back to trying strict mode on the next call. This is deliberately
   different from the first fallback (model doesn't support
   json_schema at all), which permanently switches an instance to
   json_object mode -- conflating the two would either retry forever
   against a model that flatly can't do strict mode, or permanently
   abandon strict mode after one transient Groq-side flake. Since
   Groq's own numbers put this around 10%, permanently downgrading on
   the first occurrence would throw away strict mode's reliability
   gains for the other ~90% of calls for no reason.

**Honesty about what this can and can't fix:** if Groq's own
infrastructure has a ~10% validation failure rate on this exact
request shape, no amount of correct schema design or retry logic
written here eliminates it -- it can only be absorbed gracefully,
which is what the per-call retry does. If you see this error
occasionally in a real run, that's expected, not a sign something is
still broken; the failed call simply gets one retry in a more
permissive mode, and if that fails too, the agent idles for that tick
and tries again next time.

Verified: the new error detector was tested against the real error
text from the run, confirmed it doesn't false-positive on any of the
three previously-seen error types (unsupported model, rate limit,
auth), and the full retry path was traced end to end with a mock --
including the case where the retry ALSO fails, confirmed to degrade
safely to idle rather than raising.

### A fifth real run, and a correction to the "10%" claim

A second attempt, with the retry logic in place, hit `json_validate_failed`
on **100% of calls again** -- not the ~10% rate the previous fix assumed.
Tellingly, the `[llm retry]` log line never appeared at all, which on
its own is a useful diagnostic: it means the retry path wasn't even
being exercised as designed for this run (most likely an older copy of
the file was still in use locally -- worth double-checking you're
running the freshly extracted zip each time, in the same terminal
session where the key was exported).

Independently of that, a deeper search turned up evidence that changes
the diagnosis: this isn't a rare flake. Groq's own community forum has
a thread titled "Structured Outputs ignored by openai/gpt-oss-120b"
describing the model returning free-form text instead of JSON under
`strict: true`. A model-card discussion on Hugging Face documents
`gpt-oss-*` models returning a BLANK completion specifically when the
model is uncertain about the correct answer -- which directly explains
`'failed_generation': ''` (an empty string can never validate). And a
December 2025 LangChain GitHub issue reports this exact model failing
structured output through LangChain's standard adapter, independent of
this codebase entirely. Taken together, this looks like a broader,
more persistent compatibility gap with this model on Groq's hosting
right now, not the occasional flake the original "~10%" framing
suggested -- and a schema with many always-present, mostly-null fields
(this one's situation) is exactly the kind of ambiguous case the
blank-response behavior seems to trigger most.

**The fix: change the default, not just add another fallback.**
`LLMDecider` now defaults to `json_object` mode from the FIRST call,
not as a fallback after a failed strict-mode attempt. A new
`use_strict_schema=False` parameter exists if you want to deliberately
re-test strict mode later (the retry-on-failure machinery from the
previous fix is still there and still engages correctly if you do).
Two additional changes target the suspected root cause directly:
`reasoning_effort="low"` (gpt-oss models spend tokens on hidden
chain-of-thought before producing JSON, by default at `'medium'`
effort -- competing with the actual answer for a fixed token budget)
and keeping `max_completion_tokens=500` from the previous fix.

Verified: a mocked test confirmed the new default makes exactly ONE
API call (not two) in `json_object` mode with `reasoning_effort='low'`
set correctly -- no wasted strict-mode attempt that the evidence above
suggests would just fail anyway. The full regression suite (Phase 1's
two entry points, byte-identical output) and the complete docstring
audit both still pass.

**Why this is a more honest position than the previous fix:** rather
than adding a fourth patch on top of a strict-mode-by-default design
that the evidence no longer supports, this changes the default itself.
If Groq's structured-output reliability for this model improves later,
flipping back to `use_strict_schema=True` is a one-line change away
from re-testing it.

### A sixth run revealed the real problem: stale local files, not a bug

A run after the json_object-default fix shipped produced THE EXACT SAME
error, with byte-identical dollar amounts down to the cent, on the same
exact tick-by-tick sequence as the previous run -- which is itself the
diagnostic. Real LLM calls against a live model don't reproduce
identical results across separate processes; the rule-based agents'
fixed seed does. That match, plus a `grep -n "use_strict_schema"`
coming back completely empty against the local files, confirmed the
fix had never reached the Python process actually being run -- an old
local copy (likely from an earlier extraction into a `Downloads`
folder) was still in use the whole time.

This explains every confusing signal from runs 4-6 retroactively: the
missing `[llm retry]` log line, the unchanged error text, the identical
money values. None of it was a new code problem -- the content was
correct in every zip shipped from this point forward; it simply wasn't
reaching the terminal it needed to.

That same run DID surface one genuinely new, useful data point, from
Groq's `failed_generation` field actually containing text for once
(usually it's empty): `'max completion tokens reached before
generating a valid document'`. That's direct confirmation of the
truncation hypothesis behind the `max_completion_tokens` increase and
`reasoning_effort='low'` change above -- the model was running out of
its token budget mid-generation, exactly as suspected.

**The fix, this time, targets the actual problem (distribution, not
code):** `main_llm.py` now prints a `BUILD_VERSION` banner as the very
first line of output, before any other check runs, and cross-checks it
against a matching `BUILD_VERSION` on the `LLMDecider` class itself --
catching the specific case where one file is current and the other is
stale. If you ever see a version-mismatch warning, that's the fix
working as intended: re-extract the zip fresh into a clean folder
rather than reusing or partially overwriting an old one.

### What's verified vs. not

What WAS verified directly:
- The rate limiter's timing, against the real system clock (exact
  spacing and burst behavior confirmed), AND against a real run's
  observed wait times (2.0-2.4s, matching expectations).
- `LLMDecider._parse_intent` and the json_schema/json_object fallback
  logic, against valid JSON, invalid JSON, unknown actions, malformed
  `args`, and the exact 400 error text a real run produced -- all
  handled correctly.
- The full `decide()` call path, against a MOCKED Groq client -- request
  shape, prompt content, and response extraction all confirmed correct.
- **The rate limiter, the fail-safe fallback-to-idle path, and the
  rest of the simulation's resilience under real LLM failures** -- all
  confirmed against FIVE actual runs with a real GROQ_API_KEY (a
  `json_schema` 400 in run 1, a `401 Invalid API Key` in run 2, an
  `additionalProperties:false` schema error in run 3, and
  `json_validate_failed` errors in BOTH runs 4 and 5 -- fully resilient
  every time; nothing has ever crashed across any of the five).
- The new `_verify_api_key` pre-flight check and clean Ctrl+C handling,
  against mocked failure scenarios reproducing each real error.
- The redesigned flat schema's structural correctness (no nested
  objects, every property required, additionalProperties:false set --
  the exact conditions strict mode actually checks), and
  `_build_args`'s reconstruction logic against all 10 action types,
  traced all the way through to real `actions.execute()` calls against
  the live (non-mocked) Phase 1 engine.
- The json_object-by-default path: confirmed via mock to make exactly
  ONE API call (not two) with `reasoning_effort='low'` set, with no
  wasted strict-mode attempt.

What still has NOT been verified: a successful, non-idle, in-character
LLM-driven action, chosen by the actual model rather than constructed
by a test. Five real runs so far have each hit a wall before reaching
one -- a model/schema mismatch, an invalid API key, a schema validity
bug, and `json_validate_failed` twice in a row (100% of calls both
times, not the ~10% originally assumed -- see the correction above).
Given that two consecutive runs failed completely under strict mode,
the strategy changed from "patch around an occasional flake" to
"default to the more reliable mode entirely." This should be the
strongest candidate yet, but stated plainly: this codebase cannot
control Groq's hosting reliability for this model, only respond to it
as resiliently as possible. If `json_object` mode also proves
unreliable in your next run, that would be a different and important
finding -- and at that point, switching `MODEL` to a different
Groq-hosted option becomes the more honest next step, not another
prompt tweak.

### Architecture note: this is still the same seam

`LLMDecider` implements the exact same `Decider` protocol
(`decide(agent_id, perception) -> Intent`) as `RuleBasedDecider`.
**Nothing in `world.py`, `agent.py`, `actions.py`, `economy.py`,
`governance.py`, or `engine.py` changed** to add this -- confirmed by
re-running `main.py` and `stress_test.py` after building Phase 2 and
seeing identical behavior. This is the payoff of treating
`decision.py`'s Perception/Intent contract as the one seam from the
start of the project.

As of build `2026-07-05-v7-repeal-spec-sync`, the LLM JSON schema,
system prompt, and `_build_user_prompt` output also cover governance
`repeal` and the Phase 4 `Perception` fields (`active_crises`,
`self_faction`, `speculation_buzz`, `faction_lean`, `enacted_proposals`)
so the LLM-backed path stays aligned with what `RuleBasedDecider`
already reads.

## Phase 3: recording for visualization

```
recorder.py    -- Recorder class: wraps an Engine, captures a trace
record_demo.py -- entry point: builds a town, records it, saves trace.json
```

### What this is and isn't

`recorder.py` produces a single JSON trace file from a run. A
companion browser-based visualizer (built separately, not part of this
Python package) loads that file and renders an interactive replay: a
map of the four locations with agents as colored circles that move
between them tick by tick, a sidebar with live charts (money per
agent, reputation spread over time), a scrubber/play control, and a
click-to-inspect modal showing any agent's traits and current state.

This is **replay**, not **live-driving** -- the visualizer plays back
a trace recorded ahead of time; it does not connect to a running
Python process. That was a deliberate scope choice for Phase 3: this
codebase's engine is already fully deterministic and replayable
(that's what `stress_test.py`'s snapshot pattern proved out), so
recording a trace and replaying it client-side needed zero changes to
the validated core and no new server infrastructure. Phase 5's
`live_server.py` adds the complementary live-driving path (watching a
simulation advance in real time, including real LLM-backed ticks
pausing exactly as they do in the terminal).

### Design: wrap the Engine from the outside

`Recorder` follows the exact same seam discipline as `decision.py`'s
`Decider` protocol and `llm_decider.py`'s `LLMDecider`: it wraps an
`Engine` instance and observes it from the outside, calling
`engine.step()` itself and capturing a snapshot after each call (see
`Recorder.step`). **Nothing in `world.py`, `agent.py`, `actions.py`,
`economy.py`, `governance.py`, or `engine.py` changed** to support
this -- the validated core stays untouched, the same way it stayed
untouched when Phase 2 added LLM-backed agents.

### The trace format

One JSON object, written by `Recorder.save`:

- `"locations"`: a fixed `{name: {x, y}}` map layout, assigned once in
  `recorder.py` (in a 0-1000 by 0-1000 coordinate space the frontend
  can scale freely) -- `world.py`'s `Location` has no spatial concept
  of its own (it's just a name plus a resource pool), so this is
  purely a visualization concern and deliberately lives here, not in
  the simulation core.
- `"agents_static"`: each agent's name, traits, and `decider_kind`
  (e.g. `"RuleBasedDecider"` or `"LLMDecider"`, read via
  `type(agent.decider).__name__` so the recorder never has to import
  `llm_decider.py` -- and therefore never requires the `groq` package
  -- just to know which kind of brain an agent has). Kept separate
  from the per-tick frames so this data isn't repeated once per tick.
- `"frames"`: one entry per recorded tick, each with every agent's
  location/money/inventory/reputation, the town's active rules and
  treasury balance, and that tick's new events.

### A real normalization fix: `agents_involved`

`world.log_event`'s callers (`actions.py`, `economy.py`,
`governance.py`) use inconsistent key names for "which agent did
this" depending on event kind -- `"agent"` for move/work/speak/gossip,
`"by"` for trade_rejected/vote_cast, `"from_"`/`"to"` for
trade_completed, and no agent reference at all for some kinds (e.g.
`trade_failed_insufficient_funds`, `proposal_closed`, which are
genuinely not about one specific agent). That's a reasonable shape for
the simulation's own internal purposes, but it means a frontend asking
"show me everything agent_07 was involved in" would need to
special-case every event kind's key naming.

`Recorder._normalize_event` (called from `step()`, before a frame is
saved) adds a uniform `agents_involved: list[str]` field to a *copy*
of each event, without altering any existing keys -- pulling from
`agent`/`by`/`from_`/`to`/`proposed_by`/`about`/`heard_by` wherever
each happens to appear, filtered to values that actually look like
agent IDs. This keeps the normalization complexity in exactly one
place (the recorder, which only exists to serve the visualizer)
instead of leaking back into `world.py`'s event-logging or being
re-implemented in frontend JavaScript.

### Known limitations (intentional, for now)

- `DEFAULT_LOCATION_LAYOUT` hard-codes coordinates for exactly the
  four locations `town_factory.py` currently creates (`farm`,
  `market`, `town_hall`, `tavern`). `Recorder.__init__` raises clearly
  if `engine.world.locations` contains a name with no matching layout
  entry, rather than silently dropping that location from the map --
  but adding a fifth location to the simulation does require adding
  one entry to this layout, by design, since the recorder has no way
  to invent a sensible position on its own.
- The visualizer's data is embedded directly in its HTML/JS rather
  than fetched from `trace.json` at runtime, since the artifact format
  used to build it can't read arbitrary local files. For a trace
  longer than a few dozen ticks, the embedded payload gets large
  enough that the visualizer should be regenerated with a trimmed
  trace (fewer ticks, or `heard_by` lists stripped from gossip events
  as `agents_involved` already carries that information) rather than
  embedding the full multi-hundred-KB output of a long
  `record_demo.py` run.
- The recording path has been exercised with `curfew`, `wealth_tax`, and
  `repeal` (via `rule_repealed` events in the trace), but nothing about
  the trace format is rule-type-specific -- `active_rules` is recorded
  as whatever dict `world.active_rules` happens to contain at each
  tick, so a future fourth rule type would show up automatically with
  no `recorder.py` changes needed.

## Phase 4: political/economic chaos and cross-sector intersections

```
chaos.py -- corruption, market shocks, bank runs, speculation, unrest,
            faction formation -- and the wiring connecting them to
            each other and to the existing economy/politics/norms systems
```

### What "intersection" actually means here

Before this phase, politics (`governance.py`), economy (`economy.py`),
and norms (reputation, on `Agent`) ran as three independent subsystems
that happened to share a tick clock -- a curfew didn't affect trade, a
trade dispute didn't affect politics, reputation didn't influence
voting. `chaos.py` adds real cross-sector feedback loops, not three
separate chaos generators bolted on side by side:

- **Reputation -> economy**: town-wide average reputation dropping
  below a threshold triggers a `bank_run` crisis; `decision.py`'s
  `RuleBasedDecider._respond_to_trade` reads this flag and becomes
  sharply more reluctant to accept trades while it's active -- norms
  collapsing directly suppresses commerce, not just a cosmetic flag.
- **Politics -> norms -> politics**: a caught embezzler's reputation
  collapses publicly, and every OTHER agent gets a direct memory of
  the specific scandal -- a political fact other agents can react to
  in future votes, not a hidden ledger anomaly.
- **Economy -> politics**: a sustained high Gini coefficient raises
  the probability of a spontaneous `unrest` crisis, which applies a
  mild town-wide reputation drag -- wealth concentration becomes a
  political fact with a real consequence, not just a number in a
  stress-test report.
- **Politics -> politics**: repeated voting alignment between two
  agents merges them into a shared faction (`world.factions`); a
  faction's recent vote tendency (`faction_lean`) then correlates its
  members' FUTURE votes -- the difference between independent
  per-proposal coin flips and an actual emergent voting bloc.
- **Rule repeal**: reuses the EXACT SAME propose/vote/enact pipeline as
  new rules (`governance.py`'s `propose`/`cast_vote`/`_finalize`) with
  a new `"repeal"` rule_type that removes a previously-enacted rule's
  `active_rules` keys, tracked via a `proposal_id -> keys` index so
  repeal doesn't need to know any rule's internal shape.

### A real bug, found by the stress test: corruption miscalibration

The first version of `chaos.py`'s corruption mechanism (each of 16
agents independently rolling a per-tick probability built from
treasury size plus small town_hall/official bonuses) produced **191
corruption scandals over 1000 ticks** -- roughly one every 5 ticks,
not the rare, occasional event the design called for. The root cause:
a per-agent probability that feels small in isolation (1-2%) becomes
a near-certainty in aggregate once rolled by 16 agents every tick with
no cooldown -- the same category of miscalibration as Phase 1's
`WANDERLUST_CHANCE` tuning, just not caught until this feature's own
stress test.

The consequence was severe and traceable end to end: each scandal's
0.35 reputation penalty arrived faster than `REPUTATION_DECAY_RATE`
could repair it, so town-wide average reputation entered a permanent
one-way decline (0.5 down to 0.148 over 1000 ticks, confirmed via a
`reputation_mean` trajectory added to `stress_test.py` specifically to
diagnose this). That decline crossed the `bank_run` trigger threshold
at tick ~161 and the crisis **never ended** for the rest of the run --
the same shape of bug as the original Phase 1 reputation ratchet
(hard caps that never recovered), just manifesting at the
population-average level instead of per-agent.

**The fix**: corruption's base rates were reduced roughly 5-8x, and a
`CORRUPTION_COOLDOWN_TICKS` floor (25 ticks) was added so a scandal
genuinely cannot recur for a fixed window afterward, regardless of how
the per-tick rate is tuned -- rate alone is fragile to population-size
changes (more agents rolling the same rate produces more aggregate
events), a cooldown is not. Re-running the stress test after the fix:
**20 scandals over 1000 ticks** (roughly one every 50 ticks), and 3 of
4 triggered crises (`unrest` x3, `bank_run` x1) actually resolved
within the run. A follow-up 1500-tick run confirmed two distinct
bank-run cycles, each crashing reputation hard (to 0.075 and 0.022
respectively) and recovering on its own over roughly 250-300 ticks,
with stable healthy periods between them -- a real boom-bust political
cycle, which is what "occasionally produces genuine collapse states"
(the explicit severity target for this feature) actually looks like,
as opposed to either a permanent flatline or background noise.

### Other state added

- `Agent.official_track_record`: count of an agent's own proposals
  (including repeals) that have passed -- read by corruption's
  official-status bonus, incremented by `governance.py`'s `_finalize`.
- `World.factions`: `agent_id -> faction_id`, built from repeated
  voting alignment (see `chaos.update_factions`), not assigned upfront.
- `World.active_crises`: a set of short string tags (`"bank_run"`,
  `"unrest"`) rather than fixed boolean fields, for the same reason
  `active_rules` is a flexible dict -- the set of possible crisis types
  is expected to grow.
- `Perception` gained `active_crises`, `self_faction`,
  `speculation_buzz`, `faction_lean`, and `enacted_proposals` --
  the last one deliberately exposes only `{proposal_id, rule_type}`
  per enacted rule, not `governance.py`'s internal raw-keys index,
  so an agent can target a specific repeal without Perception leaking
  an implementation detail of how rules are stored.

### Known limitations (intentional, for now)

- Faction IDs are just whichever agent_id became the faction's
  identity first (not a separately generated name) -- functional, not
  evocative. A real "faction naming" feature is a natural follow-up,
  not attempted here.
- Speculation buzz only distorts the RECEIVING side's price judgment
  in one direction (rumor makes a price look worse, modeling suspicion)
  -- it doesn't yet model buzz making a price look like a bargain
  (FOMO-driven speculation), which would need a signed buzz value
  rather than the current always-nonnegative one.
- Corruption, market shocks, and unrest are not yet wired to influence
  EACH OTHER directly (e.g. a market shock making corruption more or
  less likely) -- the intersections built so far are the ones
  explicitly scoped (reputation<->economy, wealth<->politics,
  politics<->politics), not an exhaustive cross-product of every
  mechanism against every other one.

## Phase 5: true live visualization

```
live_server.py -- local web server (stdlib only): runs the simulation
                  in a background thread, streams one frame per tick
                  to any connected browser over Server-Sent Events
```

### Run it

    python3 live_server.py            # fully rule-based, fast
    python3 live_server.py --llm      # mixes in 3 LLM-backed agents (needs GROQ_API_KEY)

Then open http://localhost:8765/ in a browser. Multiple tabs can watch
the SAME running town simultaneously -- there is one simulation per
server process, not one per browser connection.

### Why this is a different problem from Phase 3's replay

Phase 3's visualizer loads a finished JSON trace and plays it back --
there is no live Python process behind it while you're looking at it.
"Visual simulation when running" means watching the engine advance
WHILE it's actually executing, which needs something to bridge a live
Python process to a browser. `live_server.py` does that with a local
HTTP server and Server-Sent Events (SSE): the simulation runs in a
background thread, and `Recorder.step()` (see recorder.py) is called
in a loop, broadcasting each frame it returns to every connected
browser as it's produced.

SSE rather than WebSockets, deliberately: this is a strictly
one-directional feed (engine -> browser; the browser never needs to
send anything back to the simulation), and SSE is plain HTTP with no
extra protocol or library needed -- every modern browser's
`EventSource` API speaks it natively. WebSockets would be the right
tool if a future version let the browser send commands back (pause,
speed up); for the current one-way requirement, SSE is the simpler
tool that's a genuinely correct fit, not a corner cut.

Same seam discipline as every other phase: `live_server.py` wraps an
`Engine`/`Recorder` pair from the OUTSIDE. Nothing in `world.py`,
`agent.py`, `actions.py`, `economy.py`, `governance.py`, `engine.py`,
`chaos.py`, or `decision.py` changed to support this -- the simulation
has no idea whether it's being watched live, run headless, or recorded
to a file.

### `--llm` mode and visualizing "thinking"

`--llm` mirrors `main_llm.py`'s pattern exactly: the first 3 agents get
`LLMDecider`, the rest stay rule-based. When an LLM-backed tick is slow
(the real Groq rate-limiting this project has run into many times),
the frontend needs to show that as "the town is thinking," not look
frozen or broken. This is done by timing each tick's wall-clock
duration in `live_server.py` itself (not by hooking into
`llm_decider.py`'s internals, which print to stdout rather than emit
any structured signal this server could capture) -- a tick over 1.5
seconds is treated as evidence of a real LLM call (possibly
rate-limited) and surfaced in the status bar.

### A real bug, found while building this: chaos.py wasn't actually seeded

While testing `live_server.py`, two separate `main.py` runs with the
same `SEED = 7` were diffed against each other as a sanity check --
and they were NOT identical. The cause: `chaos.py`'s randomized hooks
(`apply_market_shocks_if_triggered`, `apply_corruption_if_opportunity`,
`update_unrest_state`) called Python's GLOBAL `random` module directly,
rather than an injected, seeded `random.Random` instance -- breaking
the reproducibility guarantee every other randomized component in this
codebase honors (`RuleBasedDecider` takes an injected `rng`; `main.py`
explicitly documents `SEED = 7` as making runs "fully reproducible").
This bug existed before `live_server.py` was built -- it would have
silently affected `stress_test.py` and `record_demo.py` runs too -- it
simply hadn't been caught yet because no prior session had diffed two
same-seed runs of the SAME entry point against each other.

**The fix**: `Engine.__init__` gained an `rng: random.Random | None`
parameter; all three of `chaos.py`'s randomized functions now take an
injected `rng` argument instead of calling the global module; every
entry point (`main.py`, `stress_test.py`, `record_demo.py`,
`main_llm.py`, `live_server.py`) now constructs its `Engine` with
`Engine(world, agents, rng=rng)`, passing the SAME seeded instance
already used for `build_agents`. Verified directly: two independent
`python3 main.py` runs now produce byte-identical output, including
every chaos event (corruption scandals, market shocks) -- confirmed
via a real diff, not just code review.

### Other things verified directly while building this

- The SSE stream was tested with real `curl -N` connections, not just
  read for correctness: confirmed `tick_started`/`frame` events arrive
  in the right order with valid, correctly-shaped JSON matching
  `Recorder`'s frame format exactly.
- Two simultaneous connections were tested concurrently and both
  received frames from the same shared simulation (5 and 4 frames
  respectively, overlapping correctly given their different connect
  times) -- confirming multiple browser tabs genuinely watch one
  simulation, not each spinning up their own.
- Clean shutdown was tested by sending SIGTERM with an SSE connection
  still open and blocked in `queue.get()`; the server exited in under
  a second. `ThreadingHTTPServer.daemon_threads` is set explicitly
  (`True`) rather than relied on as an implicit default, since this
  project targets Python 3.10+ and that default isn't a documented
  guarantee across the whole range.

### Known limitations (intentional, for now)

- One-directional only: the browser can watch but not send commands
  back (pause, change speed, swap which agents are LLM-backed) --
  SSE was the right tool for the CURRENT requirement; a future
  bidirectional version would need WebSockets instead.
- No persistence: closing the server loses the running town. Phase 3's
  `record_demo.py` remains the tool for "I want to keep this run and
  look at it later"; Phase 5 is for "I want to watch it happen."
- Only one simulation per server process -- there's no multi-town or
  multi-tenant support; running two towns at once means running two
  separate `live_server.py` processes on different ports.




