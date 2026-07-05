"""
main_llm.py -- Phase 2 entry point: run the town with LLM-backed agents.

Usage:
    pip install groq
    export GROQ_API_KEY=your_key_here
    python3 main_llm.py

Deliberately runs FEWER ticks and FEWER LLM-backed agents than the Phase 1
demo. Reasoning, concretely: at a 24 RPM safety-margin rate limit (see
rate_limiter.RECOMMENDED_RPM_SAFETY_MARGIN), even with sparse-thinking
interrupts cutting call volume substantially, a 16-agent, 60-tick run
could plausibly need on the order of a hundred-plus LLM calls -- at 24/min
that's several real minutes of wall-clock waiting, for a first run whose
main purpose is "does this work at all." Start small, watch it think,
then scale NUM_AGENTS/NUM_TICKS up once you've confirmed the mechanics
hold with real model reasoning instead of rule-based heuristics.

NOT all agents need to be LLM-backed at once -- this script mixes a
SMALL number of LLM-backed agents with the rest on the free, instant
RuleBasedDecider. This is a deliberate middle ground: you get to observe
real LLM reasoning in a populated, living town (the rule-based agents
still trade, vote, gossip, circulate) without paying the full rate-limit
cost of every single agent thinking via the network every tick.
"""

from __future__ import annotations

import os
import random

from engine import Engine
from town_factory import build_agents, build_world

import chaos
import economy
import governance

# Bumped every time llm_decider.py's request/fallback logic changes
# meaningfully. Printed at startup (see main(), below) specifically so
# file staleness is self-evident from a single glance at the output --
# this exists because TWO real runs in a row turned out to be against
# an old local copy of this codebase, discovered only after several
# rounds of "did the fix even run" debugging. A version banner makes
# that check immediate instead of requiring a manual grep.
BUILD_VERSION = "2026-07-05-v7-repeal-spec-sync"

NUM_AGENTS = 16
NUM_LLM_AGENTS = 3   # how many of the 16 get an LLMDecider; rest stay rule-based.
# Kept small and explicit -- see module docstring for the rate-limit math.
NUM_TICKS = 30
SEED = 7


def _verify_api_key(LLMDecider) -> bool:
    """Make ONE lightweight call to Groq before starting the simulation,
    to confirm the API key is actually valid. Added after a real run
    burned through 5+ ticks (10+ seconds of rate-limited waiting) hitting
    a 401 Invalid API Key on every single call, because nothing checked
    the key was live before committing to a full run. Failing once,
    fast, with a clear message is far better than failing silently and
    repeatedly while the rate limiter politely paces out each failure.

    Returns True if the key works, False otherwise (with an explanatory
    message already printed).
    """
    import os
    from groq import Groq
    try:
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        # The cheapest possible real call: list models. Doesn't spend
        # any of the rate-limited chat-completion budget, and fails the
        # same way (401) on a bad key as a real chat call would.
        client.models.list()
        return True
    except Exception as exc:
        print(f"GROQ_API_KEY appears invalid or Groq is unreachable: {exc!r}")
        print("Double-check the key at https://console.groq.com/keys and that")
        print("it's exported in THIS shell session (export only applies to the")
        print("current shell and its children, not retroactively).")
        return False


def main() -> None:
    """Build a town of NUM_AGENTS, swap NUM_LLM_AGENTS of them onto
    LLMDecider (the rest stay on the free RuleBasedDecider), run for
    NUM_TICKS, and print notable events plus final state for the
    LLM-backed agents specifically.

    Exits early with a clear message (no traceback) if the `groq`
    package isn't installed, GROQ_API_KEY isn't set, or the key fails
    a quick validity check -- see module docstring for setup
    instructions. Also handles Ctrl+C cleanly: a long rate-limited run
    is a normal thing to want to stop early, not a crash, so this
    prints partial results and exits 0 rather than raising
    KeyboardInterrupt up through the rate limiter's sleep call.
    """
    print(f"townsim build: {BUILD_VERSION}")
    print("(If you've made code changes and don't see the version you expect")
    print(" here, you're running a stale copy -- re-extract the zip fresh.)\n")

    try:
        from llm_decider import LLMDecider
    except ImportError:
        print("The 'groq' package isn't installed. Run: pip install groq")
        return

    if LLMDecider.BUILD_VERSION != BUILD_VERSION:
        # This specific check exists because TWO real runs in a row
        # turned out to be against a stale local copy where main_llm.py
        # and llm_decider.py had drifted apart in time -- this catches
        # that exact scenario the moment it happens, loudly, rather than
        # producing confusing behavior that looks like a NEW bug.
        print(f"WARNING: main_llm.py version ({BUILD_VERSION}) doesn't match "
              f"llm_decider.py version ({LLMDecider.BUILD_VERSION}).")
        print("This usually means you have a mix of old and new files. "
              "Re-extract the whole zip fresh into a clean folder.")
        return

    if not os.environ.get("GROQ_API_KEY"):
        print("No GROQ_API_KEY environment variable set.")
        print("Get a free key at https://console.groq.com/keys and:")
        print("  export GROQ_API_KEY=your_key_here")
        return

    print("Verifying GROQ_API_KEY...")
    if not _verify_api_key(LLMDecider):
        return
    print("API key OK.\n")

    rng = random.Random(SEED)
    economy.reset_offers()
    governance.reset()
    chaos.reset_buzz()
    chaos.reset_factions()
    chaos.reset_corruption_cooldown()

    world = build_world()
    agents = build_agents(rng, NUM_AGENTS)

    # Swap in LLMDecider for the first NUM_LLM_AGENTS agents only. Each
    # LLMDecider instance shares the same class-level rate limiter (see
    # llm_decider.LLMDecider._shared_limiter), so this doesn't multiply
    # the effective rate limit by NUM_LLM_AGENTS -- they all draw from
    # the same bucket, correctly mirroring Groq's per-organization cap.
    llm_agent_ids = list(agents.keys())[:NUM_LLM_AGENTS]
    for agent_id in llm_agent_ids:
        agents[agent_id].decider = LLMDecider(verbose=True)

    print(f"Town of {NUM_AGENTS} agents ({NUM_LLM_AGENTS} LLM-backed via Groq, "
          f"rest rule-based), {NUM_TICKS} ticks, seed={SEED}")
    print(f"LLM-backed agents: {llm_agent_ids}\n")
    print("Note: this will pause noticeably between LLM-backed ticks due "
          "to free-tier rate limiting -- that's the rate limiter working "
          "as intended, not a hang. Press Ctrl+C at any time to stop "
          "early and see partial results.\n")

    engine = Engine(world, agents, rng=rng)

    try:
        for t in range(NUM_TICKS):
            before = len(world.event_log)
            engine.step()
            new_events = world.event_log[before:]
            interesting = [e for e in new_events if e["kind"] not in ("move", "work", "idle")]
            if interesting:
                print(f"--- tick {t} ---")
                for e in interesting:
                    detail = {k: v for k, v in e.items() if k not in ("tick", "kind")}
                    print(f"  [{e['kind']}] {detail}")
    except KeyboardInterrupt:
        print("\n\nStopped early by user (Ctrl+C). Showing state as of the last completed tick:")

    print("\n--- final state (LLM-backed agents) ---")
    for agent_id in llm_agent_ids:
        a = agents[agent_id]
        print(f"  {agent_id} ({a.persona.name}): money={a.money:.2f} "
              f"inventory={a.inventory} reputation={a.reputation:.2f} "
              f"location={a.location}")


if __name__ == "__main__":
    main()
