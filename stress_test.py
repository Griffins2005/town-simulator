"""
stress_test.py -- Long-run stress harness, free (no LLM calls).

Purpose: run the engine for many more ticks than main.py's short demo,
tracking time-series metrics, to answer the question a 60-tick demo
can't: does this settle into a healthy, varied equilibrium, or does it
degenerate (one agent owns everything, the town piles into one room,
reputation flatlines, a rule passes once and is never revisited)?

Why this matters before adding an LLM: every one of these failure modes
is cheap to find and fix here, for free, at 1000 ticks. The same
investigation against a rate-limited free-tier LLM costs real wall-clock
time and burns your quota -- you want to walk in already knowing the
engine's baseline behavior, so any weirdness once Phase 2 lands is
clearly attributable to the LLM layer, not a latent Phase-1 bug.

Run: python3 stress_test.py
"""

from __future__ import annotations

import math
import random
import statistics
from collections import Counter

from engine import Engine
from town_factory import build_agents, build_world

import chaos
import economy
import governance

NUM_AGENTS = 16
NUM_TICKS = 1000
SEED = 7
SAMPLE_EVERY = 25  # record a metrics snapshot every N ticks, not every
# tick -- 1000 individual snapshots would be noisy to read and mostly
# redundant; sampling at this interval still catches slow drift while
# keeping the output digestible.


def gini(values: list) -> float:
    """Gini coefficient of `values`, in [0, 1]. 0 = perfectly equal,
    1 = one holder has everything. Standard discrete formula applied to
    sorted values. Used here specifically to answer "is the economy
    concentrating wealth in a runaway way" -- raw min/max money figures
    can't show this as cleanly, since two agents at the extremes could
    be true outliers in an otherwise healthy spread, or could be a real
    concentration trend. Gini summarizes the whole distribution in one
    number, which is what a long unattended run needs for a quick read.
    """
    n = len(values)
    if n == 0 or sum(values) == 0:
        return 0.0
    sorted_vals = sorted(values)
    cumulative = sum((i + 1) * v for i, v in enumerate(sorted_vals))
    return (2 * cumulative) / (n * sum(sorted_vals)) - (n + 1) / n


def location_entropy(agents: dict, locations: list) -> float:
    """Shannon entropy (base-2, normalized to [0,1]) of the agent
    population's location distribution. 0 = everyone in one place (the
    exact failure mode from the original "town collapses into one room"
    bug found during Phase 1). 1 = perfectly even spread across all
    locations. Normalizing by max possible entropy (log2(num_locations))
    makes this comparable across town configurations with different
    location counts, not just this specific 4-location town.
    """
    counts = {loc: 0 for loc in locations}
    for a in agents.values():
        counts[a.location] = counts.get(a.location, 0) + 1
    n = len(agents)
    if n == 0:
        return 0.0
    entropy = 0.0
    for c in counts.values():
        if c > 0:
            p = c / n
            entropy -= p * math.log2(p)
    max_entropy = math.log2(len(locations)) if len(locations) > 1 else 1.0
    return entropy / max_entropy if max_entropy > 0 else 0.0


def snapshot(tick: int, agents: dict, world) -> dict:
    """Compute a single point-in-time metrics snapshot of the town's
    state: wealth distribution (Gini, min, max), reputation spread
    (min, max, stdev, mean), spatial distribution (location entropy),
    active governance rules, active crises, faction count, treasury
    balance, and cumulative event count.

    The reputation MEAN (not just min/max/stdev) and active_crises were
    added specifically to diagnose chaos.py's bank-run trigger: a bank
    run fires when town-wide AVERAGE reputation drops below
    chaos.BANK_RUN_REPUTATION_TRIGGER, so seeing the mean's trajectory
    alongside corruption_scandal counts is what distinguishes "the bank
    run is a real, slow trust-collapse signal" from "the bank run is
    just downstream of corruption's reputation penalty being miscalibrated."

    Returns a flat dict suitable for printing as one table row and for
    later first-vs-last trend comparisons (see `main`'s "trend reads").
    """
    moneys = [a.money for a in agents.values()]
    reputations = [a.reputation for a in agents.values()]
    return {
        "tick": tick,
        "money_gini": round(gini(moneys), 3),
        "money_min": round(min(moneys), 2),
        "money_max": round(max(moneys), 2),
        "reputation_min": round(min(reputations), 3),
        "reputation_max": round(max(reputations), 3),
        "reputation_mean": round(statistics.mean(reputations), 3),
        "reputation_stdev": round(statistics.pstdev(reputations), 3),
        "location_entropy": round(location_entropy(agents, list(world.locations.keys())), 3),
        "active_rules": dict(world.active_rules),
        "active_crises": sorted(world.active_crises),
        "faction_count": len(set(world.factions.values())),
        "treasury": round(world.treasury, 2),
        "total_events": len(world.event_log),
    }


def main() -> None:
    """Build a fresh town, run it for NUM_TICKS ticks (default 1000),
    taking a metrics snapshot every SAMPLE_EVERY ticks. Prints a
    time-series table, a first-vs-last trend summary flagging
    concentration/divergence/collapse, and full-run event-kind totals.

    This is the harness that found every bug documented in README.md's
    "Stress-test findings" section -- re-run it after any change to
    decision.py, economy.py, or governance.py to check for regressions.
    """
    rng = random.Random(SEED)
    economy.reset_offers()
    governance.reset()
    chaos.reset_buzz()
    chaos.reset_factions()
    chaos.reset_corruption_cooldown()

    world = build_world()
    agents = build_agents(rng, NUM_AGENTS)
    engine = Engine(world, agents, rng=rng)

    print(f"Stress test: {NUM_AGENTS} agents, {NUM_TICKS} ticks, seed={SEED}\n")
    print(f"{'tick':>5} {'gini':>6} {'$min':>6} {'$max':>6} {'rep_min':>8} "
          f"{'rep_max':>8} {'rep_mean':>9} {'rep_sd':>7} {'loc_ent':>8} "
          f"{'fact':>5} crises  rules")
    print("-" * 110)

    history = []
    for t in range(NUM_TICKS):
        engine.step()
        if t % SAMPLE_EVERY == 0 or t == NUM_TICKS - 1:
            s = snapshot(t, agents, world)
            history.append(s)
            print(f"{s['tick']:>5} {s['money_gini']:>6} {s['money_min']:>6} "
                  f"{s['money_max']:>6} {s['reputation_min']:>8} "
                  f"{s['reputation_max']:>8} {s['reputation_mean']:>9} "
                  f"{s['reputation_stdev']:>7} {s['location_entropy']:>8} "
                  f"{s['faction_count']:>5} {s['active_crises']}  {s['active_rules']}")

    print("\n--- trend reads (first snapshot vs last) ---")
    first, last = history[0], history[-1]
    print(f"  money_gini:        {first['money_gini']} -> {last['money_gini']}  "
          f"{'(concentrating)' if last['money_gini'] > first['money_gini'] + 0.1 else '(stable)'}")
    print(f"  reputation_stdev:  {first['reputation_stdev']} -> {last['reputation_stdev']}  "
          f"{'(diverging)' if last['reputation_stdev'] > first['reputation_stdev'] + 0.1 else '(stable)'}")
    print(f"  location_entropy:  {first['location_entropy']} -> {last['location_entropy']}  "
          f"{'(collapsing toward one place!)' if last['location_entropy'] < 0.5 else '(spread maintained)'}")
    print(f"  active_rules at end: {last['active_rules']}")

    kind_counts = Counter(e["kind"] for e in world.event_log)
    print("\n--- full-run event totals ---")
    for kind, count in kind_counts.most_common():
        print(f"  {kind}: {count}")

    print("\n--- chaos diagnostics ---")
    corruption_events = [e for e in world.event_log if e["kind"] == "corruption_scandal"]
    crisis_starts = [e for e in world.event_log if e["kind"] == "crisis_started"]
    crisis_ends = [e for e in world.event_log if e["kind"] == "crisis_ended"]
    repeal_events = [e for e in world.event_log if e["kind"] == "rule_repealed"]
    print(f"  corruption scandals: {len(corruption_events)} "
          f"(total skimmed: {round(sum(e['skimmed'] for e in corruption_events), 2)})")
    print(f"  crises started: {len(crisis_starts)}, ended: {len(crisis_ends)}")
    for e in crisis_starts:
        print(f"    tick {e['tick']}: {e['crisis']} started "
              f"({ {k: v for k, v in e.items() if k not in ('tick', 'kind', 'crisis')} })")
    print(f"  rule repeals: {len(repeal_events)}")
    print(f"  final faction count: {last['faction_count']}")
    # The actual diagnostic this section exists for: does reputation_mean
    # decline gradually (a real trust-collapse signal feeding the bank
    # run) or does it drop in sharp steps that line up with
    # corruption_scandal ticks (meaning the bank run is mostly a
    # corruption-calibration artifact, not independent reputation decay)?
    print(f"  reputation_mean trajectory: {[s['reputation_mean'] for s in history]}")


if __name__ == "__main__":
    main()
