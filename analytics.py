"""
analytics.py -- Read-only forensic metrics for a running town.

Inspired by how the July 2026 Hugging Face agent-intrusion reconstruction
worked (timeline clustering, phase activity, anomaly triage over a huge
action log) -- applied here to *town* events, not computer systems.

This module never mutates World/Agent state. Recorder and live_server
call it after a tick to explain what just happened: wealth shape, who
is talking to whom, whether chatter looks like a coordinated campaign,
and which town domains (social / political / economic) a tick crossed.
"""

from __future__ import annotations

import math
from collections import Counter

from agent import Agent
from world import World

DOMAIN_SOCIAL = ("speak", "gossip", "influence_campaign", "campaign_ended",
                 "call_for_expulsion", "notoriety", "worship", "converted", "worship_session",
                 "bankrupt", "going_bankrupt", "recovered")
DOMAIN_POLITICAL = (
    "vote_cast", "rule_proposed", "rule_repealed", "proposal_closed",
    "faction_joined", "faction_formed", "lobby_succeeded", "lobby_failed",
    "member_expelled", "member_arrived", "member_welcomed", "member_restored", "vote_suspended",
    "vote_rights_restored", "proposal_deadlocked", "festival_ended", "faith_leader",
    "leader_seated", "leader_elected", "leader_impeached", "leader_stepped_down",
)
DOMAIN_ECONOMIC = (
    "trade_completed", "trade_rejected", "trade_failed_insufficient_funds",
    "work", "corruption_scandal", "market_shock", "crisis_started", "crisis_ended",
    "invention", "invention_adopted", "bankrupt", "going_bankrupt", "recovered",
)


def gini(values: list[float]) -> float:
    """Gini coefficient in [0, 1]. Same discrete formula as stress_test.py."""
    n = len(values)
    total = sum(values)
    if n == 0 or total == 0:
        return 0.0
    sorted_vals = sorted(values)
    cumulative = sum((i + 1) * v for i, v in enumerate(sorted_vals))
    return (2 * cumulative) / (n * total) - (n + 1) / n


def location_entropy(agents: dict[str, Agent], locations: list[str]) -> float:
    """Normalized Shannon entropy of where people are standing, in [0, 1]."""
    if not agents or not locations:
        return 0.0
    counts = {loc: 0 for loc in locations}
    for agent in agents.values():
        counts[agent.location] = counts.get(agent.location, 0) + 1
    n = len(agents)
    entropy = 0.0
    for count in counts.values():
        if count > 0:
            p = count / n
            entropy -= p * math.log2(p)
    max_entropy = math.log2(len(locations))
    return entropy / max_entropy if max_entropy else 0.0


def lorenz_points(moneys: list[float]) -> dict[str, list[float]]:
    """Cumulative population share vs cumulative wealth share (Lorenz curve)."""
    ordered = sorted(max(0.0, m) for m in moneys)
    n = len(ordered)
    total = sum(ordered) or 1.0
    xs = [0.0]
    ys = [0.0]
    running = 0.0
    for i, value in enumerate(ordered, 1):
        running += value
        xs.append(round(i / n, 4))
        ys.append(round(running / total, 4))
    return {"x": xs, "y": ys}


def relationship_edges(agents: dict[str, Agent], limit: int = 40) -> list[list]:
    """Top relationship edges by |weight|, as [from_id, to_id, weight]."""
    edges: list[list] = []
    for agent in agents.values():
        for other_id, weight in agent.relationships.items():
            if abs(weight) >= 0.05:
                edges.append([agent.agent_id, other_id, round(weight, 2)])
    edges.sort(key=lambda e: abs(e[2]), reverse=True)
    return edges[:limit]


def domain_counts(events: list[dict]) -> dict[str, int]:
    """How many events this tick landed in each town domain."""
    counts = {"social": 0, "political": 0, "economic": 0, "other": 0}
    for event in events:
        kind = event.get("kind", "")
        if kind in DOMAIN_SOCIAL:
            counts["social"] += 1
        elif kind in DOMAIN_POLITICAL:
            counts["political"] += 1
        elif kind in DOMAIN_ECONOMIC:
            counts["economic"] += 1
        else:
            counts["other"] += 1
    return counts


def crossings(domains: dict[str, int]) -> list[str]:
    """Domain pairs that both fired this tick -- the town analog of a
    trust-boundary crossing in the HF reconstruction (social chatter
    becoming a vote bloc, a scandal becoming a market panic).
    """
    active = [name for name, count in domains.items() if name != "other" and count > 0]
    out = []
    for i, a in enumerate(active):
        for b in active[i + 1:]:
            out.append(f"{a}->{b}")
    return out


def build_public_headlines(world: World, agents: dict[str, Agent], limit: int = 6) -> list[dict]:
    """Short, named headlines an agent can gossip about -- real events,
    not 'Did you hear about agent_07?'.
    """
    names = {aid: a.persona.name for aid, a in agents.items()}
    headlines: list[dict] = []
    for event in reversed(world.event_log):
        kind = event.get("kind")
        item: dict | None = None
        if kind == "corruption_scandal":
            who = event.get("agent")
            item = {
                "about": who,
                "text": f"{names.get(who, who)} was caught embezzling {event.get('skimmed')}",
            }
        elif kind == "crisis_started":
            item = {"about": None, "text": f"a {event.get('crisis')} has begun"}
        elif kind == "crisis_ended":
            item = {"about": None, "text": f"the {event.get('crisis')} has ended"}
        elif kind == "faction_formed":
            item = {"about": None, "text": f"the {event.get('name')} has formed"}
        elif kind == "influence_campaign":
            who = event.get("target")
            item = {
                "about": who,
                "text": event.get("name") or f"whispers are spreading about {names.get(who, who)}",
            }
        elif kind == "rule_proposed":
            who = event.get("by")
            item = {
                "about": who,
                "text": f"{names.get(who, who)} proposed a {event.get('rule_type')}",
            }
        elif kind == "proposal_closed" and event.get("passed"):
            item = {"about": None, "text": f"proposal {event.get('proposal_id')} passed"}
        elif kind == "invention":
            who = event.get("agent")
            item = {
                "about": who,
                "text": f"{names.get(who, who)} invented {event.get('name')}",
            }
        elif kind == "invention_adopted":
            who = event.get("agent")
            item = {
                "about": who,
                "text": f"{names.get(who, who)} adopted {event.get('name')}",
            }
        elif kind == "market_shock":
            item = {
                "about": None,
                "text": f"{event.get('shock_kind')} at the {event.get('location')}",
            }
        elif kind == "member_expelled":
            who = event.get("agent")
            item = {
                "about": who,
                "text": f"{names.get(who, who)} was expelled from the roll",
            }
        elif kind == "member_arrived":
            who = event.get("agent")
            item = {
                "about": who,
                "text": f"{event.get('name') or names.get(who, who)} just arrived and wants a seat",
            }
        elif kind == "member_welcomed":
            who = event.get("agent")
            item = {
                "about": who,
                "text": f"the town welcomed {event.get('name') or names.get(who, who)}",
            }
        elif kind == "vote_suspended":
            who = event.get("agent")
            item = {
                "about": who,
                "text": f"{names.get(who, who)} had their vote suspended",
            }
        elif kind == "lobby_succeeded":
            who = event.get("to")
            item = {
                "about": who,
                "text": f"{names.get(event.get('agent'), event.get('agent'))} lobbied {names.get(who, who)} to vote {event.get('lean')}",
            }
        elif kind == "converted":
            who = event.get("agent")
            item = {
                "about": who,
                "text": f"{event.get('name') or names.get(who, who)} joined the {event.get('to')}",
            }
        elif kind == "worship_session":
            item = {
                "about": None,
                "text": f"{event.get('name') or event.get('faith')} is in session at the {event.get('location')}",
            }
        elif kind == "festival_ended":
            item = {"about": None, "text": f"the {event.get('faith')} festival has ended"}
        elif kind == "bankrupt":
            who = event.get("agent")
            item = {
                "about": who,
                "text": f"{event.get('name') or names.get(who, who)} went bankrupt ({event.get('livelihood')})",
            }
        elif kind == "going_bankrupt":
            who = event.get("agent")
            item = {
                "about": who,
                "text": f"{event.get('name') or names.get(who, who)} is going bankrupt",
            }
        elif kind == "recovered":
            who = event.get("agent")
            item = {
                "about": who,
                "text": f"{event.get('name') or names.get(who, who)} recovered from {event.get('from_')}",
            }
        elif kind == "faith_leader":
            who = event.get("agent")
            item = {
                "about": who,
                "text": f"{event.get('agent_name') or names.get(who, who)} now leads the {event.get('name') or event.get('faith')}",
            }
        elif kind == "leader_seated":
            who = event.get("agent")
            item = {
                "about": who,
                "text": f"{event.get('name') or names.get(who, who)} is seated as town leader ({event.get('how')})",
            }
        elif kind == "leader_elected":
            who = event.get("agent")
            item = {
                "about": who,
                "text": f"{event.get('name') or names.get(who, who)} was elected town leader",
            }
        elif kind == "leader_impeached":
            who = event.get("agent")
            item = {
                "about": who,
                "text": f"{event.get('name') or names.get(who, who)} was impeached",
            }
        elif kind == "leader_stepped_down":
            who = event.get("agent")
            item = {
                "about": who,
                "text": f"{event.get('name') or names.get(who, who)} stepped down ({event.get('reason')})",
            }
        elif kind == "proposal_deadlocked":
            item = {
                "about": None,
                "text": f"proposal {event.get('proposal_id')} deadlocked — the leader is working the room",
            }
        elif kind == "notoriety":
            who = event.get("agent")
            item = {
                "about": who,
                "text": f"{event.get('name') or names.get(who, who)} is becoming notorious",
            }
        elif kind == "call_for_expulsion":
            who = event.get("about")
            item = {
                "about": who,
                "text": f"someone is calling for {names.get(who, who)} to be expelled",
            }
        if item:
            headlines.append(item)
        if len(headlines) >= limit:
            break
    return headlines


def compute_metrics(world: World, agents: dict[str, Agent], events: list[dict]) -> dict:
    """One-tick snapshot of the numbers the dashboard charts consume."""
    moneys = [a.money for a in agents.values()]
    reps = [a.reputation for a in agents.values()]
    kinds = Counter(e.get("kind") for e in events)
    locations = list(world.locations.keys())
    location_counts = {loc: 0 for loc in locations}
    for agent in agents.values():
        location_counts[agent.location] = location_counts.get(agent.location, 0) + 1
    domains = domain_counts(events)
    return {
        "gini": round(gini(moneys), 3),
        "location_entropy": round(location_entropy(agents, locations), 3),
        "mean_reputation": round(sum(reps) / len(reps), 3) if reps else 0.5,
        "treasury": round(world.treasury, 2),
        "faction_count": len(set(world.factions.values())),
        "gossip_volume": kinds.get("gossip", 0),
        "speak_volume": kinds.get("speak", 0),
        "trade_completed": kinds.get("trade_completed", 0),
        "vote_volume": kinds.get("vote_cast", 0),
        "event_mix": dict(kinds),
        "location_counts": location_counts,
        "lorenz": lorenz_points(moneys),
        "domains": domains,
        "crossings": crossings(domains),
    }


def score_anomalies(metrics: dict, window: list[dict], crises: list[str]) -> tuple[float, list[str]]:
    """Lightweight z-ish flags over a short rolling window.

    This is the town analog of the incident's anomaly triage: not a
    classifier, just 'this tick is louder / more concentrated than the
    recent baseline.' Score is clamped to [0, 1].
    """
    flags: list[str] = []
    score = 0.0
    if crises:
        flags.append("crisis:" + ",".join(crises))
        score += 0.35
    gossip = metrics.get("gossip_volume", 0)
    if gossip >= 6:
        flags.append("gossip_storm")
        score += 0.25
    if window:
        gossips = [m.get("gossip_volume", 0) for m in window]
        mean_g = sum(gossips) / len(gossips)
        if gossip >= mean_g + 2 and gossip >= 4:
            flags.append("chatter_spike")
            score += 0.2
        gini_now = metrics.get("gini", 0.0)
        gini_then = window[0].get("gini", gini_now)
        if gini_now - gini_then >= 0.08:
            flags.append("wealth_concentration")
            score += 0.25
        trades = [m.get("trade_completed", 0) for m in window]
        mean_t = sum(trades) / len(trades)
        if mean_t >= 1 and metrics.get("trade_completed", 0) == 0:
            flags.append("trade_freeze")
            score += 0.15
    if metrics.get("crossings"):
        flags.append("domain_crossing")
        score += 0.1
    return round(min(1.0, score), 3), flags
