"""
networks.py -- Graph theory on the living town.

The street graph is geography. This module is the social graph: a tie
exists when two residents like each other enough to be a channel.
Metrics are computed, never invented — degree, clustering, betweenness,
and brokerage (neighbors in more than one faction). Perception reads
them; agents do not get to name a centrality they do not have.
"""

from __future__ import annotations

from collections import deque

from agent import Agent
from world import World

TIE = 0.18


def adjacency(agents: dict[str, Agent]) -> dict[str, set[str]]:
    """Undirected ties: either side's relationship is at least TIE."""
    adj = {aid: set() for aid in agents}
    for agent in agents.values():
        for other_id, weight in agent.relationships.items():
            if other_id not in agents or other_id == agent.agent_id:
                continue
            if weight >= TIE or agents[other_id].relationships.get(agent.agent_id, 0) >= TIE:
                adj[agent.agent_id].add(other_id)
                adj[other_id].add(agent.agent_id)
    return adj


def _components(adj: dict[str, set[str]]) -> int:
    seen: set[str] = set()
    n = 0
    for start in adj:
        if start in seen:
            continue
        n += 1
        q = deque([start])
        seen.add(start)
        while q:
            node = q.popleft()
            for nxt in adj[node]:
                if nxt not in seen:
                    seen.add(nxt)
                    q.append(nxt)
    return n


def _clustering(adj: dict[str, set[str]], node: str) -> float:
    nbrs = list(adj.get(node) or [])
    if len(nbrs) < 2:
        return 0.0
    closed = 0
    for i, a in enumerate(nbrs):
        for b in nbrs[i + 1:]:
            if b in adj.get(a, ()):
                closed += 1
    possible = len(nbrs) * (len(nbrs) - 1) / 2
    return round(closed / possible, 3) if possible else 0.0


def _betweenness(adj: dict[str, set[str]]) -> dict[str, float]:
    """Brandes betweenness, normalized, undirected. n is 16."""
    nodes = list(adj)
    score = {n: 0.0 for n in nodes}
    for source in nodes:
        stack: list[str] = []
        pred = {n: [] for n in nodes}
        sigma = {n: 0.0 for n in nodes}
        dist = {n: -1 for n in nodes}
        sigma[source] = 1.0
        dist[source] = 0
        queue = deque([source])
        while queue:
            v = queue.popleft()
            stack.append(v)
            for w in adj[v]:
                if dist[w] < 0:
                    dist[w] = dist[v] + 1
                    queue.append(w)
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    pred[w].append(v)
        delta = {n: 0.0 for n in nodes}
        while stack:
            w = stack.pop()
            for v in pred[w]:
                if sigma[w]:
                    delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
            if w != source:
                score[w] += delta[w]
    size = len(nodes)
    norm = 2.0 / ((size - 1) * (size - 2)) if size > 2 else 1.0
    return {k: round(v * norm / 2.0, 4) for k, v in score.items()}


def snapshot(agents: dict[str, Agent], world: World, trade_pairs: list | None = None) -> dict:
    adj = adjacency(agents)
    bet = _betweenness(adj)
    n = max(1, len(adj))
    e = sum(len(v) for v in adj.values()) / 2
    density = round(2 * e / (n * (n - 1)), 3) if n > 1 else 0.0
    nodes = {}
    brokers = []
    for agent_id, agent in agents.items():
        nbrs = adj.get(agent_id) or set()
        factions = {
            world.factions.get(oid)
            for oid in nbrs
            if world.factions.get(oid)
        }
        broker = len(factions) >= 2
        row = {
            "degree": len(nbrs),
            "clustering": _clustering(adj, agent_id),
            "betweenness": bet.get(agent_id, 0.0),
            "broker": broker,
        }
        nodes[agent_id] = row
        if broker:
            brokers.append({
                "id": agent_id,
                "name": agent.persona.name,
                "betweenness": row["betweenness"],
            })
    brokers.sort(key=lambda r: r["betweenness"], reverse=True)
    return {
        "density": density,
        "ties": int(e),
        "components": _components(adj),
        "nodes": nodes,
        "brokers": brokers[:5],
        "trade_pairs": list(trade_pairs or []),
    }


def of(snapshot_row: dict | None, agent_id: str) -> dict:
    nodes = (snapshot_row or {}).get("nodes") or {}
    return dict(nodes.get(agent_id) or {
        "degree": 0, "clustering": 0.0, "betweenness": 0.0, "broker": False,
    })
