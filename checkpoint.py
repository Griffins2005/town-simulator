"""
checkpoint.py -- Honest experiments: serialize, restore, fork, compare.

The live town is not rewound. A checkpoint is a full engine snapshot
(world, agents, rng, and the module ledgers that live beside World).
A fork rebuilds a second Engine from that snapshot, optionally injects
one crisis, and runs it for N ticks. The live module state is put back
before the function returns, so the watching town keeps its own clock.

Compare is control vs treatment. Default: both towns stay rule-based
and treatment gets one inject. brains="ab" is the same snapshot with
all-rule vs the first three agents on Groq. Persist writes a pickle
under saves/ so the town can come back after the process dies.

The Decider still only proposes; the engine still validates.
"""

from __future__ import annotations

import copy
import json
import os
import pickle
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import chaos
import economy
import faith
import governance
import inventions
from agent import Agent, Persona
from decision import RuleBasedDecider
from engine import Engine
from memory import MemoryEntry, MemoryLog
from recorder import Recorder
from world import Location, World

FORK_TICKS_DEFAULT = 12
FORK_TICKS_MAX = 24
FORK_LLM_TICKS_MAX = 8
FORK_LLM_AGENTS = 3
INJECT_KINDS = ("flood", "famine", "unrest", "bank_run", "market_shock", "drought", "pollution")
SAVE_DIR = Path(__file__).resolve().parent / "saves"
LATEST_SAVE = SAVE_DIR / "latest.pkl"
INDEX_PATH = SAVE_DIR / "index.json"


def capture_modules() -> dict:
    return {
        "governance": governance.export_state(),
        "economy": economy.export_state(),
        "chaos": chaos.export_state(),
        "faith": faith.export_state(),
        "inventions": inventions.export_state(),
    }


def apply_modules(blob: dict) -> None:
    governance.import_state(blob.get("governance") or {})
    economy.import_state(blob.get("economy") or {})
    chaos.import_state(blob.get("chaos") or {})
    faith.import_state(blob.get("faith") or {})
    inventions.import_state(blob.get("inventions") or {})


def capture(engine: Engine) -> dict:
    """Full snapshot. Stays in-process; rng state is not JSON."""
    return {
        "tick": engine.world.tick,
        "modules": capture_modules(),
        "town": _capture_town(engine),
    }


def _capture_town(engine: Engine) -> dict:
    world = engine.world
    return {
        "rng": engine.rng.getstate(),
        "world": {
            "tick": world.tick,
            "locations": {
                name: dict(loc.resources) for name, loc in world.locations.items()
            },
            "active_rules": dict(world.active_rules),
            "treasury": world.treasury,
            "event_log": list(world.event_log[-300:]),
            "factions": dict(world.factions),
            "faction_names": dict(world.faction_names),
            "active_crises": sorted(world.active_crises),
            "crisis_intensity": dict(world.crisis_intensity),
            "crisis_hold": dict(world.crisis_hold),
            "notice_board": list(world.notice_board),
            "active_campaigns": list(world.active_campaigns),
            "inventions": list(world.inventions),
            "town_leader_id": world.town_leader_id,
            "office_ever_vacated": world.office_ever_vacated,
            "leader_seated_tick": getattr(world, "leader_seated_tick", 0),
        },
        "agents": [_agent_to_dict(a) for a in engine.agents.values()],
    }


def _agent_to_dict(agent: Agent) -> dict:
    persona = agent.persona
    return {
        "agent_id": agent.agent_id,
        "location": agent.location,
        "money": agent.money,
        "inventory": dict(agent.inventory),
        "relationships": dict(agent.relationships),
        "reputation": agent.reputation,
        "official_track_record": agent.official_track_record,
        "pastoral_track_record": getattr(agent, "pastoral_track_record", 0),
        "voting_rights": agent.voting_rights,
        "vote_suspended_until": agent.vote_suspended_until,
        "expelled": agent.expelled,
        "solvency": getattr(agent, "solvency", "ok"),
        "persona": {
            "name": persona.name,
            "industriousness": persona.industriousness,
            "generosity": persona.generosity,
            "sociability": persona.sociability,
            "rule_respect": persona.rule_respect,
            "risk_tolerance": persona.risk_tolerance,
            "wanderlust": getattr(persona, "wanderlust", 0.35),
            "faith": getattr(persona, "faith", "unaffiliated"),
            "piety": getattr(persona, "piety", 0.2),
            "livelihood": getattr(persona, "livelihood", "laborer"),
        },
        "memory": [
            {
                "tick": m.tick, "kind": m.kind, "subject": m.subject,
                "data": dict(m.data), "salience": m.salience,
            }
            for m in agent.memory.recent(40)
        ],
    }


def _world_from_dict(row: dict) -> World:
    world = World(locations=[
        Location(name, resources=dict(resources))
        for name, resources in (row.get("locations") or {}).items()
    ])
    world.tick = int(row.get("tick") or 0)
    world.active_rules = dict(row.get("active_rules") or {})
    world.treasury = float(row.get("treasury") or 0)
    world.event_log = [dict(e) for e in (row.get("event_log") or [])]
    world.factions = dict(row.get("factions") or {})
    world.faction_names = dict(row.get("faction_names") or {})
    world.active_crises = set(row.get("active_crises") or [])
    world.crisis_intensity = dict(row.get("crisis_intensity") or {})
    world.crisis_hold = dict(row.get("crisis_hold") or {})
    world.notice_board = list(row.get("notice_board") or [])
    world.active_campaigns = list(row.get("active_campaigns") or [])
    world.inventions = list(row.get("inventions") or [])
    world.town_leader_id = row.get("town_leader_id")
    world.office_ever_vacated = bool(row.get("office_ever_vacated"))
    world.leader_seated_tick = int(row.get("leader_seated_tick") or 0)
    return world


def _agent_from_dict(row: dict, rng: random.Random) -> Agent:
    p = row.get("persona") or {}
    log = MemoryLog()
    for item in row.get("memory") or []:
        log.add(MemoryEntry(
            int(item.get("tick") or 0),
            item.get("kind") or "memory",
            item.get("subject"),
            dict(item.get("data") or {}),
            float(item.get("salience") or 0.5),
        ))
    return Agent(
        agent_id=row["agent_id"],
        persona=Persona(
            name=p.get("name") or row["agent_id"],
            industriousness=float(p.get("industriousness") or 0.5),
            generosity=float(p.get("generosity") or 0.5),
            sociability=float(p.get("sociability") or 0.5),
            rule_respect=float(p.get("rule_respect") or 0.5),
            risk_tolerance=float(p.get("risk_tolerance") or 0.5),
            wanderlust=float(p.get("wanderlust") or 0.35),
            faith=p.get("faith") or "unaffiliated",
            piety=float(p.get("piety") or 0.2),
            livelihood=p.get("livelihood") or "laborer",
        ),
        location=row.get("location") or "homes",
        money=float(row.get("money") or 0),
        inventory=dict(row.get("inventory") or {}),
        relationships=dict(row.get("relationships") or {}),
        reputation=float(row.get("reputation") or 0.5),
        official_track_record=int(row.get("official_track_record") or 0),
        pastoral_track_record=int(row.get("pastoral_track_record") or 0),
        voting_rights=bool(row.get("voting_rights", True)),
        vote_suspended_until=int(row.get("vote_suspended_until") or 0),
        expelled=bool(row.get("expelled")),
        solvency=row.get("solvency") or "ok",
        memory=log,
        decider=RuleBasedDecider(rng),
    )


def rebuild(snapshot: dict) -> Engine:
    apply_modules(copy.deepcopy(snapshot.get("modules") or {}))
    town = copy.deepcopy(snapshot.get("town") or {})
    rng = random.Random()
    rng.setstate(town["rng"])
    world = _world_from_dict(town.get("world") or {})
    agents = {
        row["agent_id"]: _agent_from_dict(row, rng)
        for row in (town.get("agents") or [])
    }
    engine = Engine(world, agents, rng=rng)
    engine._last_thought_tick = {aid: -1 for aid in agents}
    engine._cached_intent = {}
    engine._current_buzz = {}
    return engine


def attach_llm(engine: Engine, n: int = FORK_LLM_AGENTS, model: str | None = None) -> list[str]:
    """Seat Groq on the first n agents. Rule-based stays on everyone else."""
    from llm_decider import LLMDecider, require_groq

    require_groq()
    ids = sorted(engine.agents)[: max(0, int(n))]
    for agent_id in ids:
        engine.agents[agent_id].decider = LLMDecider(verbose=False, model=model)
    return ids


def _llm_ready() -> tuple[bool, str]:
    if not os.environ.get("GROQ_API_KEY"):
        return False, "set GROQ_API_KEY to run the 3-LLM fork"
    try:
        from llm_decider import require_groq
        require_groq()
    except Exception as exc:
        return False, str(exc)
    return True, ""


def _run_branch(
    snapshot: dict,
    ticks: int,
    inject: str | None,
    intensity,
    brains: str = "rule",
    model: str | None = None,
) -> tuple[dict, list]:
    engine = rebuild(snapshot)
    llm_ids: list[str] = []
    if brains == "llm":
        llm_ids = attach_llm(engine, model=model)
    if inject in INJECT_KINDS:
        chaos.inject_crisis(engine.world, engine.agents, inject, intensity, engine.rng)
    rec = Recorder(engine)
    frames = []
    for _ in range(max(0, ticks)):
        frames.append(rec.step())
    last = frames[-1] if frames else rec.peek()
    last = dict(last)
    last["_brains"] = brains
    last["_llm_ids"] = llm_ids
    return last, frames


def experiment(
    snapshot: dict,
    ticks: int = FORK_TICKS_DEFAULT,
    inject: str | None = None,
    intensity: str | float = "serious",
    brains: str = "rule",
    model: str | None = None,
) -> dict:
    """Control vs treatment. Live module state is restored on the way out.

    brains="rule" (default): both towns stay rule-based; treatment gets the inject.
    brains="ab": same snapshot, all-rule vs first three on Groq; the same
    inject (or none) is applied to both so the brains are the variable.
    """
    brains = (brains or "rule").strip().lower()
    if brains in ("llm", "llm_ab", "compare_brains"):
        brains = "ab"
    if inject in ("", "none", "None"):
        inject = None
    cap = FORK_LLM_TICKS_MAX if brains == "ab" else FORK_TICKS_MAX
    ticks = max(1, min(cap, int(ticks or FORK_TICKS_DEFAULT)))
    if brains == "ab":
        ok, why = _llm_ready()
        if not ok:
            return {"error": why}

    live = capture_modules()
    try:
        if brains == "ab":
            control, control_frames = _run_branch(
                snapshot, ticks, inject, intensity, brains="rule", model=model
            )
            treatment, treat_frames = _run_branch(
                snapshot, ticks, inject, intensity, brains="llm", model=model
            )
        else:
            control, control_frames = _run_branch(
                snapshot, ticks, None, intensity, brains="rule", model=model
            )
            treatment, treat_frames = _run_branch(
                snapshot, ticks, inject, intensity, brains="rule", model=model
            )
    finally:
        apply_modules(live)
    start_tick = snapshot.get("tick", 0)
    return {
        "from_tick": start_tick,
        "to_tick": start_tick + ticks,
        "ticks": ticks,
        "inject": inject,
        "intensity": intensity if inject else None,
        "brains": brains,
        "control_brains": "rule",
        "treatment_brains": "llm" if brains == "ab" else "rule",
        "control": control,
        "treatment": treatment,
        "diff": diff_frames(control, treatment, control_frames, treat_frames),
    }


def persist(snapshot: dict, label: str | None = None) -> dict:
    """Write the snapshot to saves/latest.pkl and a dated archive file."""
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    payload = pickle.dumps(snapshot, protocol=4)
    LATEST_SAVE.write_bytes(payload)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    slug = "".join(ch for ch in (label or "town") if ch.isalnum() or ch in "-_")[:24] or "town"
    name = f"{slug}_{stamp}.pkl"
    path = SAVE_DIR / name
    path.write_bytes(payload)
    item = {
        "id": stamp,
        "tick": snapshot.get("tick"),
        "path": name,
        "saved_at": stamp,
        "label": label or "town",
    }
    index = [row for row in list_saves() if row.get("path") != name]
    index.append(item)
    INDEX_PATH.write_text(json.dumps(index[-16:], indent=2))
    return item


def list_saves() -> list[dict]:
    if not INDEX_PATH.exists():
        return []
    try:
        rows = json.loads(INDEX_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    return [row for row in rows if isinstance(row, dict) and (SAVE_DIR / str(row.get("path") or "")).exists()]


def load_save(name: str | None = None) -> dict | None:
    """Load latest.pkl, or a named archive under saves/."""
    path = LATEST_SAVE if not name or name in ("latest", "latest.pkl") else SAVE_DIR / Path(name).name
    if not path.exists() or path.parent != SAVE_DIR:
        return None
    try:
        blob = pickle.loads(path.read_bytes())
    except Exception:
        return None
    return blob if isinstance(blob, dict) else None


def _leader_name(frame: dict) -> str:
    lid = frame.get("town_leader_id")
    if not lid:
        return "vacant"
    return ((frame.get("agents") or {}).get(lid) or {}).get("name") or lid


def _count_solvency(frame: dict, tag: str) -> int:
    return sum(1 for st in (frame.get("agents") or {}).values() if st.get("solvency") == tag)


def _tally(frames: list) -> dict:
    c: Counter = Counter()
    for frame in frames:
        for event in frame.get("events") or []:
            kind = event.get("kind")
            if kind:
                c[kind] += 1
    return dict(c)


def diff_frames(control: dict, treatment: dict, control_frames=None, treat_frames=None) -> dict:
    """What changed because of the inject — not two marks on one tape."""
    occ = lambda f: (f.get("metrics") or {}).get("location_counts") or {}
    expelled = lambda f: sum(1 for st in (f.get("agents") or {}).values() if st.get("expelled"))
    gini = lambda f: round(float((f.get("metrics") or {}).get("gini") or 0), 3)
    return {
        "gini": [gini(control), gini(treatment)],
        "treasury": [control.get("treasury"), treatment.get("treasury")],
        "leader": [_leader_name(control), _leader_name(treatment)],
        "crises": [sorted(control.get("active_crises") or []),
                   sorted(treatment.get("active_crises") or [])],
        "bankrupt": [_count_solvency(control, "bankrupt"), _count_solvency(treatment, "bankrupt")],
        "expelled": [expelled(control), expelled(treatment)],
        "occupancy": [occ(control), occ(treatment)],
        "laws": [len(control.get("enacted_proposals") or []),
                 len(treatment.get("enacted_proposals") or [])],
        "events": {
            "control": _tally(control_frames or []),
            "treatment": _tally(treat_frames or []),
        },
    }
