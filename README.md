# Eidolon Society Lab

Repository: https://github.com/Griffins2005/town-simulator

A turn-based town where named residents trade, gossip, vote, embezzle,
panic, form factions, and invent tools. Intelligence proposes; the
engine validates. A live Society Lab UI lets you watch the town, read
each agent's six-stage decision trace, and open a separate analytics
page for inequality, coordination, and domain-crossing charts.

```
World conditions + laws
        ↓
Agent perception + memory + relationships
        ↓
Independent reasoning and choice
        ↓
Validated action and consequences
        ↓
Economic, social, and political change
        ↓
New memories, institutions, laws, and inventions
```

**Rules** define what is permitted and what follows.
**Intelligence** decides what an agent attempts.
**Memory** is what the agent believes it remembers.
**Relationships** shape trust and coordination.
**Politics** lets agents change the rules.
**Creativity** lets agents propose inventions; a catalog defines the
effect space the world can actually apply.

## Requirements

- Python 3.9 or newer
- A browser, for the live lab
- `groq` only if you run LLM-backed agents (see `requirements.txt`)

Core runs (`main.py`, `stress_test.py`, `record_demo.py`,
`live_server.py` without `--llm`) use the standard library only.

## Run it

    python3 main.py                 # 60-tick terminal demo
    python3 stress_test.py          # 1000-tick harness + time series
    python3 record_demo.py          # 200 ticks → trace.json
    python3 live_server.py          # Society Lab at http://localhost:8765/
    python3 live_server.py --llm    # same UI, 3 agents on Groq
    python3 live_server.py --resume # load saves/latest.pkl if present
    python3 live_server.py --llm --model llama-3.1-8b-instant

Then open:

- [http://localhost:8765/](http://localhost:8765/) — Observe (living town)
- [http://localhost:8765/analytics](http://localhost:8765/analytics) — charts

All five entry points use `SEED = 7`. Same seed, same rule-based
trace, including chaos events. Multiple browser tabs watch the same
running town; they do not each start a new simulation.

### LLM-backed agents

    python3 -m pip install -r requirements.txt
    export GROQ_API_KEY=your_key_here    # console.groq.com/keys
    python3 main_llm.py                  # 3 LLM agents, 30 ticks
    python3 live_server.py --llm

Use `python3 -m pip`, not bare `pip`. On this machine `pip` often
installs into Anaconda while `python3` is a different interpreter, so
`--llm` then fails with `No module named 'groq'`. If you created
`.venv`, run the venv's Python instead:

    .venv/bin/python -m pip install -r requirements.txt
    .venv/bin/python live_server.py --llm

`export` applies only to the current shell. `main_llm.py` checks the
key with a real Groq call before the town starts. Default mix: 3 of 16
agents use `LLMDecider`; the rest stay rule-based. Groq's free tier is
about 30 RPM; `rate_limiter.py` caps the shared bucket at 24 RPM.

`LLMDecider` defaults to Groq `json_object` mode (`openai/gpt-oss-120b`,
`reasoning_effort="low"`). A cheaper Groq model is optional:

    export TOWNSIM_LLM_MODEL=llama-3.1-8b-instant
    python3 live_server.py --llm --model llama-3.1-8b-instant

Pass `use_strict_schema=True` only if you
are re-testing Groq's strict JSON-schema path. Startup prints
`townsim build: 2026-07-05-v7-repeal-spec-sync` and refuses to run if
`main_llm.py` and `llm_decider.py` versions disagree.

## Publish and deploy

There is no hosted Society Lab. Shipping a change is: commit, push the
repo, then run `live_server.py` on the machine that should show the
town. `live_server.py` binds to `localhost:8765`, so the UI is not
reachable from the public internet.

Push the current branch (after you commit):

    git push origin main

On any machine that should run the lab:

    git clone https://github.com/Griffins2005/town-simulator.git
    cd town-simulator
    python3 live_server.py

Then open http://localhost:8765/ and http://localhost:8765/analytics.

LLM mode on that machine:

    python3 -m pip install -r requirements.txt
    export GROQ_API_KEY=your_key_here
    python3 live_server.py --llm

Do not commit `.env`, `GROQ_API_KEY`, or any file that holds secrets.

Two towns means two processes. If 8765 is taken, change `PORT` in
`live_server.py`. Persist writes `saves/`; `--resume` loads the latest.
`python3 record_demo.py` still writes a kept `trace.json`.

## File inventory

### Engine (stdlib)

| File | Role |
| --- | --- |
| `world.py` | Locations, clock, rules, treasury, crises, factions, notice board, inventions |
| `agent.py` | Persona, inventory, relationships, reputation |
| `memory.py` | Append-only structured memory log |
| `decision.py` | `Perception` → `Intent` / `DecisionDraft`; `RuleBasedDecider` |
| `decision_record.py` | Six-stage forensic record for one agent-tick |
| `actions.py` | Only path that mutates agent/world state |
| `economy.py` | Work, trade, demurrage, treasury, farm regen |
| `governance.py` | Propose → vote → enact / repeal (`curfew`, `wealth_tax`, `repeal`) |
| `inventions.py` | Hybrid invent / adopt; catalog of allowed effects |
| `chaos.py` | Corruption, shocks, bank runs, unrest, factions, whisper campaigns |
| `analytics.py` | Read-only metrics: Gini, Lorenz, entropy, domains, anomalies |
| `engine.py` | `step()`: perceive, deliberate, validate, record, housekeeping |
| `town_factory.py` | Shared town: 16 named residents, five locations |
| `recorder.py` | Wraps `Engine`; per-tick JSON frames |
| `checkpoint.py` | Serialize, restore, fork, persist `saves/` |

### Entry points

| File | Role |
| --- | --- |
| `main.py` | 60-tick terminal demo |
| `stress_test.py` | 1000-tick metrics harness |
| `record_demo.py` | Writes `trace.json` |
| `main_llm.py` | Mixed LLM / rule-based terminal run |
| `live_server.py` | Live SSE lab + `POST /control` |

### LLM (needs `groq`)

| File | Role |
| --- | --- |
| `rate_limiter.py` | Shared token-bucket RPM limiter |
| `llm_decider.py` | Same Decider protocol, Groq-backed |

### Society Lab UI (served by `live_server.py`)

| File | Role |
| --- | --- |
| `live_ui.html` | Observe page |
| `live_analytics.html` | Analytics page |
| `live_shared.css` / `live_shared.js` | Shared chrome and stream helpers |
| `static/` | Town backdrop, brand marks, favicon |

Locations the simulation uses: `farm`, `workshop`, `market`,
`town_hall`, `tavern`, `chapel`, `park`, `clinic`, `bank`, `homes`.
Agents walk one open street per tick; a flood can close the bridge
and greenways.

## Architecture

```
world / agent / memory     state
decision / llm_decider     intelligence proposes a DecisionDraft
actions / economy /        engine validates; only these write state
  governance / inventions
decision_record            perceived → remembered → considered
                           → chose → validated → consequences
chaos                      cross-sector crises and unofficial coordination
analytics / recorder       observe after the tick; never mutate
live_server                stdlib HTTP + SSE around Recorder
```

The invariant that matters: `agent.money`, inventories, location,
`world.active_rules`, and invention capabilities change only inside
`actions.py` (or the modules it calls). A hallucinated LLM intent
cannot mint gold by writing a successful validation.

`Recorder` and `live_server.py` wrap `Engine` from the outside. The
core does not know whether a tick is printed, saved to `trace.json`,
or streamed to a browser.

## Decision record

Every agent-tick is assembled into:

```json
{
  "perceived": {},
  "retrieved_memories": [],
  "considered_actions": [],
  "chosen_intent": {},
  "validation": {},
  "state_changes": [],
  "consequences": []
}
```

A Decider fills the first four stages. `actions.py` fills validation,
state changes, and consequences. Click a resident on Observe to read
the live trace.

## Inventions

Hybrid loop, not scripted theater:

1. Agent notices a catalog problem (`farm_scarcity`, `trade_friction`,
   `corruption`).
2. Decider proposes `invent` with a catalog kind (or an LLM label
   mapped onto one).
3. `inventions.py` checks location, resources, traits, and relevant
   memories.
4. Construction consumes food and money and registers a world effect.
5. Other agents may `adopt_invention`.

Current catalog:

| Kind | Effect |
| --- | --- |
| `water_pump` | Farm food regenerates faster |
| `price_board` | Unfair trade offers are easier to refuse |
| `public_ledger` | Longer cooldown between corruption scandals |

An unrecognized kind fails validation. It does not spawn an undefined
effect.

## Live Society Lab

`live_server.py` runs the town on a background thread and streams
`tick_started` / `frame` events over Server-Sent Events.

**Observe** (`/`): river-valley town, named residents, faith and
livelihood on the map, event feed, open proposal (yes / no / quorum),
forensics inspector, diamond timeline, pause, speed, inject flood /
famine / unrest / drought / pollution / bank run. Experiments save a
real engine snapshot, fork a second town, and compare control vs
treatment. Relationships in the rail toggles a social-graph overlay.
People opens a resident roster.

**Analytics** (`/analytics`): Lorenz curve, money vs reputation, event
mix, Gini / entropy / anomaly, social graph, occupancy heatmap, domain
crossings, money by agent, faction membership.

`POST /control` accepts `pause`, `resume`, `speed`, `inject`
(`flood`, `famine`, `unrest`, `bank_run`, `drought`, `pollution`,
`headline`), `checkpoint`, `persist`, `restore`, `load`, and `fork`
(optional `brains=ab` for same-seed all-rule vs 3-LLM).

Brand assets live in `static/` (`eidolon-appicon.png`,
`eidolon-wordmark-ui.png`, favicon). The tab icon is `/favicon.ico`.

## What a run actually does

- **Economy** — work at the farm, two-step trade, demurrage into the
  treasury, periodic redistribution. Total money (agents + treasury)
  is conserved.
- **Governance** — `curfew`, `wealth_tax`, and `repeal` share one
  propose / vote / enact pipeline. Curfew can block movement; a tax
  collects and redistributes; repeal removes enacted keys.
- **Norms** — reputation moves with trade and violations, then decays
  toward 0.5.
- **Social** — speak, gossip, notice-board slips, whisper campaigns
  when gossip concentrates on one target.
- **Politics** — factions form from repeated voting alignment and get
  generated names. Faction lean correlates later votes.
- **Chaos** — corruption scandals (with cooldown, lengthened by a
  public ledger), market shocks, bank runs when mean reputation is
  low, unrest when Gini stays high.
- **Domains** — analytics tags events as social / political /
  economic and flags ticks that cross more than one.

## Stress-test findings (still true)

`stress_test.py` (1000 ticks, seed 7) is how degenerate equilibria
were found. Re-run it after changing `decision.py`, `economy.py`,
`governance.py`, `chaos.py`, or `inventions.py`.

1. Farm food hit 0 and stayed there — `economy.regenerate_resources`
   now grows toward a cap (inventions can raise that rate).
2. Reputation pinned at 0 or 1 — each tick decays toward 0.5.
3. Price-blind trade favored stingy sellers — accept/reject now
   compares offer price to a fair reference. Long-run Gini can still
   rise from compounding variance; demurrage and a voted wealth tax
   are the counterweights.
4. Market became an absorbing room — persona `wanderlust` in
   `RuleBasedDecider` gives circulation an independent, trait-weighted
   chance.
5. Early demurrage destroyed money — it now goes to `world.treasury`
   and is redistributed. System money stays constant.
6. First corruption rates fired ~191 scandals / 1000 ticks — rates
   dropped and a cooldown was added. Expect occasional boom-bust
   bank-run cycles, not a permanent flatline.

## Known limitations

- Rule-based pricing and voting are trait-weighted heuristics. That
  is the layer LLM agents are meant to replace.
- A second vote from the same agent is refused. Lobby is the path
  that can still flip a neighbor.
- Invention kinds are a closed catalog. Agents choose among them;
  they do not author new physics.
- Experiments save a real engine snapshot, fork a second town, and
  compare control vs treatment. The live engine is not rewound by a
  fork. Restore / load put a snapshot back on the live town on purpose.
- Closing `live_server.py` without persist drops the in-memory town.
  Persist writes `saves/`; `--resume` loads `saves/latest.pkl`.
  `record_demo.py` still writes a `trace.json`.
- One simulation per server process. Two towns means two processes
  on different ports.
- Analytics charts need a network path to cdnjs for Chart.js.
- Agents walk one street per tick. A flood closes the bridge and,
  at higher intensity, the greenway and country road.

## Trace format (`record_demo.py`)

`Recorder.save` writes one JSON object:

- `locations` — `{name: {x, y}}` layout for the visualizer
- `agents_static` — name, traits, `decider_kind`
- `frames` — per-tick agent state, rules, treasury, events, metrics,
  decision records, inventions, campaigns, open proposals

Events keep their original keys and gain `agents_involved` so a
frontend does not have to special-case `agent` / `by` / `from_` / `to`.
)
