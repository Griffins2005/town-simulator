"""
live_server.py -- True live visualization: a local web server that runs
the simulation in a background thread and pushes one frame per tick to
any connected browser over Server-Sent Events (SSE).

Design principle, same as every other phase: this wraps an Engine (via
Recorder, see recorder.py) from the OUTSIDE. Nothing in world.py,
agent.py, actions.py, economy.py, governance.py, engine.py, chaos.py, or
decision.py changes to support this -- the simulation has no idea it's
being watched live versus run headless versus recorded to a file. The
ONLY new capability here is "broadcast each frame Recorder.step() returns
to whatever browsers are currently connected," using nothing but the
Python standard library (http.server + threading), matching this
project's existing policy of zero new dependencies for anything that
isn't the LLM path itself.

Why Server-Sent Events, not WebSockets: this is a strictly one-directional
feed (engine -> browser; the browser never needs to push anything back to
the simulation), and SSE is plain HTTP -- no extra protocol, no extra
library, supported natively by every modern browser's EventSource API.
WebSockets would be the right call if the browser needed to send commands
back (e.g. "pause," "speed up") in a future version; for now, one-way
push is the actual requirement, so SSE is the simpler tool that's a
genuinely correct fit, not a corner cut.

Usage:
    python3 live_server.py              # rule-based agents (default)
    python3 live_server.py --llm        # mix in LLM-backed agents (needs GROQ_API_KEY)
    Then open http://localhost:8765/ in a browser.
"""

from __future__ import annotations

import json
import queue
import random
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from engine import Engine
from recorder import Recorder
from town_factory import build_agents, build_world

import chaos
import economy
import governance

HOST = "localhost"
PORT = 8765

RULE_BASED_TICK_DELAY_SECONDS = 0.6

NUM_AGENTS = 16
NUM_LLM_AGENTS = 3
SEED = 7


class SimulationBroadcaster:
    """Owns the Engine/Recorder pair and the set of currently-connected
    browser queues. Runs the simulation in a background thread; each
    connected browser's SSE handler reads from its own per-connection
    queue, fed by this class whenever a new frame is produced.

    One broadcaster per server process -- the simulation is shared
    across all connected browsers (multiple browser tabs all watch the
    SAME running town, they don't each get their own simulation). This
    mirrors how the rate limiter in llm_decider.py is shared across all
    LLMDecider instances, for the same underlying reason: there is
    exactly one simulation, and multiple observers should see the same
    one, not accidentally multiply costs or diverge from each other.
    """

    def __init__(self, use_llm: bool) -> None:
        """
        Args:
            use_llm: if True, the first NUM_LLM_AGENTS agents use
                LLMDecider (requires GROQ_API_KEY); the rest stay
                rule-based, mirroring main_llm.py's mixed-population
                pattern exactly. If False (default), every agent is
                rule-based, mirroring main.py.
        """
        self.use_llm = use_llm
        self._subscribers: list = []
        self._subscribers_lock = threading.Lock()
        self._tick_count = 0
        self._running = False

        rng = random.Random(SEED)
        economy.reset_offers()
        governance.reset()
        chaos.reset_buzz()
        chaos.reset_factions()
        chaos.reset_corruption_cooldown()

        world = build_world()
        agents = build_agents(rng, NUM_AGENTS)

        if use_llm:
            from llm_decider import LLMDecider
            llm_agent_ids = list(agents.keys())[:NUM_LLM_AGENTS]
            for agent_id in llm_agent_ids:
                agents[agent_id].decider = LLMDecider(verbose=True)
            self.llm_agent_ids = llm_agent_ids
        else:
            self.llm_agent_ids = []

        self.engine = Engine(world, agents, rng=rng)
        self.recorder = Recorder(self.engine)

    def subscribe(self):
        """Register a new browser connection. Returns a Queue that the
        SSE handler should block-read from and forward to the client.
        Each subscriber gets every frame from the moment they connect
        onward (no replay of history before they joined).
        """
        q = queue.Queue()
        with self._subscribers_lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q) -> None:
        """Remove a browser connection, e.g. when it disconnects."""
        with self._subscribers_lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    def _broadcast(self, event_type: str, payload: dict) -> None:
        """Push one SSE event to every currently-connected subscriber."""
        message = {"type": event_type, **payload}
        with self._subscribers_lock:
            subscribers = list(self._subscribers)
        for q in subscribers:
            q.put(message)

    def run_forever(self) -> None:
        """The background thread's main loop: step the simulation
        forever, broadcasting each frame as it's produced. Ticks that
        take noticeably long (see the timing check below) signal to
        the frontend that something slow -- almost certainly an LLM
        call, possibly rate-limited -- is happening, keyed off elapsed
        wall-clock time per tick rather than any special signal from
        llm_decider.py (which prints to stdout, not captured here).
        This keeps live_server.py from needing any changes to
        llm_decider.py's internals to work.
        """
        self._running = True
        while self._running:
            tick_start = time.monotonic()
            self._broadcast("tick_started", {"tick": self.engine.world.tick})
            frame = self.recorder.step()
            elapsed = time.monotonic() - tick_start
            self._tick_count += 1
            self._broadcast("frame", {"frame": frame, "elapsed_seconds": round(elapsed, 2)})

            if not self.use_llm:
                time.sleep(RULE_BASED_TICK_DELAY_SECONDS)

    def stop(self) -> None:
        """Signal the background loop to stop after its current tick."""
        self._running = False

    def static_info(self) -> dict:
        """One-time payload sent to a browser when it first connects:
        location layout + agents_static (name/traits/decider_kind).
        """
        return {
            "locations": self.recorder.location_layout,
            "agents_static": self.recorder.agents_static_snapshot(),
            "llm_agent_ids": self.llm_agent_ids,
        }


def make_handler(broadcaster: SimulationBroadcaster):
    """Build a BaseHTTPRequestHandler subclass closed over `broadcaster`."""

    class Handler(BaseHTTPRequestHandler):
        """Routes: GET / (the HTML page), GET /static_info (one-time
        JSON), GET /stream (the SSE feed). Anything else gets a 404.
        """

        def log_message(self, format: str, *args) -> None:
            """Silence default per-request stderr logging -- with an
            SSE connection held open indefinitely, the default
            access-log behavior would flood the terminal.
            """
            pass

        def do_GET(self) -> None:
            """Route GET requests to the three supported endpoints (/,
            /static_info, /stream); anything else gets a 404.
            """
            if self.path == "/":
                self._serve_html()
            elif self.path == "/static_info":
                self._serve_json(broadcaster.static_info())
            elif self.path == "/stream":
                self._serve_stream()
            else:
                self.send_response(404)
                self.end_headers()

        def _serve_html(self) -> None:
            """Serve the single-page frontend (see `_HTML_PAGE` below)
            as a complete static HTML document.
            """
            body = _HTML_PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _serve_json(self, data: dict) -> None:
            """Serve `data` as a complete JSON response body."""
            body = json.dumps(data).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _serve_stream(self) -> None:
            """SSE endpoint. Blocks forever reading from this
            connection's subscriber queue and forwarding each message
            until the client disconnects.
            """
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()

            q = broadcaster.subscribe()
            try:
                while True:
                    message = q.get()
                    event_type = message.get("type", "message")
                    data = json.dumps(message)
                    chunk = f"event: {event_type}\ndata: {data}\n\n".encode("utf-8")
                    self.wfile.write(chunk)
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionError):
                pass
            finally:
                broadcaster.unsubscribe(q)

    return Handler


_HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>townsim live</title>
<style>
  body { font-family: -apple-system, sans-serif; background: #16151d; color: #e8e6f0; margin: 0; padding: 16px; }
  #layout { display: flex; gap: 16px; flex-wrap: wrap; }
  #mapCol { flex: 1 1 480px; min-width: 320px; }
  #sideCol { flex: 1 1 260px; min-width: 220px; display: flex; flex-direction: column; gap: 10px; }
  .panel { background: #201f2b; border-radius: 10px; padding: 0.75rem 1rem; }
  #status { font-size: 13px; color: #a39ee0; margin-bottom: 8px; }
  svg { width: 100%; background: #201f2b; border-radius: 10px; }
  canvas { max-height: 120px; }
  #eventFeed { font-size: 12px; line-height: 1.5; max-height: 160px; overflow-y: auto; }
  #detailOverlay { display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.5);
                   align-items: center; justify-content: center; }
  #detailBox { background: #201f2b; border-radius: 10px; padding: 1.25rem; max-width: 380px; }
  button { background: #3a3650; color: #e8e6f0; border: none; border-radius: 6px;
           padding: 6px 10px; cursor: pointer; }
</style>
</head>
<body>
<h1 style="font-size:18px;margin:0 0 4px">townsim live</h1>
<div id="status">connecting...</div>
<div id="layout">
  <div id="mapCol">
    <svg id="mapSvg" viewBox="0 0 1000 800" role="img"><title>Live town map</title></svg>
  </div>
  <div id="sideCol">
    <div class="panel"><p style="font-size:12px;color:#a39ee0;margin:0 0 2px">active rules</p>
      <p id="rulesLabel" style="font-size:13px;margin:0">none yet</p></div>
    <div class="panel"><p style="font-size:12px;color:#a39ee0;margin:0 0 2px">active crises</p>
      <p id="crisesLabel" style="font-size:13px;margin:0">none</p></div>
    <div class="panel"><p style="font-size:12px;color:#a39ee0;margin:0 0 6px">money by agent</p>
      <canvas id="moneyChart"></canvas></div>
    <div class="panel"><p style="font-size:12px;color:#a39ee0;margin:0 0 6px">reputation spread</p>
      <canvas id="repChart"></canvas></div>
    <div class="panel"><p style="font-size:12px;color:#a39ee0;margin:0 0 6px">this tick</p>
      <div id="eventFeed"></div></div>
  </div>
</div>
<div id="detailOverlay"><div id="detailBox">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
    <p id="detailName" style="font-weight:600;margin:0"></p>
    <button onclick="document.getElementById('detailOverlay').style.display='none'">close</button>
  </div>
  <div id="detailBody" style="font-size:13px;line-height:1.6"></div>
</div></div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<script>
let LOCS = {}, AGENTS = {}, LLM_IDS = [];
let history = [];
const palette = ['#378ADD','#1D9E75','#D85A30','#D4537E','#7F77DD','#BA7517','#888780','#639922',
                  '#A32D2D','#993556','#0F6E56','#854F0B','#534AB7','#993C1D','#185FA5','#3B6D11'];

async function init() {
  const info = await (await fetch('/static_info')).json();
  LOCS = info.locations; AGENTS = info.agents_static; LLM_IDS = info.llm_agent_ids;
  drawStaticMap();
  connectStream();
}

function drawStaticMap() {
  const svg = document.getElementById('mapSvg');
  svg.innerHTML = '';
  Object.entries(LOCS).forEach(([name, loc]) => {
    const r = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    r.setAttribute('x', loc.x - 70); r.setAttribute('y', loc.y - 40);
    r.setAttribute('width', 140); r.setAttribute('height', 80); r.setAttribute('rx', 12);
    r.setAttribute('fill', '#2b2840'); r.setAttribute('stroke', '#6b63a8'); r.setAttribute('stroke-width', '1');
    svg.appendChild(r);
    const t = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    t.setAttribute('x', loc.x); t.setAttribute('y', loc.y + 5);
    t.setAttribute('text-anchor', 'middle'); t.setAttribute('font-size', '15'); t.setAttribute('fill', '#c8c3ec');
    t.textContent = name.replace('_', ' ');
    svg.appendChild(t);
  });
  const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
  g.setAttribute('id', 'agentLayer');
  svg.appendChild(g);
}

function connectStream() {
  const es = new EventSource('/stream');
  const status = document.getElementById('status');
  es.addEventListener('tick_started', (ev) => {
    const d = JSON.parse(ev.data);
    status.textContent = 'tick ' + d.tick + ' \u2014 thinking...';
  });
  es.addEventListener('frame', (ev) => {
    const d = JSON.parse(ev.data);
    const frame = d.frame;
    history.push(frame);
    const slow = d.elapsed_seconds > 1.5;
    status.textContent = 'tick ' + frame.tick + (slow ? '  (call took ' + d.elapsed_seconds + 's)' : '') + '  \u2014 live';
    renderFrame(frame);
  });
  es.onerror = () => { status.textContent = 'disconnected \u2014 retrying...'; };
}

function renderFrame(frame) {
  const ruleKeys = Object.keys(frame.active_rules || {});
  document.getElementById('rulesLabel').textContent = ruleKeys.length ?
    ruleKeys.map(k => k.replace(/_/g, ' ')).join(', ') : 'none yet';
  const crises = frame.active_crises || [];
  document.getElementById('crisesLabel').textContent = crises.length ? crises.join(', ') : 'none';

  const svg = document.getElementById('mapSvg');
  const g = svg.querySelector('#agentLayer');
  g.innerHTML = '';
  const byLoc = {};
  Object.keys(LOCS).forEach(k => byLoc[k] = []);
  Object.entries(frame.agents).forEach(([id, st]) => { (byLoc[st.location] = byLoc[st.location] || []).push(id); });
  Object.entries(LOCS).forEach(([key, loc]) => {
    const list = byLoc[key] || [];
    list.forEach((id, i) => {
      const angle = (i / Math.max(list.length, 1)) * 2 * Math.PI;
      const rad = list.length > 1 ? 30 : 0;
      const cx = loc.x + Math.cos(angle) * rad;
      const cy = loc.y + 55 + Math.sin(angle) * 14;
      const circ = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      circ.setAttribute('cx', cx); circ.setAttribute('cy', cy); circ.setAttribute('r', 9);
      const idx = parseInt(id.split('_')[1], 10);
      circ.setAttribute('fill', palette[idx % palette.length]);
      if (LLM_IDS.includes(id)) { circ.setAttribute('stroke', '#fff'); circ.setAttribute('stroke-width', '2'); }
      circ.style.cursor = 'pointer';
      circ.addEventListener('click', () => showDetail(id, frame));
      g.appendChild(circ);
    });
  });

  const feed = document.getElementById('eventFeed');
  feed.innerHTML = (frame.events || []).map(e => describeEvent(e)).join('') ||
    '<div style="color:#6b6890">quiet tick</div>';

  updateMoneyChart(frame);
  updateRepChart();
}

function describeEvent(e) {
  const name = id => AGENTS[id] ? AGENTS[id].name : id;
  let text = e.kind.replace(/_/g, ' ');
  if (e.kind === 'speak') text = name(e.agent) + ' talks to ' + name(e.to);
  else if (e.kind === 'move') text = name(e.agent) + ' moves to ' + String(e.to).replace('_', ' ');
  else if (e.kind === 'gossip') text = name(e.agent) + ' gossips about ' + name(e.about);
  else if (e.kind === 'vote_cast') text = name(e.by) + ' votes ' + e.choice;
  else if (e.kind === 'trade_completed') text = name(e.from_) + ' trades with ' + name(e.to);
  else if (e.kind === 'corruption_scandal') text = name(e.agent) + ' embezzled ' + e.skimmed + '!';
  else if (e.kind === 'crisis_started') text = 'crisis: ' + e.crisis + ' begins';
  else if (e.kind === 'crisis_ended') text = 'crisis: ' + e.crisis + ' ends';
  else if (e.kind === 'rule_proposed') text = name(e.by) + ' proposes ' + e.rule_type;
  else if (e.kind === 'rule_repealed') text = 'a rule was repealed';
  return '<div style="padding:2px 0;border-bottom:1px solid #2e2b40">' + text + '</div>';
}

let moneyChart, repChart;
function updateMoneyChart(frame) {
  const ids = Object.keys(frame.agents);
  const labels = ids.map(id => AGENTS[id] ? AGENTS[id].name : id);
  const data = ids.map(id => frame.agents[id].money);
  if (!moneyChart) {
    moneyChart = new Chart(document.getElementById('moneyChart'), {
      type: 'bar',
      data: { labels, datasets: [{ data, backgroundColor: '#7f77dd' }] },
      options: { responsive: true, plugins: { legend: { display: false } },
        scales: { x: { ticks: { font: { size: 8 }, maxRotation: 90, minRotation: 90 }, grid: { color: '#2e2b40' } },
                  y: { ticks: { font: { size: 9 } }, grid: { color: '#2e2b40' } } } }
    });
  } else { moneyChart.data.datasets[0].data = data; moneyChart.update('none'); }
}

function updateRepChart() {
  const labels = history.map(f => f.tick);
  const mins = history.map(f => Math.min(...Object.values(f.agents).map(a => a.reputation)));
  const maxs = history.map(f => Math.max(...Object.values(f.agents).map(a => a.reputation)));
  if (!repChart) {
    repChart = new Chart(document.getElementById('repChart'), {
      type: 'line',
      data: { labels, datasets: [
        { data: mins, borderColor: '#e24b4a', pointRadius: 0, borderWidth: 1.5 },
        { data: maxs, borderColor: '#1d9e75', pointRadius: 0, borderWidth: 1.5 }] },
      options: { responsive: true, plugins: { legend: { display: false } },
        scales: { x: { display: false }, y: { min: 0, max: 1, ticks: { font: { size: 9 } }, grid: { color: '#2e2b40' } } } }
    });
  } else {
    repChart.data.labels = labels;
    repChart.data.datasets[0].data = mins;
    repChart.data.datasets[1].data = maxs;
    repChart.update('none');
  }
}

function showDetail(id, frame) {
  const a = AGENTS[id]; const st = frame.agents[id];
  document.getElementById('detailName').textContent = a.name + ' (' + id + ')' + (LLM_IDS.includes(id) ? ' [LLM]' : '');
  const traits = Object.entries(a.traits).map(([k, v]) =>
    '<div style="display:flex;justify-content:space-between"><span style="color:#a39ee0">' +
    k.replace(/_/g, ' ') + '</span><span>' + v.toFixed(2) + '</span></div>').join('');
  document.getElementById('detailBody').innerHTML =
    '<div>location \u00b7 ' + st.location.replace('_', ' ') + '</div>' +
    '<div>money \u00b7 ' + st.money.toFixed(2) + '</div>' +
    '<div>reputation \u00b7 ' + st.reputation.toFixed(2) + '</div>' +
    '<div style="margin:6px 0;color:#a39ee0">traits</div>' + traits +
    '<div style="margin-top:8px;color:#6b6890;font-size:12px">brain: ' + a.decider_kind + '</div>';
  document.getElementById('detailOverlay').style.display = 'flex';
}

init();
</script>
</body>
</html>
"""


def main() -> None:
    """Parse `--llm` from sys.argv, build the broadcaster, start the
    simulation in a background thread, and serve the web UI until
    interrupted with Ctrl+C.
    """
    import sys
    use_llm = "--llm" in sys.argv

    if use_llm:
        import os
        if not os.environ.get("GROQ_API_KEY"):
            print("--llm requires GROQ_API_KEY to be set. Run:")
            print("  export GROQ_API_KEY=your_key_here")
            return

    broadcaster = SimulationBroadcaster(use_llm=use_llm)
    sim_thread = threading.Thread(target=broadcaster.run_forever, daemon=True)
    sim_thread.start()

    handler_class = make_handler(broadcaster)
    server = ThreadingHTTPServer((HOST, PORT), handler_class)
    # Explicit, not relying on the implicit default (which happens to be
    # True on the Python version this was tested against, but this
    # project targets 3.10+ and that default isn't a documented
    # guarantee across the whole range) -- daemon request-handling
    # threads are what let the process actually exit on Ctrl+C even
    # while SSE connections are still open and blocked in q.get().
    server.daemon_threads = True
    print(f"townsim live server running at http://{HOST}:{PORT}/")
    mode_desc = (f"LLM-backed ({NUM_LLM_AGENTS} agents via Groq) + rule-based"
                 if use_llm else "fully rule-based")
    print(f"Mode: {mode_desc}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        broadcaster.stop()
        server.shutdown()


if __name__ == "__main__":
    main()
