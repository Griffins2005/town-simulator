"""
live_server.py -- Eidolon Society Lab: a local web server that runs the
simulation in a background thread and streams one frame per tick to
connected browsers over Server-Sent Events (SSE).

Wraps Engine via Recorder from the OUTSIDE. The core does not know it
is being watched. Stdlib only (http.server + threading). Groq is needed
only for --llm.

Frames stream one-way over SSE. Pause, speed, and inject are a small
POST /control endpoint -- enough to freeze the town or inject a shock,
not a full bidirectional protocol.

Usage:
    python3 live_server.py              # rule-based agents (default)
    python3 live_server.py --llm        # mix in LLM-backed agents (needs GROQ_API_KEY)
    Then open http://localhost:8765/ (Observe)
    and http://localhost:8765/analytics (charts).

Binds to localhost only — not a public host. Push the repo, clone it
where you want the lab, and run this file there. Change PORT if 8765
is taken. Closing the process drops the town; use record_demo.py to
keep a trace.json.
"""

from __future__ import annotations

import json
import queue
import random
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_UI_PATH = _ROOT / "live_ui.html"
_ANALYTICS_PATH = _ROOT / "live_analytics.html"
_STATIC_DIR = _ROOT / "static"
_MIME = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
}

from engine import Engine
from recorder import Recorder
from town_factory import build_agents, build_world

import chaos
import economy
import faith
import governance
import inventions

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
        self.paused = False
        self.delay = RULE_BASED_TICK_DELAY_SECONDS
        self._control_lock = threading.Lock()

        rng = random.Random(SEED)
        economy.reset_offers()
        governance.reset()
        chaos.reset_buzz()
        chaos.reset_factions()
        chaos.reset_campaigns()
        chaos.reset_corruption_cooldown()
        inventions.reset()
        faith.reset()

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
            with self._control_lock:
                is_paused = self.paused
                delay = self.delay
            if is_paused:
                time.sleep(0.15)
                continue
            tick_start = time.monotonic()
            self._broadcast("tick_started", {"tick": self.engine.world.tick})
            frame = self.recorder.step()
            elapsed = time.monotonic() - tick_start
            self._tick_count += 1
            self._broadcast("frame", {"frame": frame, "elapsed_seconds": round(elapsed, 2)})
            if not self.use_llm:
                time.sleep(delay)

    def apply_control(self, body: dict) -> dict:
        """Pause, resume, or change tick delay. Called from POST /control."""
        cmd = body.get("cmd")
        with self._control_lock:
            if cmd == "pause":
                self.paused = True
            elif cmd == "resume":
                self.paused = False
            elif cmd == "speed":
                try:
                    self.delay = max(0.08, min(2.5, float(body.get("delay", self.delay))))
                except (TypeError, ValueError):
                    pass
            elif cmd == "inject":
                kind = body.get("kind")
                intensity = body.get("intensity", "serious")
                if kind == "headline":
                    text = str(body.get("text") or "an observer posted a notice")
                    self.engine.world.notice_board.append({
                        "tick": self.engine.world.tick,
                        "from": "observer",
                        "about": None,
                        "text": text[:160],
                    })
                    del self.engine.world.notice_board[:-8]
                elif kind in ("famine", "unrest", "bank_run", "market_shock", "flood"):
                    info = chaos.inject_crisis(
                        self.engine.world, self.engine.agents,
                        kind, intensity, self.engine.rng,
                    )
                    frame = self.recorder.peek()
                    self._broadcast("frame", {"frame": frame, "elapsed_seconds": 0, "painted": True})
                    return {
                        "paused": self.paused, "delay": self.delay,
                        "tick": self.engine.world.tick, **info,
                    }
            return {"paused": self.paused, "delay": self.delay, "tick": self.engine.world.tick}

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
        """Routes: GET / (Observe), GET /analytics, static assets,
        GET /static_info, GET /stream. Anything else gets a 404.
        """

        def log_message(self, format: str, *args) -> None:
            """Silence default per-request stderr logging -- with an
            SSE connection held open indefinitely, the default
            access-log behavior would flood the terminal.
            """
            pass

        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]
            if path == "/":
                self._serve_file(_UI_PATH)
            elif path == "/analytics":
                self._serve_file(_ANALYTICS_PATH)
            elif path in ("/live_shared.css", "/live_shared.js"):
                self._serve_file(_ROOT / path.lstrip("/"))
            elif path in ("/favicon.ico", "/apple-touch-icon.png"):
                self._serve_static(path.lstrip("/"))
            elif path.startswith("/static/"):
                self._serve_static(path[len("/static/"):])
            elif path == "/static_info":
                self._serve_json(broadcaster.static_info())
            elif path == "/stream":
                self._serve_stream()
            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self) -> None:
            if self.path != "/control":
                self.send_response(404)
                self.end_headers()
                return
            length = int(self.headers.get("Content-Length", "0") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            try:
                body = json.loads(raw.decode("utf-8") or "{}")
            except json.JSONDecodeError:
                body = {}
            self._serve_json(broadcaster.apply_control(body))

        def _serve_file(self, path: Path) -> None:
            if not path.is_file():
                self.send_response(404)
                self.end_headers()
                return
            body = path.read_bytes()
            mime = _MIME.get(path.suffix.lower(), "application/octet-stream")
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _serve_static(self, rel: str) -> None:
            candidate = (_STATIC_DIR / rel).resolve()
            if not str(candidate).startswith(str(_STATIC_DIR.resolve())) or not candidate.is_file():
                self.send_response(404)
                self.end_headers()
                return
            self._serve_file(candidate)

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
        try:
            from llm_decider import require_groq
            require_groq()
        except ModuleNotFoundError as exc:
            print(exc)
            return

    broadcaster = SimulationBroadcaster(use_llm=use_llm)
    sim_thread = threading.Thread(target=broadcaster.run_forever, daemon=True)
    sim_thread.start()

    handler_class = make_handler(broadcaster)
    server = ThreadingHTTPServer((HOST, PORT), handler_class)
    server.daemon_threads = True
    print(f"townsim live server running at http://{HOST}:{PORT}/", flush=True)
    print(f"Analytics page: http://{HOST}:{PORT}/analytics", flush=True)
    mode_desc = (f"LLM-backed ({NUM_LLM_AGENTS} agents via Groq) + rule-based"
                 if use_llm else "fully rule-based")
    print(f"Mode: {mode_desc}", flush=True)
    print("Press Ctrl+C to stop.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        broadcaster.stop()
        server.shutdown()


if __name__ == "__main__":
    main()
