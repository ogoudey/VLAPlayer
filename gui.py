"""
GUI(UI): browser-based front end for the policy inference machine.

Design
------
rerun.io has no built-in interactive controls (no buttons) — it's a data
*viewer*, not a control-panel framework. So the pieces are split:

  - rerun handles all the "display" duty: client latency, connection
    stats, and camera feeds are logged to a rerun gRPC server and shown
    via rerun's web viewer (WASM), which we host with `rr.serve_web_viewer`.
  - A tiny FastAPI app serves ONE page: that rerun web viewer in an
    <iframe> on one side, and a real HTML <button> + live status readout
    on the other, written and controlled by us. The button POSTs to
    /api/start, which calls self.start() (inherited trigger).

Owning awake() / start()
-------------------------
GUI fires the two signals itself rather than making outside code
sequence them:

    gui.bind_player(player)

  - awake() fires as soon as the control page's HTTP server is actually
    up and serving (i.e. the server "spinning up" is the awake trigger).
    In headless mode, where there's no server to wait on, it fires
    immediately on bind.
  - start() fires the moment the Start button is pressed (server-side,
    via /api/start -> self.start()).

Both are idempotent and order-independent: it doesn't matter whether you
call bind_player() before or after the display has launched, or before
or after the button's been pressed — each signal fires exactly once, as
soon as both "the signal happened" and "a player is bound" are true.

Data flow in
------------
`self.client`, `self.connection`, and `self.camera_set` are plain
attributes (as in the base UI) that you assign from outside, whenever
those objects exist:

    gui.client = my_client
    gui.connection = my_connection
    gui.camera_set = {cam1, cam2}

A background poll loop (10 Hz by default) looks at whatever is currently
assigned to those three attributes and, if the object implements the
small duck-typed protocols below, logs + displays it automatically. You
can also push data in explicitly at any time with report_client_latency(),
report_connection_stats(), and report_camera() — useful if you'd rather
call these from inside the objects themselves instead of polling.

Expected duck-typed interfaces (all optional, all checked with hasattr):

    client.last_latency_ms() -> float | None

    connection.is_connected() -> bool
    connection.get_stats() -> dict[str, float]

    camera.id -> str                      # attribute, not a method
    camera.is_connected() -> bool
    camera.latest_frame() -> np.ndarray | None   # HxWx3 uint8, or None

None of that is required — nothing breaks if client/connection/camera_set
are None or don't implement these; the poller just skips them.
"""

from __future__ import annotations

import threading
import time
import urllib.parse
from typing import Any, Dict, Optional

import rerun as rr
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

from ui import UI

_PAGE_TEMPLATE = """<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<title>{app_id}</title>
<style>
  html, body {{ margin: 0; height: 100%; font-family: -apple-system, Segoe UI, sans-serif; background: #111318; color: #e6e6e6; }}
  .layout {{ display: flex; height: 100%; }}
  .viewer {{ flex: 1 1 auto; border: none; background: #000; }}
  .sidebar {{ width: 280px; flex: 0 0 auto; padding: 16px; box-sizing: border-box;
              border-left: 1px solid #2a2d35; display: flex; flex-direction: column; gap: 14px; }}
  h1 {{ font-size: 15px; margin: 0 0 4px 0; color: #9aa4b2; font-weight: 600; }}
  .row {{ display: flex; justify-content: space-between; font-size: 13px; color: #c4c9d4; }}
  .row span:last-child {{ color: #fff; font-variant-numeric: tabular-nums; }}
  .dot {{ display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; }}
  .dot.on {{ background: #4ade80; }}
  .dot.off {{ background: #6b7280; }}
  button {{ padding: 12px; font-size: 15px; font-weight: 600; border-radius: 8px; border: none;
            cursor: pointer; background: #4f7cff; color: white; }}
  button:disabled {{ background: #2c2f38; color: #7a7f8a; cursor: default; }}
  #cameras div {{ font-size: 12px; color: #9aa4b2; }}
</style>
</head>
<body>
  <div class="layout">
    <iframe class="viewer" src="{viewer_url}"></iframe>
    <div class="sidebar">
      <div>
        <h1>{app_id}</h1>
        <div class="row"><span><span class="dot off" id="conn-dot"></span>connection</span><span id="conn-text">-</span></div>
      </div>
      <div class="row"><span>client latency</span><span id="latency">-</span></div>
      <div>
        <h1>cameras</h1>
        <div id="cameras"></div>
      </div>
      <button id="awake-btn">Awake</button>
      <button id="start-btn">Start</button>
    </div>
  </div>
  <script>
    const btnAwake = document.getElementById("awake-btn");
    btnAwake.addEventListener("click", () => {{
      btnAwake.disabled = true;
      btnAwake.textContent = "Waking...";
      fetch("/api/awake", {{ method: "POST" }}).catch(() => {{
        btnAwake.disabled = false;
        btnAwake.textContent = "Awake";
      }});
    }});

    const btnStart = document.getElementById("start-btn");
    btnStart.addEventListener("click", () => {{
      btnStart.disabled = true;
      fetch("/api/toggle", {{ method: "POST" }}).finally(() => {{
        btnStart.disabled = false;
      }});
    }});


    async function poll() {{
      try {{
        const r = await fetch("/api/status");
        const s = await r.json();

        if (s.awake) {{
            btnAwake.disabled = true;
            btnAwake.textContent = "Awakened";
        }}


        if (!s.started) {{
            btnStart.textContent = "Start";
            btnStart.disabled = false;
        }} else if (s.paused) {{
            btnStart.textContent = "Resume";
            btnStart.disabled = false;
        }} else {{
            btnStart.textContent = "Pause";
            btnStart.disabled = false;
        }}


        
        const dot = document.getElementById("conn-dot");
        const connText = document.getElementById("conn-text");
        dot.className = "dot " + (s.connection.connected ? "on" : "off");
        connText.textContent = s.connection.connected ? "connected" : "disconnected";

        const lat = document.getElementById("latency");
        lat.textContent = (s.client_latency_ms === null) ? "-" : s.client_latency_ms.toFixed(1) + " ms";

        const camDiv = document.getElementById("cameras");
        const ids = Object.keys(s.cameras);
        camDiv.innerHTML = ids.length === 0
          ? "<div>none</div>"
          : ids.map(id => {{
              const c = s.cameras[id];
              return `<div>${{id}}: ${{c.connected ? "connected" : "disconnected"}}</div>`;
            }}).join("");
      }} catch (e) {{ /* server not up yet, ignore */ }}
      setTimeout(poll, 500);
    }}
    poll();
  </script>
</body>
</html>
"""


class GUI(UI):
    def __init__(
        self,
        headless: bool = True,
        direct_start: bool = False,
        *,
        host: str = "127.0.0.1",
        http_port: int = 8000,
        grpc_port: int = 9876,
        web_port: int = 9090,
        app_id: str = "policy-inference-gui",
        poll_hz: float = 10.0,
    ):
        # IMPORTANT: UI.__init__ may call self.start() synchronously (when
        # direct_start=True), and that call lands on GUI.start() below via
        # normal polymorphism. So everything start() might touch has to
        # exist *before* we call super().__init__(). Hence this ordering.
        self._host = host
        self._http_port = http_port
        self._grpc_port = grpc_port
        self._web_port = web_port
        self._app_id = app_id
        self._poll_hz = poll_hz

        self._status_lock = threading.Lock()
        self._status: Dict[str, Any] = {
            "awake": False,
            "started": False,
            "paused": False,
            "client_latency_ms": None,
            "connection": {"connected": False, "stats": {}},
            "cameras": {},
        }
        self._start_lock = threading.Lock()

        self._display_active = False
        self._viewer_url: Optional[str] = None
        self._http_server: Optional[uvicorn.Server] = None
        self._http_thread: Optional[threading.Thread] = None
        self._poll_thread: Optional[threading.Thread] = None
        self._poll_stop = threading.Event()

        self._player: Optional[Any] = None
        self._awake_fired = False
        self._start_fired = False

        super().__init__(headless=headless, direct_start=direct_start)

        if not self.headless:
            self._launch_display()

    # ------------------------------------------------------------------ #
    # trigger
    # ------------------------------------------------------------------ #
    def start(self) -> None:
        if not self._awake_fired:
            self.awake()
        with self._start_lock:
            already = self._status["started"]
            with self._status_lock:
                self._status["started"] = True
            if not already and self._display_active:
                rr.log("/gui/events", rr.TextLog("Start triggered"))
        super().start()

    # ------------------------------------------------------------------ #
    # player wiring — GUI fires awake()/start() on the player itself
    # ------------------------------------------------------------------ #
    def bind_player(self, player: Any) -> None:
        """Register whatever exposes awake()/start() (your Player).

        Safe to call before or after the display is up, and before or
        after Start has been pressed — the right signal(s) fire
        immediately if their condition is already met.
        """
        self._player = player

    def awake(self) -> None:
        self._fire_awake()
        with self._status_lock:
            self._status["awake"] = True
        rr.log("/gui/events", rr.TextLog("Awake triggered"))

    def _fire_awake(self) -> None:
        if self._awake_fired or self._player is None:
            return
        awake = getattr(self._player, "awake", None)
        if callable(awake):
            self._awake_fired = True
            awake()

    def _fire_player_start(self) -> None:
        if self._start_fired or self._player is None:
            return
        start = getattr(self._player, "start", None)
        if callable(start):
            self._start_fired = True
            start()

    # ------------------------------------------------------------------ #
    # start / pause / unpause — the Start button becomes a pause toggle
    # once the initial start has happened
    # ------------------------------------------------------------------ #
    def toggle_start_pause(self) -> None:
        """What the (single) Start/Pause/Resume button calls.
 
        First press: behaves like start(). Every press after that toggles
        pause <-> unpause instead.
        """
        with self._status_lock:
            started = self._status["started"]
            paused = self._status["paused"]
        if not started:
            self.start()
        elif paused:
            self.unpause()
        else:
            self.pause()
 
    def pause(self) -> None:
        with self._status_lock:
            if not self._status["started"] or self._status["paused"]:
                return
            self._status["paused"] = True
        if self._display_active:
            rr.log("/gui/events", rr.TextLog("pause() triggered"))
        self._fire_player_pause()
 
    def unpause(self) -> None:
        with self._status_lock:
            if not self._status["started"] or not self._status["paused"]:
                return
            self._status["paused"] = False
        if self._display_active:
            rr.log("/gui/events", rr.TextLog("unpause() triggered"))
        self._fire_player_unpause()
 
    def _fire_player_pause(self) -> None:
        if self._player is None:
            return
        pause = getattr(self._player, "pause", None)
        if callable(pause):
            pause()
 
    def _fire_player_unpause(self) -> None:
        if self._player is None:
            return
        unpause = getattr(self._player, "unpause", None)
        if callable(unpause):
            unpause()


    # ------------------------------------------------------------------ #
    # public reporting API — call these directly, or rely on the poller
    # ------------------------------------------------------------------ #
    def report_client_latency(self, latency_ms: float) -> None:
        with self._status_lock:
            self._status["client_latency_ms"] = latency_ms
        if self._display_active:
            rr.log("client/latency_ms", rr.Scalars(latency_ms))

    def report_connection_stats(self, stats: Dict[str, float], connected: Optional[bool] = None) -> None:
        with self._status_lock:
            if connected is not None:
                self._status["connection"]["connected"] = connected
            self._status["connection"]["stats"].update(stats)
        if not self._display_active:
            return
        if connected is not None:
            rr.log("connection/connected", rr.Scalars(1.0 if connected else 0.0))
        for key, value in stats.items():
            if isinstance(value, (int, float)):
                rr.log(f"connection/{key}", rr.Scalars(float(value)))

    def report_camera(self, camera_id: str, frame=None, connected: Optional[bool] = None) -> None:
        with self._status_lock:
            entry = self._status["cameras"].setdefault(camera_id, {"connected": False})
            if connected is not None:
                entry["connected"] = connected
            if frame is not None:
                entry["last_frame_ts"] = time.time()
        if not self._display_active:
            return
        if frame is not None:
            rr.log(f"cameras/{camera_id}", rr.Image(frame))

    # ------------------------------------------------------------------ #
    # display setup
    # ------------------------------------------------------------------ #
    def _launch_display(self) -> None:
        rr.init(self._app_id, spawn=False)
        server_uri = rr.serve_grpc(grpc_port=self._grpc_port, server_memory_limit="200MB")

        try:
            rr.serve_web_viewer(open_browser=False, web_port=self._web_port, connect_to=server_uri)
        except TypeError:
            # Older/newer SDKs may name these kwargs differently — fall back
            # to the minimal call and rely on the ?url= query param below.
            rr.serve_web_viewer(open_browser=False, connect_to=server_uri)

        self._viewer_url = (
            f"http://{self._host}:{self._web_port}/?url={urllib.parse.quote(server_uri, safe='')}"
        )
        self._display_active = True

        app = self._build_http_app()
        config = uvicorn.Config(app, host=self._host, port=self._http_port, log_level="info")
        self._http_server = uvicorn.Server(config)
        self._http_thread = threading.Thread(target=self._http_server.run, daemon=True, name="gui-http")
        self._http_thread.start()

        self._poll_thread = threading.Thread(target=self._telemetry_loop, daemon=True, name="gui-telemetry")
        self._poll_thread.start()

        print(f"[GUI] control page: http://{self._host}:{self._http_port}/")

        # Fire awake() only once the server has actually finished binding
        # and is serving (uvicorn flips Server.started once that's done),
        # not just once its thread has been launched.
        threading.Thread(target=self._wait_for_serving_then_awake, daemon=True, name="gui-await-serving").start()

    def _wait_for_serving_then_awake(self, timeout: float = 10.0) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if getattr(self._http_server, "started", False):
                break
            time.sleep(0.02)

    def _build_http_app(self) -> FastAPI:
        app = FastAPI()

        @app.get("/", response_class=HTMLResponse)
        def index():
            return self._render_page()

        @app.post("/api/start")
        def api_start():
            self.start()
            return JSONResponse({"started": True})

        @app.post("/api/awake")
        def api_awake():
            self.awake()
            return JSONResponse({"started": True})

        @app.get("/api/status")
        def api_status():
            with self._status_lock:
                return JSONResponse(dict(self._status))

        @app.post("/api/toggle")
        def api_toggle():
            self.toggle_start_pause()
            with self._status_lock:
                return JSONResponse({"started": self._status["started"], "paused": self._status["paused"]})


        return app

    def _render_page(self) -> str:
        return _PAGE_TEMPLATE.format(
            app_id=self._app_id,
            viewer_url=self._viewer_url or "",
        )

    # ------------------------------------------------------------------ #
    # background polling of client / connection / camera_set
    # ------------------------------------------------------------------ #
    def _telemetry_loop(self) -> None:
        period = 1.0 / max(self._poll_hz, 0.1)
        while not self._poll_stop.is_set():
            self._poll_client()
            self._poll_connection()
            self._poll_cameras()
            time.sleep(period)

    def _poll_client(self) -> None:
        client = self.client
        if client is None:
            return
        getter = getattr(client, "last_latency_ms", None)
        if not callable(getter):
            return
        latency = getter()
        if latency is not None:
            self.report_client_latency(latency)

    def _poll_connection(self) -> None:
        connection = self.connection
        if connection is None:
            return
        is_conn = getattr(connection, "is_connected", None)
        get_stats = getattr(connection, "get_stats", None)
        connected = is_conn() if callable(is_conn) else None
        stats = get_stats() if callable(get_stats) else {}
        if connected is not None or stats:
            self.report_connection_stats(stats, connected=connected)

    def _poll_cameras(self) -> None:
        camera_set = self.camera_set
        if not camera_set:
            return
        for cam in camera_set.camera_connections:
            cam_id = getattr(cam, "id", None) or str(id(cam))
            is_conn = getattr(cam, "is_connected", None)
            get_frame = getattr(cam, "latest_frame", None)
            connected = is_conn() if callable(is_conn) else None
            frame = get_frame() if callable(get_frame) else None
            self.report_camera(cam_id, frame=frame, connected=connected)

    # ------------------------------------------------------------------ #
    def stop(self) -> None:
        self._poll_stop.set()
        if self._http_server is not None:
            self._http_server.should_exit = True
        if self._poll_thread is not None:
            self._poll_thread.join(timeout=2)
        if self._http_thread is not None:
            self._http_thread.join(timeout=2)