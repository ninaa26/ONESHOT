"""Live WebSocket publisher for RL training visualization. Our RL training."""

from __future__ import annotations

import asyncio
import json
import threading
from dataclasses import dataclass, field
from typing import Any

from websockets.exceptions import ConnectionClosed
from websockets.server import WebSocketServerProtocol, serve


@dataclass(slots=True)
class LiveTrainingVisServer:
    """Broadcast JSON state messages to browser clients during training."""

    host: str = "127.0.0.1"
    port: int = 8765
    clients: set[WebSocketServerProtocol] = field(default_factory=set, init=False)
    _loop: asyncio.AbstractEventLoop | None = field(default=None, init=False)
    _thread: threading.Thread | None = field(default=None, init=False)
    _ready: threading.Event = field(default_factory=threading.Event, init=False)
    _closed: threading.Event = field(default_factory=threading.Event, init=False)
    _latest_payload: str | None = field(default=None, init=False)
    _start_error: Exception | None = field(default=None, init=False)

    def start(self) -> None:
        """Start the WebSocket server in a background thread."""
        if self._thread is not None:
            return
        self._closed.clear()
        self._ready.clear()
        self._start_error = None
        self._thread = threading.Thread(target=self._run_loop, name="rl-live-vis", daemon=True)
        self._thread.start()
        ready = self._ready.wait(timeout=5.0)
        if self._start_error is not None:
            err = self._start_error
            self._thread = None
            self._loop = None
            msg = f"Failed to start live visualization websocket server on {self.host}:{self.port}: {err}"
            raise RuntimeError(msg) from err
        if not ready or self._loop is None:
            self._thread = None
            self._loop = None
            msg = f"Timed out starting live visualization websocket server on {self.host}:{self.port}."
            raise RuntimeError(msg)

    def publish(self, message: dict[str, Any]) -> None:
        """Queue a state payload for all connected clients."""
        loop = self._loop
        if loop is None or self._closed.is_set():
            return
        payload = json.dumps(message)
        self._latest_payload = payload
        asyncio.run_coroutine_threadsafe(self._broadcast(payload), loop)

    def close(self) -> None:
        """Stop the background server thread."""
        loop = self._loop
        thread = self._thread
        if loop is None or thread is None:
            return
        self._closed.set()
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5.0)
        self._thread = None
        self._loop = None
        self.clients.clear()

    def _run_loop(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        try:
            server = loop.run_until_complete(serve(self._handle_client, self.host, self.port))
        except Exception as exc:
            self._start_error = exc
            self._ready.set()
            return
        self._ready.set()
        try:
            loop.run_forever()
        finally:
            server.close()
            loop.run_until_complete(server.wait_closed())
            pending = [task for task in asyncio.all_tasks(loop) if not task.done()]
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.close()

    async def _handle_client(self, ws: WebSocketServerProtocol, path: str) -> None:
        del path  # unused
        self.clients.add(ws)
        if self._latest_payload is not None:
            await ws.send(self._latest_payload)
        try:
            async for _ in ws:
                # Browser may send helm controls; training vis is output-only.
                pass
        finally:
            self.clients.discard(ws)

    async def _broadcast(self, payload: str) -> None:
        stale_clients: list[WebSocketServerProtocol] = []
        for client in tuple(self.clients):
            try:
                await client.send(payload)
            except ConnectionClosed:
                stale_clients.append(client)
        for client in stale_clients:
            self.clients.discard(client)
