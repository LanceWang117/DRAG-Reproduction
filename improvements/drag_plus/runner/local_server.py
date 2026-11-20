"""Reach or start the local Ollama server, and stop only what this process started.

A server that is already answering is used as it is and never stopped or reconfigured. A
server this class starts gets one parallel request, one loaded model and a 4096-token
context (the OpenAI endpoint cannot set the context per request), and ``close`` unloads
the model and stops the process. Every outside effect (process, HTTP, sleep) is injected,
so the tests start nothing.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional

DEFAULT_HOST = "127.0.0.1:11434"
_SERVER_SETTINGS = {"OLLAMA_CONTEXT_LENGTH": "4096", "OLLAMA_NUM_PARALLEL": "1",
                    "OLLAMA_MAX_LOADED_MODELS": "1"}
STOP_SECONDS = 10.0


class ServerError(RuntimeError):
    """The local server could not be started, reached or fed a model."""


class OllamaServer:
    def __init__(self, host: str = DEFAULT_HOST, *, ollama: str = "ollama",
                 environ: Optional[Mapping[str, str]] = None, log_path: Optional[Path] = None,
                 popen: Callable[..., Any] = subprocess.Popen,
                 run: Callable[..., Any] = subprocess.run,
                 opener: Callable[..., Any] = urllib.request.urlopen,
                 sleep: Callable[[float], None] = time.sleep, wait_seconds: float = 60.0):
        self.host, self.started = host, False
        self._ollama, self._log_path, self._wait = ollama, log_path, wait_seconds
        self._environ = dict(os.environ if environ is None else environ)
        self._popen, self._run, self._opener, self._sleep = popen, run, opener, sleep
        self._process: Any = None
        self._log: Any = None

    @property
    def url(self) -> str:
        return f"http://{self.host}"

    def _call(self, path: str, payload: Optional[dict] = None) -> Dict[str, Any]:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.url + path, data=data, method="GET" if data is None else "POST",
            headers={"Content-Type": "application/json"})
        with self._opener(request, timeout=10) as response:
            return json.loads(response.read().decode("utf-8") or "{}")

    def reachable(self) -> bool:
        try:
            self._call("/api/version")
        except (OSError, ValueError):
            return False
        return True

    def models(self) -> Dict[str, int]:
        """Installed model names with their sizes in bytes."""
        return {m["name"]: m.get("size", 0) for m in self._call("/api/tags").get("models", [])}

    def has_model(self, model: str) -> bool:
        installed = self.models()
        return model in installed or f"{model}:latest" in installed

    def ensure_running(self) -> None:
        if not self.reachable():
            self._start()

    def _start(self) -> None:
        env = {**self._environ, "OLLAMA_HOST": self.host, **_SERVER_SETTINGS}
        self._log = open(self._log_path, "ab") if self._log_path else subprocess.DEVNULL
        try:
            self._process = self._popen([self._ollama, "serve"], env=env, stdout=self._log,
                                        stderr=subprocess.STDOUT)
        except OSError as error:
            raise ServerError(f"cannot run `{self._ollama} serve`: {error}") from error
        self.started = True
        self._wait_until_reachable()

    def _wait_until_reachable(self) -> None:
        waited = 0.0
        while not self.reachable():
            code = self._process.poll()
            if code is not None:
                raise ServerError(f"`ollama serve` exited with code {code}; see {self._log_path}")
            if waited >= self._wait:
                raise ServerError(f"the server did not answer on {self.host} in {self._wait:g} s")
            self._sleep(0.5)
            waited += 0.5

    def pull(self, model: str) -> int:
        """Download ``model`` and return its size in bytes."""
        env = {**self._environ, "OLLAMA_HOST": self.host}
        try:
            done = self._run([self._ollama, "pull", model], env=env, capture_output=True, text=True)
        except OSError as error:
            raise ServerError(f"cannot run `{self._ollama} pull`: {error}") from error
        if done.returncode != 0:
            raise ServerError(f"`ollama pull {model}` failed: {done.stderr.strip()[-200:]}")
        return self.models().get(model, 0)

    def close(self, model: Optional[str] = None) -> None:
        """Unload the model and stop the server, but only a server this class started."""
        if not self.started:
            return
        if model is not None:
            try:
                self._call("/api/generate", {"model": model, "keep_alive": 0})
            except (OSError, ValueError):
                pass
        self._process.terminate()
        try:
            self._process.wait(timeout=STOP_SECONDS)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait()
        if self._log not in (None, subprocess.DEVNULL):
            self._log.close()
        self.started = False
