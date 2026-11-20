"""A fake Ollama server on localhost for the local-backend tests; the real one is never called.

Each request consumes the next scripted action, or echoes when the script is empty:
``("text", "...")`` answers, ``("status", code)`` fails, ``("hang",)`` never answers (the
client times out first), ``("empty",)`` answers with an empty completion, ``("raw", body)``
sends the body as is. A ``brain`` callable turns the prompt text into the reply instead.
GET ``/api/version`` and ``/api/tags`` (the installed ``models``) answer like the real server.
"""
import http.server
import json
import threading


def completion(text, finish_reason="stop"):
    return {"choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                         "finish_reason": finish_reason}]}


class FakeOllama:
    def __init__(self, script=(), brain=None, models=("tiny:1b",)):
        self.script, self.brain, self.models = list(script), brain, list(models)
        self.requests = []
        self.gets = []
        self._release = threading.Event()
        owner = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                owner.requests.append((self.path, json.loads(self.rfile.read(length))))
                owner._answer(self)

            def do_GET(self):
                owner.gets.append(self.path)
                owner._reply(self, 200, owner._info(self.path))

            def log_message(self, *args):
                pass

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, args=(0.01,),
                                       daemon=True)

    @property
    def base_url(self):
        return f"http://127.0.0.1:{self.server.server_address[1]}/v1"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop()

    def stop(self):
        self._release.set()
        self.server.shutdown()
        self.server.server_close()

    def _answer(self, handler):
        action = self.script.pop(0) if self.script else ("echo",)
        if action[0] == "hang":
            self._release.wait(10)
            return
        self._reply(handler, action[1] if action[0] == "status" else 200, self._body(action))

    def _info(self, path):
        if path == "/api/tags":
            return {"models": [{"name": name, "size": 1000} for name in self.models]}
        return {"version": "fake"}

    @staticmethod
    def _reply(handler, status, body):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        handler.send_response(status)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(data)))
        handler.end_headers()
        handler.wfile.write(data)

    def _body(self, action):
        kind = action[0]
        if kind == "text":
            return completion(action[1])
        if kind == "empty":
            return completion("", "length")
        if kind == "status":
            return {"error": {"message": f"scripted HTTP {action[1]}"}}
        if kind == "raw":
            return action[1]
        prompt = "\n".join(m["content"] for m in self.requests[-1][1]["messages"])
        return completion(self.brain(prompt) if self.brain else f"echo: {prompt[-40:]}")
