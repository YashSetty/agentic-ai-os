"""Minimal Ollama chat client with a response cache.

The cache stores each response and its measured wall-clock latency, keyed by a
hash of (model, options, system prompt, user prompt). Re-running an experiment
therefore replays identical LLM decisions and decision latencies, which makes
results reproducible even though LLM inference is not bit-for-bit deterministic.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.request


class OllamaClient:
    def __init__(self, model: str, host: str, temperature: float, seed: int, num_ctx: int,
                 timeout_s: float, cache_path: str | None, num_predict: int = 128,
                 max_attempts: int = 3, max_plausible_latency_s: float = 60.0, **_):
        self.model, self.host = model, host.rstrip("/")
        # num_predict caps output length (JSON mode can otherwise emit whitespace for a long time)
        self.options = {"temperature": temperature, "seed": seed, "num_ctx": num_ctx,
                        "num_predict": num_predict}
        self.timeout_s = timeout_s
        self.max_attempts = max_attempts
        self.max_plausible_latency_s = max_plausible_latency_s
        self.n_retries = 0
        self.n_discarded_slow = 0
        self.cache_path = cache_path
        self.cache: dict[str, dict] = {}
        if cache_path and os.path.exists(cache_path):
            with open(cache_path, encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    if rec["latency_s"] <= self.max_plausible_latency_s:  # drop sleep-inflated entries
                        self.cache[rec["key"]] = rec

    def _post(self, path: str, body: dict) -> dict:
        req = urllib.request.Request(self.host + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
            return json.loads(r.read())

    def model_info(self) -> dict:
        with urllib.request.urlopen(self.host + "/api/tags", timeout=10) as r:
            tags = json.loads(r.read())
        for m in tags.get("models", []):
            if m["name"] == self.model:
                return {"name": m["name"], "digest": m["digest"], **m.get("details", {})}
        raise RuntimeError(f"model {self.model!r} not found in Ollama; run: ollama pull {self.model}")

    def warm_up(self):
        self._post("/api/chat", {"model": self.model, "stream": False, "keep_alive": "60m",
                                 "messages": [{"role": "user", "content": "ok"}],
                                 "options": {"num_predict": 1, "num_ctx": self.options["num_ctx"]}})

    def chat(self, system: str, user: str) -> tuple[str, float, bool]:
        key = hashlib.sha256(json.dumps([self.model, self.options, system, user]).encode()).hexdigest()
        if key in self.cache:
            rec = self.cache[key]
            return rec["content"], rec["latency_s"], True
        body = {"model": self.model, "stream": False, "format": "json", "keep_alive": "60m",
                "options": self.options,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        # Retry transport failures (timeouts / dropped connections) and implausibly long
        # calls. The wall clock keeps counting while the computer sleeps, so a call that
        # spans a system suspend would otherwise be charged hours of "decision latency".
        # The latency charged to the simulation is that of the accepted attempt.
        attempt = 0
        while True:
            attempt += 1
            try:
                t0 = time.perf_counter()
                resp = self._post("/api/chat", body)
                latency = time.perf_counter() - t0
            except (TimeoutError, OSError) as e:
                if attempt >= self.max_attempts:
                    raise RuntimeError(f"Ollama failed {attempt} times: {e}") from e
                self.n_retries += 1
                print(f"  ! LLM call failed ({e}); retrying in {5 * attempt}s", flush=True)
                time.sleep(5 * attempt)
                continue
            if latency > self.max_plausible_latency_s:
                self.n_discarded_slow += 1
                print(f"  ! LLM call took {latency:.0f}s (system sleep?); discarded and re-measured",
                      flush=True)
                continue  # not limited by max_attempts: a suspend is not a model failure
            break
        content = resp["message"]["content"]
        rec = {"key": key, "content": content, "latency_s": latency}
        self.cache[key] = rec
        if self.cache_path:
            with open(self.cache_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
        return content, latency, False
