"""Exact replay of the LLM runs from llm_cache.jsonl without an Ollama server.
Usage (from the repo root): python scripts/replay_without_ollama.py . --config config.medium.json --out results/replay
Any cache miss raises an error, so a successful run is a bit-identical replay."""
import sys, runpy
sys.path.insert(0, sys.argv.pop(1))  # repo dir
from agentic_os import llm
class Miss(Exception): pass
llm.OllamaClient.model_info=lambda self: {"name": self.model, "digest": "cache-replay"}
llm.OllamaClient.warm_up=lambda self: None
_orig=llm.OllamaClient._post
def _post(self, path, body): raise Miss("cache miss: a new LLM call would be needed")
llm.OllamaClient._post=_post
runpy.run_path("run_experiment.py", run_name="__main__")
