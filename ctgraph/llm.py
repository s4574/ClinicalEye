"""OpenAI-compatible multi-model client with a disk cache, a token / cost ledger and an optional budget cap.

- Local models are served by vLLM (scripts/serve_vlm.sh); API models are configured in $CE_LLM_API/*.yaml
  (see llm_api/example.yaml). No key is needed to replay cached responses.
- Every response is cached on disk under a hash of (model, messages, max_tokens, request kwargs[, sample index]); re-running
  a command after a crash resumes for free, and the same request is never billed twice.
- Every billed call is appended to $CE_WORK/runs/ledger.jsonl; CE_BUDGET_USD (default: no cap) stops new calls once the ledger
  total reaches it.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import yaml
from openai import OpenAI

from . import paths as P

API_DIR = P.LLM_API
RUN_DIR = P.RUNS
LEDGER = RUN_DIR / "ledger.jsonl"
CACHE = P.LLM_CACHE

# USD per 1M tokens (input, output, cached input), used for the ledger only. Local models cost nothing; GLM is billed in CNY
# on its own account (tokens are still logged).
PRICES = {
    "claude-sonnet-5": (3.0, 15.0, 0.3),
    "claude-sonnet-5-or": (2.0, 10.0, 0.2),      # Claude Sonnet 5 through OpenRouter
    "glm-5.3-flash": (0.0, 0.0, 0.0),
    "qwen3-vl-32b-local": (0.0, 0.0, 0.0),
    "medgemma-27b-local": (0.0, 0.0, 0.0),
    "internvl3.5-38b-local": (0.0, 0.0, 0.0),
}
# per-model request defaults: GLM-5.3-Flash thinks by default; thinking is switched off (official parameter), like the
# non-thinking Qwen3-VL-Instruct / MedGemma / InternVL runs
MODEL_KW = {"glm-5.3-flash": {"extra_body": {"thinking": {"type": "disabled"}}}}
# local vLLM servers (scripts/serve_vlm.sh <preset>); CE_VLLM_BASE_URL overrides the URL for one process (e.g. a second
# server of the same model on another port: the cache key does not include the URL)
LOCAL_MODELS = {"qwen3-vl-32b-local": {"base_url": "http://127.0.0.1:8000/v1", "model_name": "qwen3-vl-32b"},
                "medgemma-27b-local": {"base_url": "http://127.0.0.1:8010/v1", "model_name": "medgemma-27b"},
                "internvl3.5-38b-local": {"base_url": "http://127.0.0.1:8020/v1", "model_name": "internvl3.5-38b"}}
BUDGET_USD = float(os.environ.get("CE_BUDGET_USD", "inf"))

_lock = threading.Lock()


def _load_cfgs() -> dict:
    out = {}
    for f in sorted(API_DIR.glob("*.yaml")):
        d = yaml.safe_load(open(f)) or {}
        out.update(d.get("models", d))
    return out


def spent() -> float:
    if not LEDGER.exists():
        return 0.0
    return sum(json.loads(l)["cost"] or 0.0 for l in open(LEDGER))


class BudgetExceeded(RuntimeError):
    pass


def extract_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?|```$", "", (text or "").strip(), flags=re.M).strip()
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    # truncated JSON (output cut mid-rationale): recover the scalar fields that were already written
    out = {}
    for k in ("answer", "best_option"):
        m = re.search(rf'"{k}"\s*:\s*"([A-Fa-f])"', text or "")
        if m:
            out[k] = m.group(1).upper()
    m = re.search(r'"confidence"\s*:\s*([0-9.]+)', text or "")
    if m and out:
        out["confidence"] = float(m.group(1))
    if out:
        out["_truncated"] = True
    return out


class LLM:
    def __init__(self, model: str, tag: str, budget_usd: float = BUDGET_USD, sample: int | None = None,
                 temperature: float | None = None):
        """sample: re-sampling index that only enters the CACHE KEY (the request is byte-identical), so independent
        samples of the same prompt can be drawn; temperature: sent explicitly when given."""
        self.sample, self.temperature = sample, temperature
        if model not in PRICES:
            raise ValueError(f"no price for {model}; add it to PRICES first")
        self.model, self.tag, self.budget = model, tag, budget_usd
        self._client, self.api_model = None, None
        RUN_DIR.mkdir(parents=True, exist_ok=True)
        CACHE.mkdir(parents=True, exist_ok=True)

    @property
    def client(self) -> OpenAI:
        """created on the first cache miss, so cached runs need neither a server nor a key"""
        if self._client is None:
            if self.model in LOCAL_MODELS:
                import httpx
                cfg = LOCAL_MODELS[self.model]
                self.api_model, self.budget = cfg["model_name"], float("inf")
                self._client = OpenAI(base_url=os.environ.get("CE_VLLM_BASE_URL", cfg["base_url"]), api_key="local",
                                      max_retries=2, http_client=httpx.Client(trust_env=False, timeout=900))
            else:
                cfgs = _load_cfgs()
                if self.model not in cfgs:
                    raise KeyError(f"no API config for {self.model} in {API_DIR} (see llm_api/example.yaml)")
                cfg = cfgs[self.model]
                self.api_model = cfg.get("model_name", self.model)
                if PRICES[self.model] == (0.0, 0.0, 0.0):    # billed elsewhere: not limited by the cap
                    self.budget = float("inf")
                self._client = OpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"], timeout=300, max_retries=2)
        return self._client

    def chat(self, messages: list[dict], max_tokens: int | None = None, cache_salt: str | None = None, **kw) -> dict:
        """cache_salt: extra cache-key component (never sent), e.g. to resend an identical request after a broken reply."""
        kw = {**MODEL_KW.get(self.model, {}), **kw}
        if self.temperature is not None and "temperature" not in kw:
            kw["temperature"] = self.temperature
        salt = [] if self.sample is None else [{"sample": self.sample}]       # cache key only, never sent
        if cache_salt is not None:
            salt = salt + [{"salt": cache_salt}]
        key = hashlib.sha256(json.dumps([self.model, messages, max_tokens, kw] + salt, sort_keys=True).encode()).hexdigest()
        cf = CACHE / f"{key}.json"
        if cf.exists():
            try:
                return json.load(open(cf))
            except (json.JSONDecodeError, OSError):    # a partially written / corrupt entry counts as a miss
                pass
        client = self.client
        with _lock:
            if spent() >= self.budget:
                raise BudgetExceeded(f"spent {spent():.2f} >= {self.budget}")
        last = None
        for attempt in range(4):
            try:
                t = time.time()
                args = dict(model=self.api_model, messages=messages, **kw)
                if max_tokens:
                    args["max_tokens"] = max_tokens
                r = client.chat.completions.create(**args)
                break
            except Exception as e:  # transient API errors
                last = e
                time.sleep(5 * (attempt + 1))
        else:
            raise last
        u = r.usage
        pin, pout, pcache = PRICES[self.model]
        cached = getattr(getattr(u, "prompt_tokens_details", None), "cached_tokens", None) or 0
        cost = ((u.prompt_tokens - cached) * pin + cached * pcache + u.completion_tokens * pout) / 1e6
        out = {"text": r.choices[0].message.content or "", "in": u.prompt_tokens, "out": u.completion_tokens,
               "cost": cost, "latency": time.time() - t}
        tcs = getattr(r.choices[0].message, "tool_calls", None)
        if tcs:     # OpenAI function calling (DeepTumorVQA reference agent)
            out["tool_calls"] = [{"id": tc.id, "name": tc.function.name, "arguments": tc.function.arguments or "{}"} for tc in tcs]
        with _lock:
            with open(LEDGER, "a") as f:
                det = getattr(u, "completion_tokens_details", None)
                f.write(json.dumps({"tag": self.tag, "model": self.model, "served_model": getattr(r, "model", None),
                                    "in": u.prompt_tokens, "out": u.completion_tokens,
                                    "reasoning": getattr(det, "reasoning_tokens", None), "cached": cached, "cost": cost,
                                    "ts": time.time()}) + "\n")
        # atomic write: a concurrent reader never sees a half-written file
        tmp = cf.with_suffix(f".json.tmp.{os.getpid()}.{threading.get_ident()}")
        with open(tmp, "w") as fh:
            json.dump(out, fh)
        os.replace(tmp, cf)
        return out


def user_msg(text: str, images_b64: list[str] = ()) -> dict:
    content = [{"type": "text", "text": text}]
    for b in images_b64:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b}"}})
    return {"role": "user", "content": content}


def pmap(fn, items, workers: int = 8):
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(fn, items))
