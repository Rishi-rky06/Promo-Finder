"""Interchangeable model providers.

Every provider implements one method, ``generate_json(task_id, system, user, schema)``,
and returns parsed JSON plus usage metadata. The pipeline never knows which
model answered, so swapping models is a config change, not a code change.

Providers:
- ``anthropic``          Claude via the official Anthropic SDK (structured JSON output).
- ``openai_compatible``  Any OpenAI-compatible Chat Completions endpoint (Gemini free tier, Groq,
                         OpenRouter, vLLM/Ollama, or an internal gateway).
- ``file``               File exchange: writes prompts to disk, reads answers from disk. Any chat model
                         or person can answer; nothing is called over the network.
The ``heuristic`` fallback is not a provider (it can't answer arbitrary prompts); see heuristic.py.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path


class ProviderError(RuntimeError):
    """A request failed. ``fatal`` errors (bad credentials) abort the whole run."""

    def __init__(self, message: str, fatal: bool = False):
        super().__init__(message)
        self.fatal = fatal


class PendingResponse(RuntimeError):
    """File provider: the request was written but no answer exists yet."""


@dataclass
class ModelResult:
    data: dict
    raw_text: str
    provider: str
    model: str
    usage: dict = field(default_factory=dict)
    cached: bool = False
    elapsed_s: float = 0.0


def extract_json(text: str) -> dict:
    """Parse JSON that may be wrapped in a markdown fence or surrounded by prose."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()  # reasoning models
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            return json.loads(text[start:end + 1])
        raise


def prompt_hash(system: str, user: str, schema: dict, model: str = "") -> str:
    blob = json.dumps({"m": model, "s": system, "u": user, "j": schema}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class Provider:
    name = "base"
    model = ""
    live = True  # False for the file provider

    def generate_json(self, task_id: str, system: str, user: str, schema: dict) -> ModelResult:
        raise NotImplementedError

    def describe(self) -> str:
        return f"{self.name}:{self.model}"


class AnthropicProvider(Provider):
    name = "anthropic"

    def __init__(self, cfg: dict):
        import anthropic  # imported lazily so other providers work without it

        self._anthropic = anthropic
        c = cfg["provider"]["anthropic"]
        self.model = c["model"]
        self.effort = c.get("effort")
        self.max_tokens = int(c.get("max_tokens", 16000))
        # Haiku has no server-side refusal fallback; keep the option off for it.
        self.use_fallback = bool(c.get("use_refusal_fallback", True)) and "haiku" not in self.model
        try:
            self.client = anthropic.Anthropic(max_retries=4)
        except Exception as exc:  # SDK raises TypeError when no credential can be resolved
            raise ProviderError(f"Anthropic client could not be created (no usable credential?): {exc}", fatal=True)

    def generate_json(self, task_id, system, user, schema):
        a = self._anthropic
        output_config: dict = {"format": {"type": "json_schema", "schema": schema}}
        if self.effort:
            output_config["effort"] = self.effort
        kwargs = dict(model=self.model, max_tokens=self.max_tokens, system=system,
                      messages=[{"role": "user", "content": user}], output_config=output_config)
        t0 = time.time()
        try:
            if self.use_fallback:
                resp = self.client.beta.messages.create(
                    **kwargs, betas=["server-side-fallback-2026-07-01"], fallbacks="default")
            else:
                resp = self.client.messages.create(**kwargs)
        except TypeError as exc:  # raised by the SDK when no api_key / auth token can be resolved
            raise ProviderError(f"No usable Anthropic credential (set ANTHROPIC_API_KEY): {exc}", fatal=True)
        except a.AuthenticationError as exc:
            raise ProviderError(f"Authentication failed: {exc}", fatal=True)
        except a.PermissionDeniedError as exc:
            raise ProviderError(f"Permission denied: {exc}", fatal=True)
        except a.NotFoundError as exc:
            raise ProviderError(f"Model or endpoint not found ({self.model}): {exc}", fatal=True)
        except a.BadRequestError as exc:
            raise ProviderError(f"Bad request: {exc}")
        except a.RateLimitError as exc:
            raise ProviderError(f"Rate limited after retries: {exc}")
        except a.APIStatusError as exc:
            raise ProviderError(f"API error {exc.status_code}: {exc}")
        except a.APIConnectionError as exc:
            raise ProviderError(f"Connection error: {exc}")
        elapsed = time.time() - t0
        if resp.stop_reason == "refusal":
            details = getattr(resp, "stop_details", None)
            raise ProviderError(f"Model declined the request (refusal): {details}")
        if resp.stop_reason == "max_tokens":
            raise ProviderError("Response truncated at max_tokens; raise provider.anthropic.max_tokens")
        text = "".join(b.text for b in resp.content if b.type == "text")
        try:
            data = extract_json(text)
        except json.JSONDecodeError as exc:
            raise ProviderError(f"Model returned invalid JSON: {exc}")
        u = resp.usage
        usage = {"input_tokens": u.input_tokens, "output_tokens": u.output_tokens,
                 "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", 0) or 0,
                 "request_id": getattr(resp, "_request_id", None)}
        return ModelResult(data=data, raw_text=text, provider=self.name, model=resp.model, usage=usage,
                           elapsed_s=elapsed)


class OpenAICompatibleProvider(Provider):
    name = "openai_compatible"

    def __init__(self, cfg: dict):
        from openai import OpenAI

        c = cfg["provider"]["openai_compatible"]
        self.model = c.get("model") or ""
        if not self.model:
            raise ProviderError("provider.openai_compatible.model is empty; set it in your config", fatal=True)
        key_env = c.get("api_key_env", "OPENAI_API_KEY")
        key = os.environ.get(key_env, "")
        if not key:
            raise ProviderError(f"Environment variable {key_env} is not set", fatal=True)
        self.json_mode = c.get("json_mode", "schema")
        self.max_tokens = int(c.get("max_tokens", 16000))
        self.temperature = c.get("temperature")
        self.extra_body = c.get("extra_body") or None   # provider-specific request fields (e.g. reasoning off)
        self.client = OpenAI(base_url=c.get("base_url") or None, api_key=key, max_retries=4)

    def generate_json(self, task_id, system, user, schema):
        import openai

        if self.json_mode != "schema":
            system = system + "\n\nReturn ONLY a JSON object matching this JSON schema:\n" + json.dumps(schema)
        kwargs: dict = dict(model=self.model, max_tokens=self.max_tokens,
                            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body
        if self.json_mode == "schema":
            kwargs["response_format"] = {"type": "json_schema",
                                         "json_schema": {"name": "promo_output", "schema": schema, "strict": True}}
        elif self.json_mode == "object":
            kwargs["response_format"] = {"type": "json_object"}
        t0 = time.time()
        try:
            resp = self.client.chat.completions.create(**kwargs)
        except openai.AuthenticationError as exc:
            raise ProviderError(f"Authentication failed: {exc}", fatal=True)
        except openai.NotFoundError as exc:
            raise ProviderError(f"Model or endpoint not found ({self.model}): {exc}", fatal=True)
        except openai.BadRequestError as exc:
            raise ProviderError(f"Bad request (try json_mode='object' or 'prompt'): {exc}")
        except openai.APIStatusError as exc:
            raise ProviderError(f"API error {exc.status_code}: {exc}")
        except openai.APIConnectionError as exc:
            raise ProviderError(f"Connection error: {exc}")
        choice = resp.choices[0]
        if choice.finish_reason == "length":
            raise ProviderError("Response truncated; raise provider.openai_compatible.max_tokens")
        text = choice.message.content or ""
        try:
            data = extract_json(text)
        except json.JSONDecodeError as exc:
            raise ProviderError(f"Model returned invalid JSON: {exc}")
        usage = {}
        if resp.usage:
            usage = {"input_tokens": resp.usage.prompt_tokens, "output_tokens": resp.usage.completion_tokens}
        return ModelResult(data=data, raw_text=text, provider=self.name, model=resp.model or self.model,
                           usage=usage, elapsed_s=time.time() - t0)


class FileExchangeProvider(Provider):
    """Writes ``requests/<task>.json`` + a paste-ready ``.md``; reads ``responses/<task>.json``.

    If a prompt changes after a response was written, the old response is moved to
    ``responses/stale/`` so an answer is never matched to the wrong question.
    """

    name = "file"
    live = False

    def __init__(self, exchange_dir: Path, responder_label: str = "unspecified"):
        self.dir = Path(exchange_dir)
        self.req_dir = self.dir / "requests"
        self.resp_dir = self.dir / "responses"
        self.req_dir.mkdir(parents=True, exist_ok=True)
        self.resp_dir.mkdir(parents=True, exist_ok=True)
        self.model = responder_label

    def generate_json(self, task_id, system, user, schema):
        h = prompt_hash(system, user, schema)
        req_path = self.req_dir / f"{task_id}.json"
        resp_path = self.resp_dir / f"{task_id}.json"
        if req_path.exists():
            old = json.loads(req_path.read_text(encoding="utf-8")).get("prompt_sha256")
            if old != h and resp_path.exists():
                stale = self.resp_dir / "stale"
                stale.mkdir(exist_ok=True)
                shutil.move(str(resp_path), str(stale / f"{task_id}.{int(time.time())}.json"))
        req_path.write_text(json.dumps({"task_id": task_id, "prompt_sha256": h, "system": system, "user": user,
                                        "schema": schema}, ensure_ascii=False, indent=2), encoding="utf-8")
        (self.req_dir / f"{task_id}.md").write_text(
            f"<!-- task {task_id} | answer with JSON only and save it as responses/{task_id}.json -->\n\n"
            f"# Instructions\n\n{system}\n\n# Input\n\n{user}\n\n# Required JSON schema\n\n```json\n"
            f"{json.dumps(schema, indent=1)}\n```\n", encoding="utf-8")
        if not resp_path.exists():
            raise PendingResponse(task_id)
        text = resp_path.read_text(encoding="utf-8")
        try:
            data = extract_json(text)
        except json.JSONDecodeError as exc:
            raise ProviderError(f"responses/{task_id}.json is not valid JSON: {exc}")
        return ModelResult(data=data, raw_text=text, provider=self.name, model=self.model)


class CachingProvider(Provider):
    """Wraps a live provider; identical (model, prompt, schema) requests are served from disk."""

    def __init__(self, inner: Provider, cache_dir: Path):
        self.inner = inner
        self.name, self.model, self.live = inner.name, inner.model, inner.live
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def generate_json(self, task_id, system, user, schema):
        key = prompt_hash(system, user, schema, model=f"{self.inner.name}:{self.inner.model}")
        path = self.dir / f"{key}.json"
        if path.exists():
            d = json.loads(path.read_text(encoding="utf-8"))
            return ModelResult(data=d["data"], raw_text=d["raw_text"], provider=d["provider"], model=d["model"],
                               usage=d.get("usage", {}), cached=True)
        res = self.inner.generate_json(task_id, system, user, schema)
        path.write_text(json.dumps({"task_id": task_id, "data": res.data, "raw_text": res.raw_text,
                                    "provider": res.provider, "model": res.model, "usage": res.usage},
                                   ensure_ascii=False), encoding="utf-8")
        return res


def build_provider(cfg: dict, run_dir: Path, responder_label: str = "unspecified") -> Provider | None:
    """Return the configured provider, or None for the heuristic fallback."""
    name = cfg["provider"]["name"]
    if name == "heuristic":
        return None
    if name == "file":
        ex = cfg["provider"].get("file", {}).get("exchange_dir") or (run_dir / "exchange")
        return FileExchangeProvider(Path(ex), responder_label=responder_label)
    if name == "anthropic":
        inner: Provider = AnthropicProvider(cfg)
    elif name == "openai_compatible":
        inner = OpenAICompatibleProvider(cfg)
    else:
        raise ProviderError(f"Unknown provider '{name}'", fatal=True)
    return CachingProvider(inner, Path(cfg["provider"].get("cache_dir", ".cache/model_responses")))
