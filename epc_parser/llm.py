"""LLM access layer.

Responsibilities:
  * one place that talks to Gemini (swap the backend for tests or another provider)
  * content-addressed disk cache -> re-runs and crash-resumes cost nothing
  * retries with exponential backoff + jitter for 429/5xx/timeouts
  * finish-reason classification (MAX_TOKENS, RECITATION, SAFETY ...) returned to callers, who decide how to degrade
  * JSON extraction, pydantic validation, one repair attempt
  * token and cost accounting per model and per task
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import random
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from .config import Settings

log = logging.getLogger("epc.llm")
T = TypeVar("T", bound=BaseModel)

BLOCKED_REASONS = {"SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "IMAGE_SAFETY",
                   "IMAGE_RECITATION", "IMAGE_PROHIBITED_CONTENT", "LANGUAGE", "OTHER", "BLOCKED"}
# RECITATION fires when the output reproduces text the model recognises as memorised (standard-form
# boilerplate: bank-guarantee formats, safety guides, general conditions). It is a completed response, not
# a transient error, and re-issuing the SAME temperature-0 request reproduces it deterministically — so we
# retry with a small temperature perturbation to break the exact-match heuristic while keeping the read
# essentially verbatim. Callers opt in via call(recitation_retries=N).
RECITATION_REASONS = {"RECITATION", "IMAGE_RECITATION"}
RETRYABLE_HTTP = {408, 429, 500, 502, 503, 504}


def _parse_retry_after(err: Exception) -> float | None:
    """Best-effort: pull a server-advised delay (seconds) out of an APIError. Handles the
    "retryDelay": "12s" shape in RESOURCE_EXHAUSTED details and a numeric retry_after attribute."""
    for attr in ("retry_after", "retry_delay"):
        v = getattr(err, attr, None)
        if isinstance(v, (int, float)) and v > 0:
            return float(v)
    m = re.search(r'retry(?:_?delay|-?after)["\':\s]*?(\d+(?:\.\d+)?)\s*s', str(err), re.IGNORECASE)
    return float(m.group(1)) if m else None


def _looks_transient(err: Exception) -> bool:
    """Catch transient failures that don't arrive as a typed APIError (httpx read timeouts, stray 5xx
    strings) so the retry ladder still covers them."""
    s = str(err).lower()
    return any(t in s for t in (
        "429", "resource_exhausted", "quota", "rate limit", "rate-limit", "unavailable", "overloaded",
        "deadline", "timeout", "timed out", "temporarily", "try again", "500", "502", "503", "504",
        "connection reset", "connection aborted", "econnreset", "server disconnected", "remote end closed",
    ))


@dataclass
class RawResponse:
    text: str
    finish_reason: str
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0


@dataclass
class CallResult:
    parsed: BaseModel | None
    finish_reason: str
    raw_text: str = ""
    error: str | None = None
    from_cache: bool = False
    model: str = ""

    @property
    def ok(self) -> bool:
        return self.parsed is not None

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "MAX_TOKENS"

    @property
    def blocked(self) -> bool:
        return self.finish_reason in BLOCKED_REASONS


class Backend(Protocol):
    async def generate(self, *, model: str, system: str | None, parts: list[str | bytes], schema: type[BaseModel] | None,
                       thinking: str, media_resolution: str, max_output_tokens: int, meta: dict,
                       temperature: float | None = None) -> RawResponse: ...


class TransientError(Exception):
    """A retryable failure (429 / 5xx / timeout / connection). Carries an optional server-advised retry
    delay (seconds) parsed from the error — Gemini RESOURCE_EXHAUSTED often includes one."""

    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class GeminiBackend:
    def __init__(self, settings: Settings):
        from google import genai
        from google.genai import types

        self.types = types
        if settings.use_vertex:
            self.client = genai.Client(vertexai=True)
        else:
            if not settings.api_key:
                raise RuntimeError("Set GEMINI_API_KEY (or GOOGLE_API_KEY), or GOOGLE_GENAI_USE_VERTEXAI=true.")
            self.client = genai.Client(api_key=settings.api_key)
        self.settings = settings
        self._schema_disabled: set[str] = set()
        # capability probe: older google-genai builds reject thinking_config / media_resolution.
        try:
            fields = set(types.GenerateContentConfig.model_fields.keys())
        except Exception:
            fields = set()
        self._supports_media = "media_resolution" in fields and hasattr(types, "MediaResolution")
        self._supports_thinking = "thinking_config" in fields and hasattr(types, "ThinkingConfig")

    async def generate(self, *, model, system, parts, schema, thinking, media_resolution, max_output_tokens, meta,
                       temperature=None):
        from google.genai import errors

        t = self.types
        contents = [t.Part.from_bytes(data=p, mime_type="image/jpeg") if isinstance(p, bytes) else t.Part.from_text(text=p)
                    for p in parts]
        level = thinking.upper()
        if "pro" in model and level == "MINIMAL":  # Pro models do not support minimal thinking
            level = "LOW"
        use_schema = bool(schema) and self.settings.use_response_schema and schema.__name__ not in self._schema_disabled
        cfg_kw = dict(
            system_instruction=system,
            response_mime_type="application/json",
            response_schema=schema if use_schema else None,
            max_output_tokens=max_output_tokens,
            temperature=self.settings.temperature if temperature is None else temperature,
        )
        if self._supports_thinking:
            try:
                cfg_kw["thinking_config"] = t.ThinkingConfig(thinking_level=getattr(t.ThinkingLevel, level))
            except Exception:
                pass
        if self._supports_media:
            try:
                cfg_kw["media_resolution"] = getattr(t.MediaResolution, f"MEDIA_RESOLUTION_{media_resolution.upper()}")
            except Exception:
                pass
        cfg = t.GenerateContentConfig(**cfg_kw)
        try:
            resp = await self.client.aio.models.generate_content(
                model=model, contents=[t.Content(role="user", parts=contents)], config=cfg)
        except errors.APIError as e:
            if e.code in RETRYABLE_HTTP or _looks_transient(e):
                raise TransientError(f"HTTP {e.code}: {e}", _parse_retry_after(e)) from e
            if e.code == 400 and use_schema and "schema" in str(e).lower():
                log.warning("Response schema %s rejected by API; falling back to JSON mode", schema.__name__)
                self._schema_disabled.add(schema.__name__)
                return await self.generate(model=model, system=system, parts=parts, schema=schema, thinking=thinking,
                                           media_resolution=media_resolution, max_output_tokens=max_output_tokens,
                                           meta=meta, temperature=temperature)
            raise
        except (asyncio.TimeoutError, ConnectionError) as e:
            raise TransientError(str(e)) from e

        um = resp.usage_metadata
        usage = dict(input_tokens=(um.prompt_token_count or 0) if um else 0,
                     output_tokens=(um.candidates_token_count or 0) if um else 0,
                     thinking_tokens=(um.thoughts_token_count or 0) if um else 0)
        if not resp.candidates:
            block = getattr(getattr(resp, "prompt_feedback", None), "block_reason", None)
            # Keep the SPECIFIC prompt-block reason (SAFETY / BLOCKLIST / PROHIBITED_CONTENT / ...) rather
            # than a bare "BLOCKED", so an unreadable page's cause is never lost in the logs.
            reason = getattr(block, "name", None) or ("BLOCKED" if block else "OTHER")
            return RawResponse("", reason, **usage)
        cand = resp.candidates[0]
        finish = cand.finish_reason.name if cand.finish_reason else "STOP"
        text = "".join(p.text for p in (cand.content.parts if cand.content and cand.content.parts else [])
                       if getattr(p, "text", None) and not getattr(p, "thought", False))
        return RawResponse(text, finish, **usage)


@dataclass
class CostTracker:
    settings: Settings
    by_model: dict[str, dict[str, int]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(int)))
    by_task: dict[str, dict[str, float]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(float)))

    def add(self, task: str, model: str, r: RawResponse) -> None:
        m = self.by_model[model]
        m["calls"] += 1
        m["input_tokens"] += r.input_tokens
        m["output_tokens"] += r.output_tokens
        m["thinking_tokens"] += r.thinking_tokens
        p = self.settings.price_for(model)
        usd = r.input_tokens / 1e6 * p.input_per_m + (r.output_tokens + r.thinking_tokens) / 1e6 * p.output_per_m
        self.by_task[task]["calls"] += 1
        self.by_task[task]["usd"] += usd

    def report(self) -> dict[str, Any]:
        total = 0.0
        models = {}
        for name, m in self.by_model.items():
            p = self.settings.price_for(name)
            usd = m["input_tokens"] / 1e6 * p.input_per_m + (m["output_tokens"] + m["thinking_tokens"]) / 1e6 * p.output_per_m
            total += usd
            models[name] = {**m, "usd": round(usd, 4)}
        return {"total_usd": round(total, 4), "by_model": models,
                "by_task": {k: {"calls": int(v["calls"]), "usd": round(v["usd"], 4)} for k, v in self.by_task.items()}}


def extract_json(text: str) -> Any:
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        start = min([i for i in (t.find("{"), t.find("[")) if i >= 0], default=-1)
        end = max(t.rfind("}"), t.rfind("]"))
        if start >= 0 and end > start:
            return json.loads(t[start:end + 1])
        raise


class LLMClient:
    def __init__(self, settings: Settings, backend: Backend):
        self.s = settings
        self.backend = backend
        self.cost = CostTracker(settings)
        self._sem = asyncio.Semaphore(settings.concurrency)
        self.cache_dir = Path(settings.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def model_for(self, role: str) -> str:
        return {"reader": self.s.model_reader, "scanner": self.s.model_scanner, "reasoner": self.s.model_reasoner,
                "escalation": self.s.model_escalation, "summarizer": self.s.model_summarizer}[role]

    def _key(self, model, system, parts, schema, thinking, media, max_out) -> str:
        h = hashlib.sha256()
        for piece in (model, system or "", schema.__name__ if schema else "", json.dumps(schema.model_json_schema()) if schema else "",
                      thinking, media, str(max_out), str(self.s.temperature)):
            h.update(piece.encode())
            h.update(b"\x00")
        for p in parts:
            h.update(hashlib.sha256(p if isinstance(p, bytes) else p.encode()).digest())
        return h.hexdigest()

    async def call(self, task: str, *, role: str, system: str | None, parts: list[str | bytes], schema: type[T],
                   thinking: str = "minimal", media_resolution: str = "medium", max_output_tokens: int = 8192,
                   meta: dict | None = None, model: str | None = None, use_cache: bool = True,
                   recitation_retries: int = 0) -> CallResult:
        model = model or self.model_for(role)
        key = self._key(model, system, parts, schema, thinking, media_resolution, max_output_tokens)
        cache_file = self.cache_dir / key[:2] / f"{key}.json"
        if use_cache and cache_file.exists():
            try:
                data = json.loads(cache_file.read_text())
                return CallResult(schema.model_validate(data["parsed"]), data["finish_reason"], from_cache=True, model=model)
            except Exception:  # corrupt cache entry: ignore and recompute
                pass

        raw = await self._generate_with_retries(task, model, system, parts, schema, thinking, media_resolution,
                                                max_output_tokens, meta or {})
        if isinstance(raw, CallResult):
            return raw
        # RECITATION recovery (opt-in): a memorised-text block is deterministic at temperature 0, so
        # re-read with an escalating temperature to splinter the exact-match heuristic. The transcription
        # stays essentially verbatim (the model is reading printed text, not composing).
        rec_attempt = 0
        while raw.finish_reason in RECITATION_REASONS and rec_attempt < recitation_retries:
            rec_attempt += 1
            temp = min(1.0, 0.4 * rec_attempt)
            log.warning("%s RECITATION block; retry %d/%d at temperature %.1f", task, rec_attempt,
                        recitation_retries, temp)
            raw = await self._generate_with_retries(task + ":rec", model, system, parts, schema, thinking,
                                                    media_resolution, max_output_tokens, meta or {}, temperature=temp)
            if isinstance(raw, CallResult):
                return raw
        if raw.finish_reason not in ("STOP", "FINISH_REASON_UNSPECIFIED"):
            return CallResult(None, raw.finish_reason, raw.text, error=f"finish_reason={raw.finish_reason}", model=model)

        parsed, err = self._parse(raw.text, schema)
        if parsed is None:
            # one repair attempt: same inputs, told exactly what was wrong
            repair_parts = [*parts, f"Your previous answer was not valid for the required JSON schema: {err[:600]}\n"
                                    "Return the complete corrected JSON object only."]
            raw2 = await self._generate_with_retries(task + ":repair", model, system, repair_parts, schema, thinking,
                                                     media_resolution, max_output_tokens, meta or {})
            if isinstance(raw2, CallResult):
                return raw2
            if raw2.finish_reason not in ("STOP", "FINISH_REASON_UNSPECIFIED"):
                return CallResult(None, raw2.finish_reason, raw2.text, error=f"finish_reason={raw2.finish_reason}", model=model)
            parsed, err = self._parse(raw2.text, schema)
            if parsed is None:
                return CallResult(None, "INVALID_JSON", raw2.text, error=err, model=model)
            raw = raw2

        if use_cache:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache_file.with_suffix(".tmp")
            tmp.write_text(json.dumps({"parsed": parsed.model_dump(mode="json"), "finish_reason": raw.finish_reason}))
            tmp.replace(cache_file)
        return CallResult(parsed, raw.finish_reason, raw.text, model=model)

    async def _generate_with_retries(self, task, model, system, parts, schema, thinking, media, max_out, meta,
                                     temperature=None):
        attempt = 0
        while True:
            try:
                async with self._sem:
                    raw = await asyncio.wait_for(
                        self.backend.generate(model=model, system=system, parts=parts, schema=schema, thinking=thinking,
                                              media_resolution=media, max_output_tokens=max_out,
                                              meta={**meta, "task": task}, temperature=temperature),
                        timeout=self.s.request_timeout_s)
                self.cost.add(task, model, raw)
                return raw
            except Exception as e:
                transient = isinstance(e, (TransientError, asyncio.TimeoutError, ConnectionError)) or _looks_transient(e)
                if not transient:  # non-retryable (bad request, auth ...)
                    log.error("%s failed: %s", task, e)
                    return CallResult(None, "ERROR", error=str(e), model=model)
                attempt += 1
                if attempt > self.s.max_retries:
                    return CallResult(None, "ERROR", error=f"retries exhausted: {e}", model=model)
                # Exponential backoff with full jitter; a server-advised retry delay overrides (up to 120s).
                base = min(60.0, self.s.retry_base * (2 ** attempt))
                delay = random.uniform(0, base)
                ra = getattr(e, "retry_after", None)
                if ra:
                    delay = min(120.0, max(delay, ra))
                log.debug("%s transient error (%s); retry %d in %.1fs", task, e, attempt, delay)
                await asyncio.sleep(delay)

    @staticmethod
    def _parse(text: str, schema: type[T]) -> tuple[T | None, str]:
        try:
            return schema.model_validate(extract_json(text)), ""
        except (json.JSONDecodeError, ValidationError, ValueError) as e:
            return None, str(e)
