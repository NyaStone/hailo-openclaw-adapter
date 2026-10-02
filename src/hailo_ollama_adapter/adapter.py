"""Hailo-to-OpenAI/Ollama API adapter.

A FastAPI adapter that exposes OpenAI- and Ollama-compatible HTTP endpoints
while proxying requests to a local Hailo 5.3.0 inference server.

Works around Hailo 5.3.0 prompt-renderer quirks: control-character rejection,
newline-in-content rejection, and system-role-on-continuation rejection.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from hailo_ollama_adapter.backend import (
    BackendBusyError,
    BackendGenerationError,
    BackendTimeoutError,
    BackendUnavailableError,
    NativeHailoBackend,
)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

REQUEST_TIMEOUT = 180.0

_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

logger = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(application: FastAPI) -> AsyncIterator[None]:
    backend = getattr(application.state, "inference_backend", None)
    if backend is None:
        queue_size = int(os.environ.get("HAILO_QUEUE_SIZE", "1"))
        backend = NativeHailoBackend(queue_size=queue_size)
        application.state.inference_backend = backend
    await backend.start(_configured_models())
    try:
        yield
    finally:
        await backend.close()


app = FastAPI(title="Hailo Adapter", version="1.0.0", lifespan=_lifespan)


# --------------------------------------------------------------------------- #
# Text sanitization
# --------------------------------------------------------------------------- #

def _sanitize(text: str) -> str:
    """Strip ASCII control chars that Hailo's parser rejects."""
    return _CONTROL_CHAR_RE.sub("", text)


def _flatten_newlines(text: str) -> str:
    """Collapse newlines to spaces.

    Hailo 5.3.0's prompt renderer re-encodes content through an internal
    template that doesn't escape newlines, so any literal newline -- even
    when the outer JSON correctly escapes it -- causes a parse error.
    """
    return text.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")


def _deep_sanitize(obj: Any) -> Any:
    if isinstance(obj, str):
        return _sanitize(obj)
    if isinstance(obj, dict):
        return {k: _deep_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_deep_sanitize(item) for item in obj]
    return obj


# --------------------------------------------------------------------------- #
# Conversation helpers
# --------------------------------------------------------------------------- #

def _extract_text(content: Any) -> str:
    """Coerce OpenAI-style content (string or list of parts) to plain text."""
    if isinstance(content, list):
        return " ".join(
            item.get("text", "")
            # Inference request translation
            if isinstance(item, dict) and item.get("type") == "text"
        )
            def _build_inference_request(
def _generation_options(request_data: dict) -> dict:
    ollama_options = request_data.get("options")
            ) -> tuple[dict, list[dict], dict, list[dict] | None, bool, str]:
                if not isinstance(request_data, dict):
                    raise HTTPException(status_code=400, detail="A JSON object is required")
        ollama_options if isinstance(ollama_options, dict) else {},
                if not isinstance(is_stream, bool):
                    raise HTTPException(status_code=400, detail="stream must be a boolean")
        request_data,
    ]
    generation = {}
    for source in sources:
        for source_name, target_name in _GENERATION_OPTION_ALIASES.items():
            if source_name in source:
                generation[target_name] = _deep_sanitize(source[source_name])
    return generation


# --------------------------------------------------------------------------- #
# Response formatting
# --------------------------------------------------------------------------- #

def to_openai_chunk(
    content: str,
    model: str,
    finish_reason: str | None = None,
    is_meta: bool = False,
) -> str:
    now = int(time.time())
    delta = (
        {"role": "assistant", "content": content}
        if is_meta
        else {"content": content}
    )
    chunk = {
        "id": f"chatcmpl-{now}",
        "object": "chat.completion.chunk",
        "created": now,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(chunk)}\n\n"


def _openai_full_response(content: str, model: str) -> dict:
    now = int(time.time())
    return {
        "id": f"chatcmpl-{now}",
        "object": "chat.completion",
        "created": now,
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop",
        }],
    }


def _ollama_full_response(content: str, model: str) -> dict:
    return {
        "model": model,
        "created_at": f"{int(time.time())}",
        "message": {"role": "assistant", "content": content},
        "done": True,
    }


# --------------------------------------------------------------------------- #
# Hailo client
# --------------------------------------------------------------------------- #


def _upstream_error_detail(response: httpx.Response) -> str:
    """Extract a bounded explicit error without reflecting arbitrary bodies."""
    detail: Any = None
    try:
        payload = response.json()
    except (json.JSONDecodeError, ValueError):
        payload = None

    if isinstance(payload, dict):
        detail = payload.get("error") or payload.get("detail")
        if isinstance(detail, dict):
            detail = detail.get("message")

    if not isinstance(detail, str) or not detail.strip():
        return f"Hailo upstream returned HTTP {response.status_code}"

    return _flatten_newlines(_sanitize(detail)).strip()[:MAX_UPSTREAM_ERROR_CHARS]


def _upstream_error_response(exc: httpx.HTTPStatusError) -> JSONResponse:
    """Preserve Hailo's status with a bounded downstream error response."""
    status = exc.response.status_code
    logger.warning("Hailo upstream rejected chat request: status=%d", status)
    return JSONResponse(
        status_code=status,
        content={"error": _upstream_error_detail(exc.response)},
    )


def _build_payload(
    request_data: dict,
    default_stream: bool,
) -> tuple[bytes, bool, str]:
    is_stream = request_data.get("stream", default_stream)
    public_model_id = request_data.get("model")
    if not isinstance(public_model_id, str) or not public_model_id:
        raise HTTPException(status_code=400, detail="A model ID is required")
    model = _find_configured_model(public_model_id)
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")
    messages = normalize_messages(request_data.get("messages", []))
    tools = request_data.get("tools")
    if tools is not None and not isinstance(tools, list):
        raise HTTPException(status_code=400, detail="tools must be a list")
    if tools is not None:
        tools = _deep_sanitize(tools)
    return (
        model,
        messages,
        _generation_options(request_data),
        tools,
        is_stream,
        public_model_id,
    )


def _get_backend() -> Any:
    backend = getattr(app.state, "inference_backend", None)
    if backend is None:
        raise BackendUnavailableError("Hailo backend has not started")
    return backend


def _backend_error_response(exc: Exception) -> JSONResponse:
    if isinstance(exc, BackendBusyError):
        return JSONResponse(status_code=503, content={"error": str(exc)})
    if isinstance(exc, BackendTimeoutError):
        return JSONResponse(status_code=504, content={"error": str(exc)})
    if isinstance(exc, BackendGenerationError):
        return JSONResponse(status_code=502, content={"error": str(exc)})
    return JSONResponse(status_code=503, content={"error": str(exc)})


# --------------------------------------------------------------------------- #
# Model discovery
# --------------------------------------------------------------------------- #

_VALIDATED_HEF_PROFILES = {
    "Qwen2.5-Coder-1.5B-Instruct.hef": {
        "details": {
            "format": "hef",
            "family": "qwen2",
            "families": ["qwen2"],
            "parameter_size": "1.5B",
        },
        "context_length": 2048,
        "capabilities": ["completion", "tools"],
    },
}


def _configured_models() -> list[dict]:
    """Return existing HEFs explicitly mapped to validated model profiles."""
    config_error = (
        "HAILO_MODELS must be a JSON object mapping public IDs to HEF paths"
    )
    raw_mapping = os.environ.get("HAILO_MODELS", "{}")
    try:
        mapping = json.loads(raw_mapping)
    except json.JSONDecodeError:
        logger.error(config_error)
        return []
    if not isinstance(mapping, dict):
        logger.error(config_error)
        return []

    models = []
    for model_id, raw_path in mapping.items():
        if (
            not isinstance(model_id, str)
            or not model_id
            or not isinstance(raw_path, str)
        ):
            continue
        path = Path(raw_path)
        profile = _VALIDATED_HEF_PROFILES.get(path.name)
        if profile is None:
            logger.warning("Ignoring HEF without a validated profile: %s", path.name)
            continue
        try:
            file_stat = path.stat()
        except OSError:
            continue
        if not path.is_file():
            continue

        family = profile["details"]["family"]
        models.append({
            "name": model_id,
            "model": model_id,
            "modified_at": datetime.fromtimestamp(
                file_stat.st_mtime,
                tz=timezone.utc,
            ).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            "size": file_stat.st_size,
            "digest": "",
            "details": profile["details"],
            "model_info": {f"{family}.context_length": profile["context_length"]},
            "capabilities": profile["capabilities"],
            "hef_path": str(path),
        })
    return models


def _find_configured_model(name: str) -> dict | None:
    return next(
        (model for model in _configured_models() if model["name"] == name),
        None,
    )


def _ollama_model_info(model: dict) -> dict:
    return {
        key: model[key]
        for key in ("name", "model", "modified_at", "size", "digest", "details")
    }


async def _get_models() -> list[dict]:
    """Read configured usable models without acquiring the inference slot."""
    return _configured_models()


async def _get_model_details(name: str) -> dict:
    model = _find_configured_model(name)
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")
    return model


# --------------------------------------------------------------------------- #
# Streaming generators
# --------------------------------------------------------------------------- #

async def _stream_openai(content: str, model: str) -> AsyncIterator[str]:
    """Frame a completed native text generation as OpenAI SSE."""
    yield to_openai_chunk("", model, is_meta=True)
    if content:
        yield to_openai_chunk(content, model)
    yield to_openai_chunk("", model, finish_reason="stop")
    yield "data: [DONE]\n\n"


async def _stream_ollama(content: str, model: str) -> AsyncIterator[str]:
    """Frame a completed native text generation as Ollama NDJSON."""
    yield json.dumps({
        "model": model,
        "created_at": f"{int(time.time())}",
        "message": {"role": "assistant", "content": content},
        "done": True,
    }) + "\n"


# --------------------------------------------------------------------------- #
# OpenAI-compatible endpoints
# --------------------------------------------------------------------------- #

@app.post("/chat/completions")
@app.post("/v1/chat/completions")
@app.post("/api/chat/completions")
async def chat_completions(request: Request) -> Any:
    """Serve OpenAI chat completions, defaulting requests to non-streaming."""
    try:
        native_model, messages, generation, tools, is_stream, model = _build_inference_request(
            await request.json(), default_stream=False,
        )
        backend = _get_backend()
        if tools is None:
            content = await backend.generate(native_model["hef_path"], messages, generation)
        else:
            content = await backend.generate(
                native_model["hef_path"], messages, generation, tools=tools,
            )
        if is_stream:
            return StreamingResponse(
                _stream_openai(content, model), media_type="text/event-stream",
            )
        return _openai_full_response(content, model)
    except HTTPException:
        raise
    except (BackendBusyError, BackendUnavailableError, BackendTimeoutError,
            BackendGenerationError) as exc:
        return _backend_error_response(exc)
    except Exception as exc:
        logger.exception("Error in chat adapter")
        return JSONResponse(status_code=500, content={"error": str(exc)})


@app.get("/models")
@app.get("/v1/models")
@app.get("/api/v1/models")
async def list_models() -> dict:
    """List available models (OpenAI-compatible endpoint)."""
    models = await _get_models()
    return {
        "object": "list",
        "data": [
            {
                "id": model["name"],
                "object": "model",
                "created": int(time.time()),
                "owned_by": "hailo",
                "permission": [],
                "root": model["name"],
                "parent": None,
            }
            for model in models
        ],
    }


@app.get("/models/{model_id}")
@app.get("/v1/models/{model_id}")
@app.get("/api/v1/models/{model_id}")
async def get_model(model_id: str) -> dict:
    """Return a single model object matching model_id (OpenAI-compatible).

    Uses the internal _get_models() cache and matches against the model's
    "name" or "model" fields. Returns 404 when not found.
    """
    models = await _get_models()
    for model in models:
        if model.get("name") == model_id or model.get("model") == model_id:
            return {
                "id": model["name"],
                "object": "model",
                "created": int(time.time()),
                "owned_by": "hailo",
                "permission": [],
                "root": model["name"],
                "parent": None,
            }
    raise HTTPException(status_code=404, detail="Model not found")


@app.get("/readyz")
async def readiness() -> Any:
    backend = getattr(app.state, "inference_backend", None)
    status = backend.status if backend is not None else {
        "status": "not_started",
        "ready": False,
        "error": "Hailo backend has not started",
        "queued": 0,
    }
    return JSONResponse(status_code=200 if status["ready"] else 503, content=status)


# --------------------------------------------------------------------------- #
# Ollama-compatible endpoints
# --------------------------------------------------------------------------- #

@app.get("/api/tags")
async def api_tags() -> dict:
    return {"models": [_ollama_model_info(model) for model in await _get_models()]}


@app.post("/api/tags/refresh")
async def api_tags_refresh() -> dict:
    """Re-read configured model paths and report the usable entries."""
    models = [_ollama_model_info(model) for model in await _get_models()]
    return {"models": models, "refreshed": True}


@app.post("/api/show")
async def api_show(request: Request) -> dict:
    try:
        body = await request.json()
    except (json.JSONDecodeError, ValueError):
        body = {}
    if not isinstance(body, dict):
        body = {}
    name = body.get("model") or body.get("name")
    if not isinstance(name, str) or not name:
        raise HTTPException(status_code=400, detail="A model ID is required")
    model = await _get_model_details(name)
    return {
        "details": model["details"],
        "model_info": model["model_info"],
        "capabilities": model["capabilities"],
    }


@app.post("/api/chat")
async def api_chat(request: Request) -> Any:
    """ Serve Ollama chat requests, defaulting to NDJSON streaming."""
    try:
        native_model, messages, generation, tools, is_stream, model = _build_inference_request(
            await request.json(), default_stream=True,
        )
        backend = _get_backend()
        if tools is None:
            content = await backend.generate(native_model["hef_path"], messages, generation)
        else:
            content = await backend.generate(
                native_model["hef_path"], messages, generation, tools=tools,
            )
        if is_stream:
            return StreamingResponse(
                _stream_ollama(content, model), media_type="application/x-ndjson",
            )
        return _ollama_full_response(content, model)
    except HTTPException:
        raise
    except (BackendBusyError, BackendUnavailableError, BackendTimeoutError,
            BackendGenerationError) as exc:
        return _backend_error_response(exc)
