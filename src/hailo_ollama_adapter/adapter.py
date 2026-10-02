"""Direct HailoRT backend with OpenAI- and Ollama-compatible APIs.

The native runtime is initialized lazily through the FastAPI lifespan so
protocol tests and fake backends do not require HailoRT bindings.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
import xml.etree.ElementTree as ET
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from jsonschema import Draft202012Validator, SchemaError, ValidationError

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

_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

logger = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(application: FastAPI) -> AsyncIterator[None]:
    backend = getattr(application.state, "inference_backend", None)
    if backend is None:
        try:
            queue_size = int(os.environ.get("HAILO_QUEUE_SIZE", "1"))
        except ValueError:
            queue_size = 0
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
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        )
    return content if isinstance(content, str) else str(content)


def normalize_messages(messages: list[dict]) -> list[dict]:
    normalized = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        normalized_message = _deep_sanitize(message)
        normalized_message["role"] = normalized_message.get("role", "user")
        normalized_message["content"] = _sanitize(
            _extract_text(message.get("content", ""))
        )
        normalized.append(normalized_message)
    return normalized


_GENERATION_OPTION_ALIASES = {
    "temperature": "temperature",
    "top_p": "top_p",
    "top_k": "top_k",
    "frequency_penalty": "frequency_penalty",
    "seed": "seed",
    "do_sample": "do_sample",
    "num_predict": "max_generated_tokens",
    "max_generated_tokens": "max_generated_tokens",
    "max_tokens": "max_generated_tokens",
    "max_completion_tokens": "max_generated_tokens",
}


def _generation_options(request_data: dict) -> dict:
    ollama_options = request_data.get("options")
    sources = [
        ollama_options if isinstance(ollama_options, dict) else {},
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
    is_role_header: bool = False,
    tool_calls: list[dict[str, Any]] | None = None,
) -> str:
    now = int(time.time())
    delta = (
        {"role": "assistant", "content": content}
        if is_role_header
        else {"content": content}
    )
    if tool_calls:
        delta["tool_calls"] = tool_calls
    chunk = {
        "id": f"chatcmpl-{now}",
        "object": "chat.completion.chunk",
        "created": now,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(chunk)}\n\n"


def _openai_tool_call(call: dict[str, Any], index: int | None = None) -> dict:
    formatted = {"id": call["id"], "type": "function"}
    if index is not None:
        formatted["index"] = index
    formatted["function"] = {
        "name": call["name"],
        "arguments": json.dumps(
            call["arguments"],
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    }
    return formatted


def _openai_full_response(result: ChatResult, model: str) -> dict:
    now = int(time.time())
    message: dict[str, Any] = {
        "role": "assistant",
        "content": result.content if result.content or not result.tool_calls else None,
    }
    if result.tool_calls:
        message["tool_calls"] = [_openai_tool_call(call) for call in result.tool_calls]
    return {
        "id": f"chatcmpl-{now}",
        "object": "chat.completion",
        "created": now,
        "model": model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": "tool_calls" if result.tool_calls else "stop",
        }],
    }


def _ollama_full_response(result: ChatResult, model: str) -> dict:
    message: dict[str, Any] = {"role": "assistant", "content": result.content}
    if result.tool_calls:
        message["tool_calls"] = [
            {
                "id": call["id"],
                "type": "function",
                "function": {
                    "name": call["name"],
                    "arguments": call["arguments"],
                },
            }
            for call in result.tool_calls
        ]
    return {
        "model": model,
        "created_at": f"{int(time.time())}",
        "message": message,
        "done": True,
    }


# --------------------------------------------------------------------------- #
# Direct inference request translation
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ChatRequest:
    hef_path: str
    messages: list[dict]
    generation: dict
    tools: list[dict] | None
    tool_inventory: dict[str, dict]
    tool_choice: str
    stream: bool
    public_model_id: str


@dataclass(frozen=True)
class ChatResult:
    content: str
    tool_calls: list[dict[str, Any]]


def _validate_tool_inventory(tools: list[dict] | None) -> dict[str, dict]:
    inventory = {}
    for tool in tools or []:
        if (
            not isinstance(tool, dict)
            or tool.get("type", "function") != "function"
            or not isinstance(tool.get("function"), dict)
        ):
            raise HTTPException(
                status_code=400,
                detail="Each tool must define a function",
            )
        function = tool["function"]
        name = function.get("name")
        schema = function.get("parameters", {"type": "object"})
        if not isinstance(name, str) or not name or not isinstance(schema, dict):
            raise HTTPException(status_code=400, detail="Invalid function tool schema")
        if name in inventory:
            raise HTTPException(status_code=400, detail="Tool names must be unique")
        try:
            Draft202012Validator.check_schema(schema)
        except SchemaError as exc:
            raise HTTPException(
                status_code=400,
                detail="Invalid function JSON schema",
            ) from exc
        inventory[name] = schema
    return inventory


def _build_inference_request(
    request_data: dict,
    default_stream: bool,
) -> ChatRequest:
    if not isinstance(request_data, dict):
        raise HTTPException(status_code=400, detail="A JSON object is required")
    is_stream = request_data.get("stream", default_stream)
    if not isinstance(is_stream, bool):
        raise HTTPException(status_code=400, detail="stream must be a boolean")
    public_model_id = request_data.get("model")
    if not isinstance(public_model_id, str) or not public_model_id:
        raise HTTPException(status_code=400, detail="A model ID is required")
    model = _find_configured_model(public_model_id)
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")
    messages = normalize_messages(request_data.get("messages", []))
    tools = request_data.get("tools")
    tool_inventory = {}
    if tools is not None and not isinstance(tools, list):
        raise HTTPException(status_code=400, detail="tools must be a list")
    if tools is not None:
        tools = _deep_sanitize(tools)
        tool_inventory = _validate_tool_inventory(tools)
    tool_choice = request_data.get("tool_choice", "auto")
    if not isinstance(tool_choice, str) or tool_choice not in ("auto", "none"):
        raise HTTPException(
            status_code=400,
            detail="tool_choice must be 'auto' or 'none' for this backend",
        )
    return ChatRequest(
        hef_path=model["hef_path"],
        messages=messages,
        generation=_generation_options(request_data),
        tools=tools,
        tool_inventory=tool_inventory,
        tool_choice=tool_choice,
        stream=is_stream,
        public_model_id=public_model_id,
    )


def _get_backend() -> Any:
    backend = getattr(app.state, "inference_backend", None)
    if backend is None:
        raise BackendUnavailableError("Hailo backend has not started")
    return backend


async def _run_chat_request(
    request_data: dict,
    default_stream: bool,
) -> tuple[ChatRequest, ChatResult]:
    chat_request = _build_inference_request(request_data, default_stream)
    generated = await _get_backend().generate(
        chat_request.hef_path,
        chat_request.messages,
        chat_request.generation,
        tools=chat_request.tools,
        tool_choice=chat_request.tool_choice,
    )
    if not isinstance(generated, str):
        raise BackendGenerationError("Native Hailo returned an invalid response")
    result = _parse_generated_response(
        generated,
        chat_request.tool_inventory if chat_request.tool_choice == "auto" else {},
    )
    return chat_request, result


def _xml_value(element: ET.Element) -> Any:
    children = list(element)
    if children:
        values: dict[str, Any] = {}
        for child in children:
            value = _xml_value(child)
            if child.tag in values:
                existing = values[child.tag]
                values[child.tag] = (
                    existing + [value]
                    if isinstance(existing, list)
                    else [existing, value]
                )
            else:
                values[child.tag] = value
        return values
    text = (element.text or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _parse_tool_payload(payload: str) -> dict[str, Any]:
    try:
        call = json.loads(payload)
    except json.JSONDecodeError as json_error:
        try:
            root = ET.fromstring(f"<root>{payload}</root>")
        except ET.ParseError as exc:
            raise BackendGenerationError("Malformed tool call output") from exc
        name_element = root.find("name")
        arguments_element = root.find("arguments")
        if name_element is None or arguments_element is None:
            raise BackendGenerationError("Incomplete tool call output") from json_error
        raw_arguments = (arguments_element.text or "").strip()
        if raw_arguments:
            try:
                arguments = json.loads(raw_arguments)
            except json.JSONDecodeError:
                arguments = _xml_value(arguments_element)
        else:
            arguments = _xml_value(arguments_element)
        call = {"name": (name_element.text or "").strip(), "arguments": arguments}
    if (
        not isinstance(call, dict)
        or not isinstance(call.get("name"), str)
        or not call["name"]
        or not isinstance(call.get("arguments"), dict)
    ):
        raise BackendGenerationError(
            "Tool call must contain a name and object arguments"
        )
    return call


def _parse_generated_response(
    generated: str,
    tool_inventory: dict[str, dict],
) -> ChatResult:
    open_count = generated.count("<tool_call>")
    close_count = generated.count("</tool_call>")
    if (
        open_count != close_count
        or ("<tool_call" in generated and open_count == 0)
        or ("</tool_call" in generated and close_count == 0)
    ):
        raise BackendGenerationError("Incomplete or malformed tool call output")
    if not open_count:
        return ChatResult(generated.strip(), [])

    calls = []
    for match in re.finditer(r"<tool_call>(.*?)</tool_call>", generated, re.DOTALL):
        parsed = _parse_tool_payload(match.group(1).strip())
        schema = tool_inventory.get(parsed["name"])
        if schema is None:
            raise BackendGenerationError("Generated tool name is not in this request")
        try:
            Draft202012Validator(schema).validate(parsed["arguments"])
        except ValidationError as exc:
            raise BackendGenerationError(
                "Generated tool arguments fail schema validation"
            ) from exc
        calls.append({
            "id": f"call_{uuid.uuid4().hex}",
            "name": parsed["name"],
            "arguments": parsed["arguments"],
        })
    remaining = re.sub(r"<tool_call>.*?</tool_call>", "", generated, flags=re.DOTALL)
    return ChatResult(remaining.strip(), calls)


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

async def _stream_openai(result: ChatResult, model: str) -> AsyncIterator[str]:
    """Frame a completed native generation as OpenAI SSE."""
    yield to_openai_chunk("", model, is_role_header=True)
    if result.content:
        yield to_openai_chunk(result.content, model)
    for index, call in enumerate(result.tool_calls):
        yield to_openai_chunk(
            "",
            model,
            tool_calls=[_openai_tool_call(call, index)],
        )
    yield to_openai_chunk(
        "",
        model,
        finish_reason="tool_calls" if result.tool_calls else "stop",
    )
    yield "data: [DONE]\n\n"


async def _stream_ollama(result: ChatResult, model: str) -> AsyncIterator[str]:
    """Frame a completed native generation as Ollama NDJSON."""
    yield json.dumps(_ollama_full_response(result, model)) + "\n"


# --------------------------------------------------------------------------- #
# OpenAI-compatible endpoints
# --------------------------------------------------------------------------- #

@app.post("/chat/completions")
@app.post("/v1/chat/completions")
@app.post("/api/chat/completions")
async def chat_completions(request: Request) -> Any:
    """Serve OpenAI chat completions, defaulting requests to non-streaming."""
    try:
        chat_request, result = await _run_chat_request(
            await request.json(),
            default_stream=False,
        )
        if chat_request.stream:
            return StreamingResponse(
                _stream_openai(result, chat_request.public_model_id),
                media_type="text/event-stream",
            )
        return _openai_full_response(result, chat_request.public_model_id)
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
        chat_request, result = await _run_chat_request(
            await request.json(),
            default_stream=True,
        )
        if chat_request.stream:
            return StreamingResponse(
                _stream_ollama(result, chat_request.public_model_id),
                media_type="application/x-ndjson",
            )
        return _ollama_full_response(result, chat_request.public_model_id)
    except HTTPException:
        raise
    except (BackendBusyError, BackendUnavailableError, BackendTimeoutError,
            BackendGenerationError) as exc:
        return _backend_error_response(exc)
