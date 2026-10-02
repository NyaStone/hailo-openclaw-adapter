"""Public API tests for model discovery and direct inference behavior."""

from __future__ import annotations

import asyncio
import json
import threading
from typing import Any

import httpx
import pytest

from hailo_ollama_adapter import adapter
from hailo_ollama_adapter.backend import (
    BackendBusyError,
    BackendGenerationError,
    BackendTimeoutError,
    BackendUnavailableError,
    NativeHailoBackend,
)


class FakeInferenceBackend:
    """Record public API inference requests without native Hailo imports."""

    def __init__(
        self,
        response: str = "Hailo says hello.",
        error: Exception | None = None,
        started: Any = None,
        release: Any = None,
    ) -> None:
        self.response = response
        self.error = error
        self.started = started
        self.release = release
        self.status = {"status": "ready", "ready": True, "error": None, "queued": 0}
        self.requests: list[dict[str, Any]] = []
        self.started_models: list[dict[str, Any]] | None = None
        self.close_calls = 0

    async def start(self, models: list[dict[str, Any]]) -> None:
        self.started_models = models

    async def close(self) -> None:
        self.close_calls += 1

    async def generate(
        self,
        hef_path: str,
        messages: list[dict[str, Any]],
        generation: dict[str, Any],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str = "auto",
    ) -> str:
        self.requests.append({
            "hef_path": hef_path,
            "messages": messages,
            "generation": generation,
            "tools": tools,
            "tool_choice": tool_choice,
        })
        if self.started is not None:
            self.started.set()
        if self.release is not None:
            await self.release.wait()
        if self.error is not None:
            raise self.error
        return self.response


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path",
    ["/api/chat", "/v1/chat/completions"],
)
async def test_public_routes_generate_text_with_mapped_hef(
    path: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    hef_path = tmp_path / "Qwen2.5-Coder-1.5B-Instruct.hef"
    hef_path.touch()
    monkeypatch.setenv("HAILO_MODELS", json.dumps({"public-id": str(hef_path)}))
    backend = FakeInferenceBackend()
    monkeypatch.setattr(adapter.app.state, "inference_backend", backend, raising=False)
    transport = httpx.ASGITransport(app=adapter.app)

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://adapter",
    ) as client:
        response = await client.post(
            path,
            json={
                "model": "public-id",
                "messages": [{"role": "user", "content": "Say hello."}],
                "stream": False,
            },
        )

    assert response.status_code == 200
    payload = response.json()
    if path == "/api/chat":
        assert payload["message"] == {
            "role": "assistant",
            "content": "Hailo says hello.",
        }
    else:
        assert payload["choices"] == [{
            "index": 0,
            "message": {"role": "assistant", "content": "Hailo says hello."},
            "finish_reason": "stop",
        }]
    assert backend.requests == [{
        "hef_path": str(hef_path),
        "messages": [{"role": "user", "content": "Say hello."}],
        "generation": {},
        "tools": None,
        "tool_choice": "auto",
    }]


@pytest.mark.asyncio
async def test_explicit_tool_choice_reaches_inference_backend(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    hef_path = tmp_path / "Qwen2.5-Coder-1.5B-Instruct.hef"
    hef_path.touch()
    monkeypatch.setenv("HAILO_MODELS", json.dumps({"public-id": str(hef_path)}))
    backend = FakeInferenceBackend()
    monkeypatch.setattr(adapter.app.state, "inference_backend", backend, raising=False)
    transport = httpx.ASGITransport(app=adapter.app)

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://adapter",
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "public-id",
                "messages": [{"role": "user", "content": "Do not use tools."}],
                "tool_choice": "none",
                "stream": False,
            },
        )

    assert response.status_code == 200
    assert backend.requests[0]["tool_choice"] == "none"


@pytest.mark.asyncio
async def test_discovery_lists_only_configured_usable_hefs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    hef_path = tmp_path / "Qwen2.5-Coder-1.5B-Instruct.hef"
    hef_path.touch()
    monkeypatch.setenv("HAILO_MODELS", json.dumps({"coder:1.5b": str(hef_path)}))
    transport = httpx.ASGITransport(app=adapter.app)

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://adapter",
    ) as client:
        tags = await client.get("/api/tags")
        details = await client.post("/api/show", json={"model": "coder:1.5b"})

    assert tags.status_code == 200
    assert details.status_code == 200
    assert [model["name"] for model in tags.json()["models"]] == ["coder:1.5b"]
    assert details.json()["details"]["format"] == "hef"
    assert details.json()["model_info"]["qwen2.context_length"] == 2048
    assert details.json()["capabilities"] == ["completion", "tools"]


@pytest.mark.asyncio
async def test_discovery_omits_unusable_hefs_without_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    missing_hef = tmp_path / "Qwen2.5-Coder-1.5B-Instruct.hef"
    monkeypatch.setenv("HAILO_MODELS", json.dumps({"coder:1.5b": str(missing_hef)}))
    transport = httpx.ASGITransport(app=adapter.app)

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://adapter",
    ) as client:
        response = await client.get("/api/tags")

    assert response.json() == {"models": []}


@pytest.mark.asyncio
async def test_unknown_model_details_return_not_found(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    hef_path = tmp_path / "Qwen2.5-Coder-1.5B-Instruct.hef"
    hef_path.touch()
    monkeypatch.setenv("HAILO_MODELS", json.dumps({"coder:1.5b": str(hef_path)}))
    transport = httpx.ASGITransport(app=adapter.app)

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://adapter",
    ) as client:
        response = await client.post("/api/show", json={"model": "unknown"})

    assert response.status_code == 404


def test_chat_translation_selects_the_configured_local_hef(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    hef_path = tmp_path / "Qwen2.5-Coder-1.5B-Instruct.hef"
    hef_path.touch()
    monkeypatch.setenv("HAILO_MODELS", json.dumps({"public-id": str(hef_path)}))

    (
        model,
        messages,
        generation,
        tools,
        tool_choice,
        is_stream,
        response_model,
    ) = adapter._build_inference_request(
        {
            "model": "public-id",
            "messages": [{"role": "user", "content": "hello"}],
        },
        default_stream=False,
    )

    assert model["hef_path"] == str(hef_path)
    assert messages == [{"role": "user", "content": "hello"}]
    assert generation == {}
    assert tools is None
    assert tool_choice == "auto"
    assert is_stream is False
    assert response_model == "public-id"


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/api/chat", "/v1/chat/completions"])
async def test_unknown_chat_model_is_rejected_before_backend_call(
    path: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    hef_path = tmp_path / "Qwen2.5-Coder-1.5B-Instruct.hef"
    hef_path.touch()
    monkeypatch.setenv("HAILO_MODELS", json.dumps({"coder:1.5b": str(hef_path)}))

    backend = FakeInferenceBackend()
    monkeypatch.setattr(adapter.app.state, "inference_backend", backend, raising=False)
    transport = httpx.ASGITransport(app=adapter.app, raise_app_exceptions=False)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://adapter",
    ) as client:
        response = await client.post(
            path,
            json={"model": "unknown", "messages": [{"role": "user", "content": "hi"}]},
        )

    assert response.status_code == 404
    assert backend.requests == []


@pytest.mark.asyncio
async def test_discovery_remains_responsive_during_generation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    hef_path = tmp_path / "Qwen2.5-Coder-1.5B-Instruct.hef"
    hef_path.touch()
    monkeypatch.setenv("HAILO_MODELS", json.dumps({"coder:1.5b": str(hef_path)}))
    started = asyncio.Event()
    release = asyncio.Event()
    backend = FakeInferenceBackend(started=started, release=release)
    monkeypatch.setattr(adapter.app.state, "inference_backend", backend, raising=False)
    transport = httpx.ASGITransport(app=adapter.app)

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://adapter",
        timeout=1,
    ) as client:
        request_task = asyncio.create_task(client.post(
            "/api/chat",
            json={
                "model": "coder:1.5b",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": False,
            },
        ))
        await started.wait()
        try:
            response = await client.get("/api/tags")
        finally:
            release.set()
            await request_task

    assert response.status_code == 200
    assert response.json()["models"][0]["name"] == "coder:1.5b"


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/api/chat", "/v1/chat/completions"])
@pytest.mark.parametrize(
    ("error", "status_code"),
    [
        (BackendBusyError("queue full"), 503),
        (BackendUnavailableError("backend failed"), 503),
        (BackendTimeoutError("native generation continues"), 504),
        (BackendGenerationError("native generation failed"), 502),
    ],
)
async def test_chat_endpoint_reports_backend_state(
    error: Exception,
    status_code: int,
    path: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    hef_path = tmp_path / "Qwen2.5-Coder-1.5B-Instruct.hef"
    hef_path.touch()
    monkeypatch.setenv("HAILO_MODELS", json.dumps({"configured": str(hef_path)}))
    backend = FakeInferenceBackend(error=error)
    monkeypatch.setattr(adapter.app.state, "inference_backend", backend, raising=False)
    transport = httpx.ASGITransport(app=adapter.app, raise_app_exceptions=False)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://adapter",
    ) as client:
        response = await client.post(
            path,
            json={
                "model": "configured",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": False,
            },
        )

    assert response.status_code == status_code
    assert response.json() == {"error": str(error)}


@pytest.mark.asyncio
async def test_readiness_reflects_backend_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = FakeInferenceBackend()
    monkeypatch.setattr(adapter.app.state, "inference_backend", backend, raising=False)
    transport = httpx.ASGITransport(app=adapter.app)

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://adapter",
    ) as client:
        ready = await client.get("/readyz")
        backend.status = {
            "status": "failed",
            "ready": False,
            "error": "device unavailable",
            "queued": 0,
        }
        failed = await client.get("/readyz")

    assert ready.status_code == 200
    assert ready.json()["status"] == "ready"
    assert failed.status_code == 503
    assert failed.json()["status"] == "failed"


@pytest.mark.asyncio
async def test_application_lifespan_owns_backend_resources(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    hef_path = tmp_path / "Qwen2.5-Coder-1.5B-Instruct.hef"
    hef_path.touch()
    monkeypatch.setenv("HAILO_MODELS", json.dumps({"configured": str(hef_path)}))
    backend = FakeInferenceBackend()
    monkeypatch.setattr(adapter.app.state, "inference_backend", backend, raising=False)

    async with adapter.app.router.lifespan_context(adapter.app):
        assert backend.started_models is not None
        assert backend.started_models[0]["hef_path"] == str(hef_path)

    assert backend.close_calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/api/chat", "/v1/chat/completions"])
async def test_public_routes_preserve_conversation_for_injected_backend(
    path: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    hef_path = tmp_path / "Qwen2.5-Coder-1.5B-Instruct.hef"
    hef_path.touch()
    monkeypatch.setenv("HAILO_MODELS", json.dumps({"configured": str(hef_path)}))
    backend = FakeInferenceBackend(response="Tool result received.")
    monkeypatch.setattr(adapter.app.state, "inference_backend", backend, raising=False)
    tool_schema = {
        "type": "function",
        "function": {
            "name": "lookup_weather",
            "description": "Look up weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
    messages = [
        {"role": "system", "content": "Follow these rules.\nKeep both lines."},
        {"role": "user", "content": "What is the weather?\nUse the tool."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "call-weather-1",
                "type": "function",
                "function": {
                    "name": "lookup_weather",
                    "arguments": {"city": "Testville"},
                },
            }],
        },
        {
            "role": "tool",
            "tool_call_id": "call-weather-1",
            "name": "lookup_weather",
            "content": "{\"temperature_c\":17,\n\"condition\":\"light rain\"}",
        },
        *[
            message
            for turn in range(4)
            for message in (
                {
                    "role": "user",
                    "content": (
                        "x" * 2101 if turn == 0 else f"Prior user message {turn}"
                    ),
                },
                {"role": "assistant", "content": f"Prior answer {turn}"},
            )
        ],
        {"role": "user", "content": "Summarize the result.\nBe concise."},
    ]
    request_data: dict[str, Any] = {
        "model": "configured",
        "messages": messages,
        "tools": [tool_schema],
        "tool_choice": "auto",
        "stream": False,
    }
    if path == "/api/chat":
        request_data["options"] = {
            "temperature": 0.2,
            "top_p": 0.8,
            "num_predict": 64,
        }
    else:
        request_data.update({"temperature": 0.2, "top_p": 0.8, "max_tokens": 64})

    transport = httpx.ASGITransport(app=adapter.app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://adapter",
    ) as client:
        response = await client.post(path, json=request_data)

    assert response.status_code == 200
    assert len(backend.requests) == 1
    assert backend.requests[0]["messages"] == messages
    assert backend.requests[0]["tools"] == [tool_schema]
    assert backend.requests[0]["tool_choice"] == "auto"
    assert backend.requests[0]["generation"] == {
        "temperature": 0.2,
        "top_p": 0.8,
        "max_generated_tokens": 64,
    }


@pytest.mark.asyncio
async def test_cancelled_waiter_does_not_release_native_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = NativeHailoBackend(queue_size=1, request_timeout=10)
    started = threading.Event()
    release = threading.Event()
    lock = threading.Lock()
    active_calls = 0
    maximum_active_calls = 0

    def initialize(_model_paths: list[str]) -> None:
        return None

    def generate(_job: Any) -> str:
        nonlocal active_calls, maximum_active_calls
        with lock:
            active_calls += 1
            maximum_active_calls = max(maximum_active_calls, active_calls)
        started.set()
        release.wait(timeout=5)
        with lock:
            active_calls -= 1
        return "complete"

    monkeypatch.setattr(backend, "_initialize", initialize)
    monkeypatch.setattr(backend, "_generate_native", generate)
    await backend.start([{"hef_path": "fake.hef"}])

    first = asyncio.create_task(backend.generate("fake.hef", [], {}))
    try:
        assert await asyncio.to_thread(started.wait, 1)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first

        second = asyncio.create_task(backend.generate("fake.hef", [], {}))
        await asyncio.sleep(0)
        with pytest.raises(BackendBusyError):
            await backend.generate("fake.hef", [], {})
        release.set()
        assert await second == "complete"
        assert maximum_active_calls == 1
    finally:
        release.set()
        await backend.close()
