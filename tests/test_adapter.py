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
    BackendContextOverflowError,
    BackendGenerationError,
    BackendOutputExhaustedError,
    BackendTimeoutError,
    BackendUnavailableError,
    NativeHailoBackend,
    _clean_generated_text,
)


class FakeInferenceBackend:
    """Record public API inference requests without native Hailo imports."""

    def __init__(
        self,
        response: str | list[str] = "Hailo says hello.",
        error: Exception | None = None,
        started: Any = None,
        release: Any = None,
    ) -> None:
        self.responses = response if isinstance(response, list) else [response]
        self.response_index = 0
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
        response = self.responses[min(self.response_index, len(self.responses) - 1)]
        self.response_index += 1
        return response


class FakeNativeCompletion:
    def __init__(self, response: str, status_name: str) -> None:
        self.response = response
        self.generation_status = type("Status", (), {"name": status_name})()

    def __enter__(self) -> FakeNativeCompletion:
        return self

    def __exit__(self, *_: Any) -> None:
        return None

    def read_all(self, timeout_ms: int) -> str:
        return self.response


class FakeNativeLLM:
    def __init__(
        self,
        capacity: int = 10000,
        status_name: str = "LOGICAL_END_OF_GENERATION",
    ) -> None:
        self.capacity = capacity
        self.status_name = status_name
        self.tokenized_prompt = ""
        self.generate_calls: list[dict[str, Any]] = []

    def clear_context(self) -> None:
        return None

    def prompt_template(self) -> str:
        return "{{ messages }}|TOOLS={{ tools }}|ASSISTANT"

    def tokenize(self, text: str) -> list[str]:
        self.tokenized_prompt = text
        return text.split()

    def max_context_capacity(self) -> int:
        return self.capacity

    def generate(
        self,
        prompt: list[dict[str, Any]],
        **arguments: Any,
    ) -> FakeNativeCompletion:
        self.generate_calls.append({"prompt": prompt, "arguments": arguments})
        return FakeNativeCompletion("Native response.", self.status_name)

    def release(self) -> None:
        return None


async def _start_fake_native_backend(
    monkeypatch: pytest.MonkeyPatch,
    hef_path: str,
    native_llm: FakeNativeLLM,
) -> NativeHailoBackend:
    def initialize(backend: NativeHailoBackend, model_paths: list[str]) -> None:
        backend._models = dict.fromkeys(model_paths, native_llm)
        backend._model_context_lengths = dict.fromkeys(
            model_paths,
            native_llm.max_context_capacity(),
        )

    monkeypatch.setattr(NativeHailoBackend, "_initialize", initialize)
    backend = NativeHailoBackend()
    await backend.start([{"hef_path": hef_path}])
    monkeypatch.setattr(adapter.app.state, "inference_backend", backend, raising=False)
    return backend


def test_native_response_removes_qwen_end_marker() -> None:
    assert _clean_generated_text("A short answer.<|im_end|>") == "A short answer."


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
@pytest.mark.parametrize("path", ["/api/chat", "/v1/chat/completions"])
async def test_streaming_routes_terminate_with_protocol_marker(
    path: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    hef_path = tmp_path / "Qwen2.5-Coder-1.5B-Instruct.hef"
    hef_path.touch()
    monkeypatch.setenv("HAILO_MODELS", json.dumps({"public-id": str(hef_path)}))
    backend = FakeInferenceBackend(response="Streamed answer.")
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
                "messages": [{"role": "user", "content": "Answer."}],
                "stream": True,
            },
        )

    assert response.status_code == 200
    if path == "/api/chat":
        records = [json.loads(line) for line in response.text.splitlines()]
        assert records[-1]["done"] is True
        assert records[-1]["message"]["content"] == "Streamed answer."
    else:
        events = [
            line[6:]
            for line in response.text.splitlines()
            if line.startswith("data: ")
        ]
        assert events[-1] == "[DONE]"
        chunks = [json.loads(event) for event in events[:-1]]
        content = "".join(
            chunk["choices"][0]["delta"].get("content", "") for chunk in chunks
        )
        assert content == "Streamed answer."
        assert chunks[-1]["choices"][0]["finish_reason"] == "stop"


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/api/chat", "/v1/chat/completions"])
async def test_streaming_routes_emit_text_and_each_tool_call_once(
    path: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    hef_path = tmp_path / "Qwen2.5-Coder-1.5B-Instruct.hef"
    hef_path.touch()
    monkeypatch.setenv("HAILO_MODELS", json.dumps({"configured": str(hef_path)}))
    backend = FakeInferenceBackend(
        response=(
            'Checking now. <tool_call>{"name":"lookup_weather",'
            '"arguments":{"city":"Testville"}}</tool_call>'
            '<tool_call>{"name":"lookup_time",'
            '"arguments":{"timezone":"UTC"}}</tool_call>'
        ),
    )
    monkeypatch.setattr(adapter.app.state, "inference_backend", backend, raising=False)
    tools = [
        {
            "type": "function",
            "function": {
                "name": name,
                "parameters": {
                    "type": "object",
                    "properties": {argument: {"type": "string"}},
                    "required": [argument],
                    "additionalProperties": False,
                },
            },
        }
        for name, argument in (
            ("lookup_weather", "city"),
            ("lookup_time", "timezone"),
        )
    ]
    transport = httpx.ASGITransport(app=adapter.app)

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://adapter",
    ) as client:
        response = await client.post(
            path,
            json={
                "model": "configured",
                "messages": [{"role": "user", "content": "Check weather and time."}],
                "tools": tools,
                "stream": True,
            },
        )

    assert response.status_code == 200
    if path == "/api/chat":
        records = [json.loads(line) for line in response.text.splitlines()]
        assert records[-1]["done"] is True
        assert records[-1]["message"]["content"] == "Checking now."
        calls = records[-1]["message"]["tool_calls"]
        assert [call["function"] for call in calls] == [
            {"name": "lookup_weather", "arguments": {"city": "Testville"}},
            {"name": "lookup_time", "arguments": {"timezone": "UTC"}},
        ]
    else:
        events = [
            line[6:]
            for line in response.text.splitlines()
            if line.startswith("data: ")
        ]
        assert events[-1] == "[DONE]"
        chunks = [json.loads(event) for event in events[:-1]]
        assert "".join(
            chunk["choices"][0]["delta"].get("content", "") for chunk in chunks
        ) == "Checking now."
        call_deltas = [
            delta
            for chunk in chunks
            for delta in chunk["choices"][0]["delta"].get("tool_calls", [])
        ]
        assert [call["index"] for call in call_deltas] == [0, 1]
        assert len({call["id"] for call in call_deltas}) == 2
        assert all(call["id"].startswith("call_") for call in call_deltas)
        assert [
            {
                "name": call["function"]["name"],
                "arguments": json.loads(call["function"]["arguments"]),
            }
            for call in call_deltas
        ] == [
            {"name": "lookup_weather", "arguments": {"city": "Testville"}},
            {"name": "lookup_time", "arguments": {"timezone": "UTC"}},
        ]
        assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/api/chat", "/v1/chat/completions"])
async def test_streaming_routes_emit_a_single_tool_call(
    path: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    hef_path = tmp_path / "Qwen2.5-Coder-1.5B-Instruct.hef"
    hef_path.touch()
    monkeypatch.setenv("HAILO_MODELS", json.dumps({"configured": str(hef_path)}))
    backend = FakeInferenceBackend(
        response=(
            '<tool_call>{"name":"lookup_weather",'
            '"arguments":{"city":"Testville"}}</tool_call>'
        ),
    )
    monkeypatch.setattr(adapter.app.state, "inference_backend", backend, raising=False)
    transport = httpx.ASGITransport(app=adapter.app)

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://adapter",
    ) as client:
        response = await client.post(
            path,
            json={
                "model": "configured",
                "messages": [{"role": "user", "content": "Check the weather."}],
                "tools": [{
                    "type": "function",
                    "function": {
                        "name": "lookup_weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    },
                }],
                "stream": True,
            },
        )

    assert response.status_code == 200
    if path == "/api/chat":
        records = [json.loads(line) for line in response.text.splitlines()]
        assert records[-1]["done"] is True
        calls = records[-1]["message"]["tool_calls"]
        assert len(calls) == 1
        assert calls[0]["function"] == {
            "name": "lookup_weather",
            "arguments": {"city": "Testville"},
        }
    else:
        events = [
            line[6:]
            for line in response.text.splitlines()
            if line.startswith("data: ")
        ]
        chunks = [json.loads(event) for event in events[:-1]]
        calls = [
            call
            for chunk in chunks
            for call in chunk["choices"][0]["delta"].get("tool_calls", [])
        ]
        assert events[-1] == "[DONE]"
        assert len(calls) == 1
        assert calls[0]["index"] == 0
        assert calls[0]["type"] == "function"
        assert calls[0]["function"]["name"] == "lookup_weather"
        assert json.loads(calls[0]["function"]["arguments"]) == {
            "city": "Testville"
        }
        assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/api/chat", "/v1/chat/completions"])
async def test_streaming_generation_failure_has_no_success_terminal_marker(
    path: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    hef_path = tmp_path / "Qwen2.5-Coder-1.5B-Instruct.hef"
    hef_path.touch()
    monkeypatch.setenv("HAILO_MODELS", json.dumps({"configured": str(hef_path)}))
    error = BackendGenerationError("native generation failed")
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
                "messages": [{"role": "user", "content": "Answer."}],
                "stream": True,
            },
        )

    assert response.status_code == 502
    assert response.json() == {"error": str(error)}
    assert response.headers["content-type"].startswith("application/json")
    assert "[DONE]" not in response.text
    assert '"done": true' not in response.text


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/api/chat", "/v1/chat/completions"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    ("tool_choice", "expected_name"),
    [
        ("required", "lookup_weather"),
        ({"type": "function", "function": {"name": "lookup_time"}}, "lookup_time"),
    ],
)
async def test_required_and_forced_tool_choices_return_matching_calls(
    path: str,
    stream: bool,
    tool_choice: Any,
    expected_name: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    hef_path = tmp_path / "Qwen2.5-Coder-1.5B-Instruct.hef"
    hef_path.touch()
    monkeypatch.setenv("HAILO_MODELS", json.dumps({"configured": str(hef_path)}))
    backend = FakeInferenceBackend(
        response=(
            f'<tool_call>{{"name":"{expected_name}","arguments":{{}}}}</tool_call>'
        ),
    )
    monkeypatch.setattr(adapter.app.state, "inference_backend", backend, raising=False)
    tools = [
        {
            "type": "function",
            "function": {"name": name, "parameters": {"type": "object"}},
        }
        for name in ("lookup_weather", "lookup_time")
    ]
    transport = httpx.ASGITransport(app=adapter.app)

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://adapter",
    ) as client:
        response = await client.post(
            path,
            json={
                "model": "configured",
                "messages": [{"role": "user", "content": "Look it up."}],
                "tools": tools,
                "tool_choice": tool_choice,
                "stream": stream,
            },
        )

    assert response.status_code == 200
    assert backend.requests[0]["tools"] == tools
    if path == "/api/chat":
        call = response.json()["message"]["tool_calls"][0]
        assert call["function"]["name"] == expected_name
    elif stream:
        data_events = [
            line[6:]
            for line in response.text.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"
        ]
        chunks = [json.loads(event) for event in data_events]
        calls = [
            call
            for chunk in chunks
            for call in chunk["choices"][0]["delta"].get("tool_calls", [])
        ]
        assert calls[0]["function"]["name"] == expected_name
        assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"
    else:
        assert response.json()["choices"][0]["finish_reason"] == "tool_calls"
        call = response.json()["choices"][0]["message"]["tool_calls"][0]
        assert call["function"]["name"] == expected_name


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

    assert tags.status_code == 200
    assert [model["name"] for model in tags.json()["models"]] == ["coder:1.5b"]
    assert tags.json()["models"][0]["details"] == {"format": "hef"}


def test_discovery_uses_installed_hailo_apps_agent_models(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    hef_path = tmp_path / "Catalog-Model.hef"
    hef_path.touch()
    monkeypatch.delenv("HAILO_MODELS", raising=False)
    monkeypatch.setattr(
        adapter,
        "_hailo_apps_model_catalog",
        lambda: [{"name": "Catalog-Model", "hef_path": str(hef_path)}],
    )

    models = adapter._configured_models()

    assert models == [{
        "name": "Catalog-Model",
        "model": "Catalog-Model",
        "modified_at": models[0]["modified_at"],
        "size": 0,
        "digest": "",
        "details": {"format": "hef"},
        "model_info": {},
        "capabilities": ["completion", "tools"],
        "hef_path": str(hef_path),
    }]


def test_discovery_errors_when_no_installed_hef_is_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HAILO_MODELS", raising=False)
    monkeypatch.setattr(adapter, "_hailo_apps_model_catalog", lambda: [])

    with pytest.raises(RuntimeError, match="No installed LLM HEFs"):
        adapter._configured_models()


@pytest.mark.asyncio
async def test_api_show_reports_loaded_hef_context_capacity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    hef_path = tmp_path / "Catalog-Model.hef"
    hef_path.touch()
    monkeypatch.delenv("HAILO_MODELS", raising=False)
    monkeypatch.setattr(
        adapter,
        "_hailo_apps_model_catalog",
        lambda: [{"name": "Catalog-Model", "hef_path": str(hef_path)}],
    )

    def initialize(backend: NativeHailoBackend, model_paths: list[str]) -> None:
        native_llm = FakeNativeLLM(capacity=4096)
        backend._models = dict.fromkeys(model_paths, native_llm)
        backend._model_context_lengths = dict.fromkeys(model_paths, 4096)

    monkeypatch.setattr(NativeHailoBackend, "_initialize", initialize)
    monkeypatch.setattr(adapter.app.state, "inference_backend", None, raising=False)
    transport = httpx.ASGITransport(app=adapter.app)

    async with adapter._lifespan(adapter.app):
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://adapter",
        ) as client:
            response = await client.post(
                "/api/show",
                json={"model": "Catalog-Model"},
            )

    assert response.status_code == 200
    assert response.json()["model_info"] == {"hailo.context_length": 4096}


def test_discovery_errors_when_configured_hef_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    missing_hef = tmp_path / "Qwen2.5-Coder-1.5B-Instruct.hef"
    monkeypatch.setenv("HAILO_MODELS", json.dumps({"coder:1.5b": str(missing_hef)}))

    with pytest.raises(RuntimeError, match="No installed LLM HEFs"):
        adapter._configured_models()


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

    request = adapter._build_inference_request(
        {
            "model": "public-id",
            "messages": [{"role": "user", "content": "hello"}],
        },
        default_stream=False,
    )

    assert request.hef_path == str(hef_path)
    assert request.messages == [{"role": "user", "content": "hello"}]
    assert request.generation == {}
    assert request.tools is None
    assert request.tool_choice == "auto"
    assert request.stream is False
    assert request.public_model_id == "public-id"


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
@pytest.mark.parametrize("path", ["/api/chat", "/v1/chat/completions"])
async def test_public_routes_return_validated_tool_calls(
    path: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    hef_path = tmp_path / "Qwen2.5-Coder-1.5B-Instruct.hef"
    hef_path.touch()
    monkeypatch.setenv("HAILO_MODELS", json.dumps({"configured": str(hef_path)}))
    backend = FakeInferenceBackend(
        response=(
            'Checking now. <tool_call>{"name":"lookup_weather",'
            '"arguments":{"city":"Testville"}}</tool_call>'
            "<tool_call><name>lookup_time</name><arguments>"
            "<timezone>UTC</timezone></arguments></tool_call>"
        ),
    )
    monkeypatch.setattr(adapter.app.state, "inference_backend", backend, raising=False)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup_weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "lookup_time",
                "parameters": {
                    "type": "object",
                    "properties": {"timezone": {"type": "string"}},
                    "required": ["timezone"],
                    "additionalProperties": False,
                },
            },
        },
    ]
    transport = httpx.ASGITransport(app=adapter.app)

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://adapter",
    ) as client:
        response = await client.post(
            path,
            json={
                "model": "configured",
                "messages": [{"role": "user", "content": "Check weather and time."}],
                "tools": tools,
                "stream": False,
            },
        )

    assert response.status_code == 200
    payload = response.json()
    if path == "/api/chat":
        calls = payload["message"]["tool_calls"]
        assert payload["message"]["content"] == "Checking now."
        assert [call["function"] for call in calls] == [
            {"name": "lookup_weather", "arguments": {"city": "Testville"}},
            {"name": "lookup_time", "arguments": {"timezone": "UTC"}},
        ]
    else:
        choice = payload["choices"][0]
        calls = choice["message"]["tool_calls"]
        assert choice["message"]["content"] == "Checking now."
        assert choice["finish_reason"] == "tool_calls"
        assert [
            {"name": call["function"]["name"],
             "arguments": json.loads(call["function"]["arguments"])}
            for call in calls
        ] == [
            {"name": "lookup_weather", "arguments": {"city": "Testville"}},
            {"name": "lookup_time", "arguments": {"timezone": "UTC"}},
        ]
    assert len({call["id"] for call in calls}) == 2
    assert all(call["id"].startswith("call_") for call in calls)
    assert backend.requests[0]["tools"] == tools


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/api/chat", "/v1/chat/completions"])
@pytest.mark.parametrize(
    "messages",
    [
        [{
            "role": "user",
            "content": [{
                "type": "image_url",
                "image_url": {"url": "https://example.test/image.png"},
            }],
        }],
        [{"role": "user", "content": [{"type": "audio", "data": "AAAA"}]}],
        [{"role": "user", "content": "Describe this.", "images": ["AAAA"]}],
        [{
            "role": "user",
            "content": {
                "type": "image_url",
                "url": "https://example.test/image.png",
            },
        }],
    ],
)
async def test_text_only_models_reject_unsupported_media(
    path: str,
    messages: list[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    hef_path = tmp_path / "Qwen2.5-Coder-1.5B-Instruct.hef"
    hef_path.touch()
    monkeypatch.setenv("HAILO_MODELS", json.dumps({"configured": str(hef_path)}))
    backend = FakeInferenceBackend()
    monkeypatch.setattr(adapter.app.state, "inference_backend", backend, raising=False)
    transport = httpx.ASGITransport(app=adapter.app)

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://adapter",
    ) as client:
        response = await client.post(
            path,
            json={"model": "configured", "messages": messages},
        )

    assert response.status_code == 400
    assert "media" in response.json()["detail"].lower()
    assert backend.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/api/chat", "/v1/chat/completions"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    ("field", "value"),
    [("presence_penalty", 0.5), ("response_format", {"type": "json_object"})],
)
async def test_unsupported_generation_options_are_rejected(
    path: str,
    stream: bool,
    field: str,
    value: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    hef_path = tmp_path / "Qwen2.5-Coder-1.5B-Instruct.hef"
    hef_path.touch()
    monkeypatch.setenv("HAILO_MODELS", json.dumps({"configured": str(hef_path)}))
    backend = FakeInferenceBackend()
    monkeypatch.setattr(adapter.app.state, "inference_backend", backend, raising=False)
    payload: dict[str, Any] = {
        "model": "configured",
        "messages": [{"role": "user", "content": "Answer."}],
        "stream": stream,
    }
    if path == "/api/chat":
        payload["options"] = {field: value}
    else:
        payload[field] = value
    transport = httpx.ASGITransport(app=adapter.app)

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://adapter",
    ) as client:
        response = await client.post(path, json=payload)

    assert response.status_code == 400
    assert field in response.json()["detail"]
    assert backend.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/api/chat", "/v1/chat/completions"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    ("tool_choice", "response_body"),
    [
        ("required", "No tool call was generated for required choice"),
        (
            {"type": "function", "function": {"name": "lookup_time"}},
            "Generated tool call does not match the forced tool",
        ),
    ],
)
async def test_required_and_forced_tool_choices_cannot_succeed_without_matching_calls(
    path: str,
    stream: bool,
    tool_choice: Any,
    response_body: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    hef_path = tmp_path / "Qwen2.5-Coder-1.5B-Instruct.hef"
    hef_path.touch()
    monkeypatch.setenv("HAILO_MODELS", json.dumps({"configured": str(hef_path)}))
    generated = (
        '<tool_call>{"name":"lookup_weather","arguments":{}}</tool_call>'
        if isinstance(tool_choice, dict)
        else "A plain-text answer."
    )
    backend = FakeInferenceBackend(response=generated)
    monkeypatch.setattr(adapter.app.state, "inference_backend", backend, raising=False)
    tools = [
        {
            "type": "function",
            "function": {"name": name, "parameters": {"type": "object"}},
        }
        for name in ("lookup_weather", "lookup_time")
    ]
    transport = httpx.ASGITransport(app=adapter.app, raise_app_exceptions=False)

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://adapter",
    ) as client:
        response = await client.post(
            path,
            json={
                "model": "configured",
                "messages": [{"role": "user", "content": "Check the weather."}],
                "tools": tools,
                "tool_choice": tool_choice,
                "stream": stream,
            },
        )

    assert response.status_code == 502
    assert response.json()["error"] == response_body
    assert response.headers["content-type"].startswith("application/json")
    assert "[DONE]" not in response.text
    assert '"done": true' not in response.text


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/api/chat", "/v1/chat/completions"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (
            BackendContextOverflowError("context overflow"),
            {"error": "context overflow", "code": "context_length_exceeded"},
        ),
        (
            BackendOutputExhaustedError("output exhausted"),
            {"error": "output exhausted", "code": "output_limit_exceeded"},
        ),
    ],
)
async def test_limit_failures_have_matching_meaning_for_all_response_modes(
    path: str,
    stream: bool,
    error: Exception,
    expected: dict[str, str],
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
                "messages": [{"role": "user", "content": "Answer."}],
                "stream": stream,
            },
        )

    expected_status = 413 if isinstance(error, BackendContextOverflowError) else 502
    assert response.status_code == expected_status
    assert response.json() == expected
    assert response.headers["content-type"].startswith("application/json")
    assert "[DONE]" not in response.text
    assert '"done": true' not in response.text


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/api/chat", "/v1/chat/completions"])
async def test_native_context_budget_counts_full_transcript_and_tool_schema(
    path: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    hef_path = tmp_path / "Qwen2.5-Coder-1.5B-Instruct.hef"
    hef_path.touch()
    monkeypatch.setenv("HAILO_MODELS", json.dumps({"configured": str(hef_path)}))
    native_llm = FakeNativeLLM(capacity=1)
    backend = await _start_fake_native_backend(monkeypatch, str(hef_path), native_llm)
    messages = [
        {"role": "system", "content": "Preserve the system instruction."},
        {"role": "user", "content": "Earlier transcript details stay present."},
        {"role": "user", "content": "Final question."},
    ]
    tools = [{
        "type": "function",
        "function": {
            "name": "lookup_weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
        },
    }]
    transport = httpx.ASGITransport(app=adapter.app)

    try:
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://adapter",
        ) as client:
            response = await client.post(
                path,
                json={
                    "model": "configured",
                    "messages": messages,
                    "tools": tools,
                    "max_tokens": 12,
                },
            )
    finally:
        await backend.close()

    assert response.status_code == 413
    detail = response.json()["error"]
    assert "Rendered prompt" in detail
    assert "output allowance (12 tokens)" in detail
    assert "capacity (1)" in detail
    assert "Preserve the system instruction." in native_llm.tokenized_prompt
    assert "Earlier transcript details stay present." in native_llm.tokenized_prompt
    assert "Final question." in native_llm.tokenized_prompt
    assert "lookup_weather" in native_llm.tokenized_prompt
    assert "city" in native_llm.tokenized_prompt
    assert native_llm.generate_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/api/chat", "/v1/chat/completions"])
@pytest.mark.parametrize(
    ("status_name", "expected_status", "expected_code"),
    [
        ("LOGICAL_END_OF_GENERATION", 200, None),
        ("MAX_TOKENS_REACHED", 502, "output_limit_exceeded"),
    ],
)
async def test_native_terminal_status_controls_public_success(
    path: str,
    status_name: str,
    expected_status: int,
    expected_code: str | None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    hef_path = tmp_path / "Qwen2.5-Coder-1.5B-Instruct.hef"
    hef_path.touch()
    monkeypatch.setenv("HAILO_MODELS", json.dumps({"configured": str(hef_path)}))
    native_llm = FakeNativeLLM(status_name=status_name)
    backend = await _start_fake_native_backend(monkeypatch, str(hef_path), native_llm)
    transport = httpx.ASGITransport(app=adapter.app, raise_app_exceptions=False)

    try:
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://adapter",
        ) as client:
            response = await client.post(
                path,
                json={
                    "model": "configured",
                    "messages": [{"role": "user", "content": "Answer."}],
                    "max_tokens": 8,
                },
            )
    finally:
        await backend.close()

    assert response.status_code == expected_status
    if expected_code is None:
        assert response.status_code == 200
    else:
        assert response.json()["code"] == expected_code


@pytest.mark.asyncio
async def test_tool_result_replay_continues_with_fresh_request_context(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    hef_path = tmp_path / "Qwen2.5-Coder-1.5B-Instruct.hef"
    hef_path.touch()
    monkeypatch.setenv("HAILO_MODELS", json.dumps({"configured": str(hef_path)}))
    backend = FakeInferenceBackend(response=[
        '<tool_call>{"name":"lookup_weather",'
        '"arguments":{"city":"Testville"}}</tool_call>',
        "It is 17 degrees and raining in Testville.",
    ])
    monkeypatch.setattr(adapter.app.state, "inference_backend", backend, raising=False)
    tools = [{
        "type": "function",
        "function": {
            "name": "lookup_weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }]
    transport = httpx.ASGITransport(app=adapter.app)

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://adapter",
    ) as client:
        first = await client.post(
            "/api/chat",
            json={
                "model": "configured",
                "messages": [{"role": "user", "content": "What's the weather?"}],
                "tools": tools,
                "stream": False,
            },
        )
        first_call = first.json()["message"]["tool_calls"][0]
        replay_messages = [
            {"role": "user", "content": "What's the weather?"},
            {"role": "assistant", "content": "", "tool_calls": [first_call]},
            {
                "role": "tool",
                "tool_call_id": first_call["id"],
                "name": "lookup_weather",
                "content": '{"temperature_c":17,"condition":"light rain"}',
            },
        ]
        second = await client.post(
            "/api/chat",
            json={
                "model": "configured",
                "messages": replay_messages,
                "tools": tools,
                "stream": False,
            },
        )

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json()["message"]["content"] == (
        "It is 17 degrees and raining in Testville."
    )
    assert backend.requests[1]["messages"] == replay_messages
    assert backend.requests[1]["tools"] == tools


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "generated",
    [
        '<tool_call>{"name":"lookup_weather","arguments":{"city":"X"}}',
        '<tool_call>{"name":"unknown_tool","arguments":{}}</tool_call>',
        '<tool_call>{"name":"lookup_weather","arguments":{}}</tool_call>',
        '<tool_call>{not json}</tool_call>',
    ],
)
async def test_invalid_tool_calls_fail_instead_of_returning_text(
    generated: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    hef_path = tmp_path / "Qwen2.5-Coder-1.5B-Instruct.hef"
    hef_path.touch()
    monkeypatch.setenv("HAILO_MODELS", json.dumps({"configured": str(hef_path)}))
    backend = FakeInferenceBackend(response=generated)
    monkeypatch.setattr(adapter.app.state, "inference_backend", backend, raising=False)
    transport = httpx.ASGITransport(app=adapter.app, raise_app_exceptions=False)

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://adapter",
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "configured",
                "messages": [{"role": "user", "content": "What's the weather?"}],
                "tools": [{
                    "type": "function",
                    "function": {
                        "name": "lookup_weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                            "additionalProperties": False,
                        },
                    },
                }],
                "stream": False,
            },
        )

    assert response.status_code == 502
    assert "tool" in response.json()["error"].lower()


@pytest.mark.asyncio
async def test_tool_inventory_is_isolated_between_requests(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    hef_path = tmp_path / "Qwen2.5-Coder-1.5B-Instruct.hef"
    hef_path.touch()
    monkeypatch.setenv("HAILO_MODELS", json.dumps({"configured": str(hef_path)}))
    backend = FakeInferenceBackend(response=[
        '<tool_call>{"name":"lookup_weather",'
        '"arguments":{"city":"X"}}</tool_call>',
        '<tool_call>{"name":"lookup_weather",'
        '"arguments":{"city":"X"}}</tool_call>',
    ])
    monkeypatch.setattr(adapter.app.state, "inference_backend", backend, raising=False)
    transport = httpx.ASGITransport(app=adapter.app, raise_app_exceptions=False)
    common = {
        "model": "configured",
        "messages": [{"role": "user", "content": "Call a tool."}],
        "stream": False,
    }
    weather_tool = {
        "type": "function",
        "function": {
            "name": "lookup_weather",
            "parameters": {"type": "object", "properties": {
                "city": {"type": "string"},
            }},
        },
    }
    time_tool = {
        "type": "function",
        "function": {
            "name": "lookup_time",
            "parameters": {"type": "object", "properties": {
                "timezone": {"type": "string"},
            }},
        },
    }

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://adapter",
    ) as client:
        first = await client.post(
            "/api/chat",
            json={**common, "tools": [weather_tool]},
        )
        second = await client.post(
            "/api/chat",
            json={**common, "tools": [time_tool]},
        )

    assert first.status_code == 200
    assert second.status_code == 502
    assert backend.requests[0]["tools"] == [weather_tool]
    assert backend.requests[1]["tools"] == [time_tool]


@pytest.mark.asyncio
async def test_cancelled_waiter_does_not_release_native_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = NativeHailoBackend(queue_size=1, request_timeout=0.05)
    started = threading.Event()
    release = threading.Event()
    lock = threading.Lock()
    active_calls = 0
    maximum_active_calls = 0

    def initialize(model_paths: list[str]) -> None:
        backend._model_context_lengths = dict.fromkeys(model_paths, 4096)

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
        with pytest.raises(BackendTimeoutError):
            await second
        assert active_calls == 1
        release.set()
        await backend.close()
        assert maximum_active_calls == 1
    finally:
        release.set()
        if backend.status["status"] != "stopped":
            await backend.close()
