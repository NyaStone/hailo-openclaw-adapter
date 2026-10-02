"""Correctness tests for model discovery and upstream HTTP errors."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from hailo_ollama_adapter import adapter


class StubResponse:
    """Provide the minimal successful response interface used by list probes."""

    def __init__(self, payload: Any, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        return None

    def json(self) -> Any:
        return self._payload


class RecordingAsyncClient:
    """Record model-list URLs and return a configurable native catalog."""

    requested_urls: list[str] = []
    response_payload: Any = {
        "models": [
            {"name": "qwen3:1.7b"},
            {"name": "qwen2.5-coder:1.5b"},
        ]
    }

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    async def __aenter__(self) -> RecordingAsyncClient:
        return self

    async def __aexit__(self, *_args: Any) -> None:
        return None

    async def get(self, url: str, **_kwargs: Any) -> StubResponse:
        type(self).requested_urls.append(url)
        return StubResponse(type(self).response_payload)


class ErrorPostingAsyncClient:
    """Return a native non-success response for chat error propagation tests."""

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    async def __aenter__(self) -> ErrorPostingAsyncClient:
        return self

    async def __aexit__(self, *_args: Any) -> None:
        return None

    async def post(self, url: str, **_kwargs: Any) -> httpx.Response:
        request = httpx.Request("POST", url)
        return httpx.Response(
            404,
            request=request,
            json={"error": "model 'missing' not found"},
        )


def upstream_status_error(
    status_code: int,
    payload: dict[str, str],
) -> httpx.HTTPStatusError:
    """Build a realistic Hailo status exception for endpoint tests."""
    request = httpx.Request("POST", adapter.HAILO_URL)
    response = httpx.Response(status_code, request=request, json=payload)
    return httpx.HTTPStatusError(
        f"upstream returned {status_code}",
        request=request,
        response=response,
    )


@pytest.fixture(autouse=True)
def reset_recording_client() -> None:
    """Clear request history before each test."""
    RecordingAsyncClient.requested_urls.clear()


def test_flatten_newlines_preserves_words_without_literal_line_breaks() -> None:
    assert adapter._flatten_newlines("alpha\r\nbeta\ngamma\rdelta") == (
        "alpha beta gamma delta"
    )


@pytest.mark.asyncio
async def test_refresh_uses_native_api_tags(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(adapter.httpx, "AsyncClient", RecordingAsyncClient)

    models = await adapter._refresh_models_from_hailo()

    assert RecordingAsyncClient.requested_urls == ["http://127.0.0.1:8000/api/tags"]
    assert [model["name"] for model in models] == [
        "qwen3:1.7b",
        "qwen2.5-coder:1.5b",
    ]


@pytest.mark.asyncio
async def test_discovery_lists_only_configured_usable_hefs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    hef_path = tmp_path / "Qwen2.5-Coder-1.5B-Instruct.hef"
    hef_path.touch()
    monkeypatch.setenv("HAILO_MODELS", json.dumps({"coder:1.5b": str(hef_path)}))
    transport = httpx.ASGITransport(app=adapter.app)

    async with httpx.AsyncClient(transport=transport, base_url="http://adapter") as client:
        tags = await client.get("/api/tags")
        details = await client.post("/api/show", json={"model": "coder:1.5b"})

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

    async with httpx.AsyncClient(transport=transport, base_url="http://adapter") as client:
        response = await client.get("/api/tags")

    assert response.json() == {"models": []}


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/api/chat", "/v1/chat/completions"])
async def test_unknown_chat_model_is_rejected_before_upstream_call(
    path: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    hef_path = tmp_path / "Qwen2.5-Coder-1.5B-Instruct.hef"
    hef_path.touch()
    monkeypatch.setenv("HAILO_MODELS", json.dumps({"coder:1.5b": str(hef_path)}))

    async def unexpected_post(_body: bytes) -> dict[str, Any]:
        raise AssertionError("unknown model must be rejected before inference")

    monkeypatch.setattr(adapter, "_post_hailo", unexpected_post)
    transport = httpx.ASGITransport(app=adapter.app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://adapter") as client:
        response = await client.post(
            path,
            json={"model": "unknown", "messages": [{"role": "user", "content": "hi"}]},
        )

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_discovery_does_not_wait_for_inference_slot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    hef_path = tmp_path / "Qwen2.5-Coder-1.5B-Instruct.hef"
    hef_path.touch()
    monkeypatch.setenv("HAILO_MODELS", json.dumps({"coder:1.5b": str(hef_path)}))
    await adapter._hailo_semaphore.acquire()
    transport = httpx.ASGITransport(app=adapter.app)

    try:
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://adapter",
            timeout=1,
        ) as client:
            response = await client.get("/api/tags")
    finally:
        adapter._hailo_semaphore.release()

    assert response.status_code == 200
    assert response.json()["models"][0]["name"] == "coder:1.5b"


@pytest.mark.asyncio
async def test_post_hailo_rejects_non_success_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(adapter.httpx, "AsyncClient", ErrorPostingAsyncClient)

    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        await adapter._post_hailo(b"{}")

    assert exc_info.value.response.status_code == 404


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/v1/chat/completions", "/api/chat"])
async def test_chat_endpoint_preserves_upstream_status(
    path: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fail_post(_body: bytes) -> dict[str, Any]:
        """Simulate a non-streaming upstream status failure."""
        raise upstream_status_error(404, {"error": "model 'missing' not found"})

    monkeypatch.setattr(adapter, "_post_hailo", fail_post)
    transport = httpx.ASGITransport(app=adapter.app, raise_app_exceptions=False)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://adapter",
    ) as client:
        response = await client.post(
            path,
            json={
                "model": "missing",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": False,
            },
        )

    assert response.status_code == 404
    assert response.json() == {"error": "model 'missing' not found"}
