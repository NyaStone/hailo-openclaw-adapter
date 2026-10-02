"""Serialized direct HailoRT inference backend."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


class BackendBusyError(RuntimeError):
    """Raised when the bounded inference queue is full."""


class BackendUnavailableError(RuntimeError):
    """Raised when inference resources are not ready or have failed."""


class BackendTimeoutError(RuntimeError):
    """Raised when a request deadline expires while native work continues."""


class BackendGenerationError(RuntimeError):
    """Raised when native generation fails and the backend becomes unhealthy."""


@dataclass
class _InferenceRequest:
    hef_path: str
    messages: list[dict[str, Any]]
    generation: dict[str, Any]
    tools: list[dict[str, Any]] | None
    tool_choice: str
    result: asyncio.Future[str]


class NativeHailoBackend:
    """Own Hailo resources and execute all native calls on one worker thread."""

    def __init__(self, queue_size: int = 1, request_timeout: float = 180.0) -> None:
        self._queue_size = queue_size
        self._configuration_error = (
            "HAILO_QUEUE_SIZE must be an integer greater than or equal to 1"
            if queue_size < 1
            else None
        )
        self._request_timeout = request_timeout
        self._state = "stopped"
        self._error: str | None = None
        self._queue: asyncio.Queue[_InferenceRequest | None] | None = None
        self._executor: ThreadPoolExecutor | None = None
        self._worker: asyncio.Task[None] | None = None
        self._vdevice: Any = None
        self._models: dict[str, Any] = {}

    @property
    def status(self) -> dict[str, Any]:
        return {
            "status": self._state,
            "ready": self._state == "ready",
            "error": self._error,
            "queued": self._queue.qsize() if self._queue is not None else 0,
        }

    async def start(self, models: Sequence[dict[str, Any]]) -> None:
        """Initialize the device and configured models during app startup."""
        if self._state != "stopped":
            return
        if self._configuration_error is not None:
            self._state = "failed"
            self._error = self._configuration_error
            return
        model_paths = sorted({model["hef_path"] for model in models})
        if not model_paths:
            self._state = "failed"
            self._error = "No configured, usable HEF models"
            return

        self._state = "starting"
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="hailo-inference",
        )
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(self._executor, self._initialize, model_paths)
        except Exception as exc:
            self._state = "failed"
            self._error = type(exc).__name__
            logger.exception("Failed to initialize direct Hailo backend")
            return

        self._queue = asyncio.Queue(maxsize=self._queue_size)
        self._state = "ready"
        self._worker = asyncio.create_task(self._run_worker())

    def _initialize(self, model_paths: list[str]) -> None:
        from hailo_apps.python.core.common.defines import SHARED_VDEVICE_GROUP_ID
        from hailo_platform import VDevice
        from hailo_platform.genai import LLM

        params = VDevice.create_params()
        params.group_id = SHARED_VDEVICE_GROUP_ID
        self._vdevice = VDevice(params)
        try:
            for path in model_paths:
                self._models[path] = LLM(self._vdevice, path)
        except Exception:
            self._release_native()
            raise

    async def generate(
        self,
        hef_path: str,
        messages: list[dict[str, Any]],
        generation: dict[str, Any],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str = "auto",
    ) -> str:
        """Queue generation without letting caller cancellation release ownership."""
        if self._state != "ready" or self._queue is None:
            raise BackendUnavailableError(self._error or "Hailo backend is not ready")
        loop = asyncio.get_running_loop()
        result: asyncio.Future[str] = loop.create_future()
        result.add_done_callback(_consume_future_exception)
        job = _InferenceRequest(
            hef_path,
            messages,
            generation,
            tools,
            tool_choice,
            result,
        )
        try:
            self._queue.put_nowait(job)
        except asyncio.QueueFull as exc:
            raise BackendBusyError("Hailo inference queue is full") from exc

        try:
            return await asyncio.wait_for(
                asyncio.shield(result),
                timeout=self._request_timeout,
            )
        except TimeoutError as exc:
            raise BackendTimeoutError(
                "Hailo inference is still running after the request deadline"
            ) from exc

    async def _run_worker(self) -> None:
        assert self._queue is not None
        assert self._executor is not None
        loop = asyncio.get_running_loop()
        while True:
            job = await self._queue.get()
            if job is None:
                self._queue.task_done()
                return
            try:
                content = await loop.run_in_executor(
                    self._executor,
                    self._generate_native,
                    job,
                )
            except Exception as exc:
                self._state = "failed"
                self._error = type(exc).__name__
                logger.exception("Direct Hailo generation failed")
                if not job.result.done():
                    job.result.set_exception(
                        BackendGenerationError("Native Hailo generation failed")
                    )
                self._fail_queued()
            else:
                if not job.result.done():
                    job.result.set_result(content)
            finally:
                self._queue.task_done()

    def _generate_native(self, job: _InferenceRequest) -> str:
        llm = self._models.get(job.hef_path)
        if llm is None:
            raise RuntimeError("Configured HEF was not loaded at startup")
        llm.clear_context()
        prompt = []
        for message in job.messages:
            native_message = {
                "role": message["role"],
                "content": [{"type": "text", "text": message["content"]}],
            }
            for field in ("tool_calls", "tool_call_id", "name"):
                if field in message:
                    native_message[field] = message[field]
            prompt.append(native_message)

        arguments = dict(job.generation)
        if job.tools is not None and job.tool_choice == "auto":
            arguments["tools"] = job.tools
        result = llm.generate_all(prompt=prompt, **arguments)
        if not isinstance(result, str):
            raise TypeError("HailoRT returned a non-text generation result")
        return result

    def _fail_queued(self) -> None:
        assert self._queue is not None
        while True:
            try:
                job = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            if job is not None and not job.result.done():
                job.result.set_exception(
                    BackendUnavailableError("Hailo backend failed during generation")
                )
            self._queue.task_done()

    async def close(self) -> None:
        """Drain accepted work, then release native resources on their owner thread."""
        if self._state == "stopped":
            return
        self._state = "stopping"
        if self._worker is not None and self._queue is not None:
            await self._queue.put(None)
            await self._worker

        release_error: Exception | None = None
        if self._executor is not None:
            loop = asyncio.get_running_loop()
            try:
                await loop.run_in_executor(self._executor, self._release_native)
            except Exception as exc:
                release_error = exc
                logger.exception("Failed to release direct Hailo resources")
            self._executor.shutdown(wait=True)

        if release_error is not None:
            self._state = "failed"
            self._error = "resource_release_failed"
        else:
            self._state = "stopped"

    def _release_native(self) -> None:
        errors = []
        for llm in self._models.values():
            try:
                llm.clear_context()
            except Exception as exc:
                errors.append(exc)
            try:
                llm.release()
            except Exception as exc:
                errors.append(exc)
        self._models.clear()
        if self._vdevice is not None:
            try:
                self._vdevice.release()
            except Exception as exc:
                errors.append(exc)
            self._vdevice = None
        if errors:
            raise RuntimeError(
                "One or more Hailo resources failed to release"
            ) from errors[0]


def _consume_future_exception(future: asyncio.Future[Any]) -> None:
    if not future.cancelled():
        future.exception()
