# 04: Serve requests through the direct Hailo backend

**What to build:** Answer text requests through HailoRT using one serialized owner of inference resources while keeping the protocol service responsive.

**Blocked by:** 02: Expose configured model discovery; 03: Preserve conversations through the public APIs.

**Status:** resolved

- [x] The configured public model ID selects its mapped HEF and native generation returns a text response through both public APIs.
- [x] Native imports remain lazy for fake-backend protocol use; resources initialize and release through application lifespan.
- [x] Blocking native generation runs away from the async event loop, and one worker serializes device and context access.
- [x] Queueing is bounded and overload returns a clear busy response; readiness and failed states are explicit.
- [x] Client disconnect or timeout does not release native ownership before completion or verified cancellation; backend failures and shutdown have defined cleanup behavior.
- [x] CLI, runtime configuration, and documentation describe the direct Hailo Apps runtime, model mapping, readiness, and hardware concurrency without requiring Hailo-Ollama.

## Answer

Implemented the direct HailoRT backend with lifespan-owned `VDevice`/`LLM`
resources, a serialized executor worker, bounded queue, `/readyz`, and both
Ollama/OpenAI text response formats. Fake-backend tests cover API routing,
conversation preservation, streaming framing, readiness, overload, and
cancellation/timeout ownership. Tool-call response parsing and context-overflow
enforcement remain outside this text-generation ticket in the parent spec.

- Feature branch: `tickets/04`
- Tested commit: `2b994b570b6108730589a8bc7c4144e444477453`
- On `home`: `python -m pytest -q` -> 26 passed; `ruff check src tests` -> passed.
- HAILO10H smoke: mapped Qwen2.5-Coder HEF initialized; `/api/chat` and
	`/v1/chat/completions` both returned clean text; `/readyz` returned 200 and
	application lifespan released resources.
