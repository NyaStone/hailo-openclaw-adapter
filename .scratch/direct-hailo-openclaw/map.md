# Direct Hailo OpenClaw Map

## Notes

- On `home` (`edgeAI`), the Hailo Apps checkout is clean at `891ce70` (`Release/26.03.1`); `venv_hailo_apps` exists.
- HailoRT-CLI 5.3.0 identifies a working HAILO10H device with firmware 5.3.0.
- Hailo Apps catalogs and now has installed `Qwen2.5-Coder-1.5B-Instruct` for `agent` and `v2a_demo` on `hailo10h`; `Qwen2-VL-2B-Instruct` is installed as well.
- The adapter is checked out at `hailo_apps/python/gen_ai_apps/hailo_openclaw_adapter` on `home`; focused and full test runs each passed all 5 tests in `venv_hailo_apps`.
- The Qwen coder's 2,048-token context completed a schema-driven synthetic weather call and a grounded fresh-context transcript replay. Its prompt template requires structured assistant `tool_calls` and wraps plain-JSON tool results in `<tool_response>`; observed tool-call body formatting varied.

## Decisions-so-far

- [Issue 02](issues/02-expose-configured-model-discovery.md): public model IDs now resolve only to configured existing Qwen2.5-Coder HEFs; discovery reports the validated HEF format, 2,048-token context, and completion/tools capabilities, while unknown IDs fail before inference.
- [Issue 03](issues/03-preserve-conversations-through-public-apis.md): both public chat APIs now pass the same complete sanitized conversation, request-specific tools, and normalized supported generation options to the backend; fake-backend tests verify the protocol boundary without native Hailo imports.
- [Issue 04](issues/04-serve-requests-through-direct-hailo.md): replaced the Hailo-Ollama proxy with a lifespan-owned, serialized HailoRT backend; bounded queue, readiness, cancellation/timeout ownership, and both text APIs are covered by fake-backend tests and a HAILO10H smoke test.
- [Issue 06](issues/06-stream-text-and-tool-call-responses.md): streamed OpenAI and Ollama responses preserve text and validated calls, emit protocol-appropriate terminal markers, and retain HTTP error framing when generation fails; public-route fake-backend tests cover both APIs.
- [Issue 07](issues/07-enforce-request-and-generation-limits.md): both chat APIs reject unsupported media/options, enforce required/forced tool choices, and return explicit context-overflow or output-exhaustion errors consistently for streamed and non-streamed requests. Native fake-model tests cover rendered transcript/schema token accounting and Hailo completion status.

## Fog
- Full OpenClaw tool-catalog capacity and model capability across other Qwen/LLM HEFs remain untested; validate those before advertising broader context or model support.
- Validate rendered-prompt token accounting and native completion-status mapping on the target HEF; verify the 256-token default output allowance against hardware behavior. See [Issue 07](issues/07-enforce-request-and-generation-limits.md).
- Full OpenClaw tool-catalog capacity and model capability across other Qwen/LLM HEFs remain untested; validate those before advertising broader context or model support.