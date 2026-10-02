# 05: Complete the non-streaming tool handshake

**What to build:** Let OpenClaw provide request-specific tool schemas, receive structured model-selected calls, execute them itself, and submit results for continued inference.

**Blocked by:** 03: Preserve conversations through the public APIs; 04: Serve requests through the direct Hailo backend.

**Status:** resolved

- [x] Tool schemas are passed to native inference for the current request only.
- [x] Ollama-compatible responses return argument objects; OpenAI-compatible responses return tool-call IDs and JSON-string arguments.
- [x] Generated tool names must match the request inventory and arguments must validate against the corresponding JSON schema.
- [x] Assistant tool-call history and matching tool results are replayed in fresh context so the next response can use the supplied result.
- [x] The adapter returns calls without executing them; invalid, malformed, or incomplete calls fail rather than becoming successful text completions.
- [x] Public-route tests cover call IDs, arguments, tool-result replay, multiple or sequential calls, and cross-request isolation with a fake backend.

## Answer

Implemented request-scoped native tool schemas and a validated non-streaming
tool-call handshake through both public APIs. Complete JSON and XML-like Hailo
tool payloads become protocol tool calls with generated call IDs; the adapter
validates names and argument objects against that request's JSON Schemas and
returns malformed, incomplete, unknown, or invalid calls as generation errors.
OpenClaw remains the only tool executor. Existing conversation replay carries
assistant calls and matching tool results through fresh native context on the
next request. Public-route fake-backend tests cover multiple calls, sequential
result replay, and cross-request inventory isolation.
