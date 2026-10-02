# 06: Stream text and tool-call responses

**What to build:** Make streamed Ollama and OpenAI-compatible responses carry the same text, calls, terminal status, and error meaning as non-streaming responses.

**Blocked by:** 05: Complete the non-streaming tool handshake.

**Status:** resolved

- [x] OpenAI tool-call deltas include stable call IDs, indexes, names, and JSON-string arguments that standard clients can assemble.
- [x] Ollama NDJSON emits every completed tool call exactly once and ends successful streams with `done: true`.
- [x] Text, calls, finish reasons, and failures have equivalent meaning between streaming and non-streaming modes.
- [x] A generation failure never emits successful terminal framing.
- [x] Public-route tests cover streamed text, one or multiple calls, terminal markers, and error framing in both API formats.

## Answer

Streaming now preserves validated tool calls alongside assistant text. OpenAI SSE
emits indexed call deltas with stable IDs and JSON-string arguments, then uses
`tool_calls` as the finish reason; Ollama NDJSON emits the full assistant
message, including each call exactly once, with `done: true`. Since generation
completes before response framing begins, failures retain the existing HTTP
error response and never emit a successful terminal marker. Public-route tests
cover text, single and multiple calls, terminal markers, and generation errors
for both APIs.
