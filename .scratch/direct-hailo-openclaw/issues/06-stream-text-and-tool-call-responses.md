# 06: Stream text and tool-call responses

**What to build:** Make streamed Ollama and OpenAI-compatible responses carry the same text, calls, terminal status, and error meaning as non-streaming responses.

**Blocked by:** 05: Complete the non-streaming tool handshake.

**Status:** ready-for-agent

- [ ] OpenAI tool-call deltas include stable call IDs, indexes, names, and JSON-string arguments that standard clients can assemble.
- [ ] Ollama NDJSON emits every completed tool call exactly once and ends successful streams with `done: true`.
- [ ] Text, calls, finish reasons, and failures have equivalent meaning between streaming and non-streaming modes.
- [ ] A generation failure never emits successful terminal framing.
- [ ] Public-route tests cover streamed text, one or multiple calls, terminal markers, and error framing in both API formats.
