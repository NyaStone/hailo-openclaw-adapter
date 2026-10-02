# 03: Preserve conversations through the public APIs

**What to build:** Make both public API formats preserve supported conversation history through an injected fake backend, independently of native Hailo bindings.

**Blocked by:** 01: Validate the target HEF contract on home.

**Status:** resolved

- [x] Public-route tests verify system instructions, multiline text, and the full supported message history are preserved.
- [x] Empty-content assistant tool-call messages retain call IDs, names, and arguments; matching tool-result associations remain intact on later requests.
- [x] Request-scoped tool schemas and supported generation settings reach the injected backend without hard-coded tool inventories.
- [x] The same internal conversation representation serves Ollama-compatible and OpenAI-compatible requests.
- [x] Fake-backend protocol tests run without importing native Hailo modules.

## Answer

Both public chat routes now use the same conversation mapping. It preserves all
messages, multiline text, empty assistant tool-call content and metadata, and
tool-result associations without truncating the transcript. Request-scoped
schemas and tool policy reach the backend, and supported Ollama/OpenAI
generation settings map to the Hailo generation option names. The route tests
exercise a fake backend without native Hailo imports, including a transcript
longer than the former history and message-size limits.

On `home`, revision `eaf19a7` passed Ruff and the focused public-route tests
(`2 passed`). The full project suite is run on the final ticket-resolution
revision.
