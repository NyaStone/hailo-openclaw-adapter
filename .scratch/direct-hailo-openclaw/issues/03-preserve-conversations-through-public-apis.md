# 03: Preserve conversations through the public APIs

**What to build:** Make both public API formats preserve supported conversation history through an injected fake backend, independently of native Hailo bindings.

**Blocked by:** 01: Validate the target HEF contract on home.

**Status:** ready-for-agent

- [ ] Public-route tests verify system instructions, multiline text, and the full supported message history are preserved.
- [ ] Empty-content assistant tool-call messages retain call IDs, names, and arguments; matching tool-result associations remain intact on later requests.
- [ ] Request-scoped tool schemas and supported generation settings reach the injected backend without hard-coded tool inventories.
- [ ] The same internal conversation representation serves Ollama-compatible and OpenAI-compatible requests.
- [ ] Fake-backend protocol tests run without importing native Hailo modules.
