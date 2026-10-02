# 05: Complete the non-streaming tool handshake

**What to build:** Let OpenClaw provide request-specific tool schemas, receive structured model-selected calls, execute them itself, and submit results for continued inference.

**Blocked by:** 03: Preserve conversations through the public APIs; 04: Serve requests through the direct Hailo backend.

**Status:** ready-for-agent

- [ ] Tool schemas are passed to native inference for the current request only.
- [ ] Ollama-compatible responses return argument objects; OpenAI-compatible responses return tool-call IDs and JSON-string arguments.
- [ ] Generated tool names must match the request inventory and arguments must validate against the corresponding JSON schema.
- [ ] Assistant tool-call history and matching tool results are replayed in fresh context so the next response can use the supplied result.
- [ ] The adapter returns calls without executing them; invalid, malformed, or incomplete calls fail rather than becoming successful text completions.
- [ ] Public-route tests cover call IDs, arguments, tool-result replay, multiple or sequential calls, and cross-request isolation with a fake backend.
