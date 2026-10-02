# 07: Enforce request and generation limits

**What to build:** Report requests and generated results that cannot be honored safely instead of silently changing their meaning or presenting incomplete calls as successful.

**Blocked by:** 05: Complete the non-streaming tool handshake; 06: Stream text and tool-call responses.

**Status:** ready-for-agent

- [ ] Unsupported media and generation options are explicitly rejected for models that cannot handle them.
- [ ] OpenAI-compatible `tool_choice` values for none, required, and a forced tool name are enforced or explicitly rejected when unsupported.
- [ ] Context accounting includes the rendered prompt, full transcript, tool schemas, and output allowance; overflow is reported without silently dropping history or instructions.
- [ ] Output exhaustion is distinct from a normal stop; truncated, malformed, or schema-invalid calls cannot be reported as successful executable calls.
- [ ] Public tests verify matching failure meaning in streaming and non-streaming Ollama and OpenAI-compatible responses.
