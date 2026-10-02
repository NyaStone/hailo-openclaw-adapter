# 07: Enforce request and generation limits

**What to build:** Report requests and generated results that cannot be honored safely instead of silently changing their meaning or presenting incomplete calls as successful.

**Blocked by:** 05: Complete the non-streaming tool handshake; 06: Stream text and tool-call responses.

**Status:** resolved

- [x] Unsupported media and generation options are explicitly rejected for models that cannot handle them.
- [x] OpenAI-compatible `tool_choice` values for none, required, and a forced tool name are enforced or explicitly rejected when unsupported.
- [x] Context accounting includes the rendered prompt, full transcript, tool schemas, and output allowance; overflow is reported without silently dropping history or instructions.
- [x] Output exhaustion is distinct from a normal stop; truncated, malformed, or schema-invalid calls cannot be reported as successful executable calls.
- [x] Public tests verify matching failure meaning in streaming and non-streaming Ollama and OpenAI-compatible responses.

## Answer

Both public chat APIs now reject unsupported inputs and enforce tool-choice
semantics. The native backend tokenizes the model-rendered full transcript and
tool schemas, reserves the requested output budget (256 tokens by default), and
reports context overflow before generation. HailoRT's `MAX_TOKENS_REACHED`
terminal state is reported as output exhaustion; malformed and invalid tool
calls remain generation failures. Public tests cover both protocols, both
streaming modes, and a fake native model at the FastAPI boundary.

Validation on `home`, branch `tickets/07`, commit `ff79a979c8a9b87bf004d81d4cbe6d98a2c72180`:
`pytest tests/test_adapter.py -q` (86 passed), `pytest -q` (86 passed), and
`ruff check src tests` (passed). Native hardware inference remains a separate
target-HEF validation gate.
