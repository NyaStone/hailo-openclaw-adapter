# 01: Validate the target HEF contract on home

**What to build:** Establish a usable remote Hailo Apps runtime and verify that a configured LLM HEF can follow the conversation and native tool contract required by the adapter.

**Type:** research

**Blocked by:** None (can start immediately).

**Status:** resolved

- [x] Reuse the existing Hailo Apps checkout and virtual environment when usable; check out this adapter as a project-owned app without modifying the framework or system-level code.
- [x] Identify an available configured LLM HEF and inspect its prompt-template behavior for system instructions, multiline text, assistant call history, tool results, and fresh-context transcript replay.
- [x] Demonstrate a harmless schema-driven call followed by a grounded response to a synthetic tool result; do not execute a real tool.
- [x] Record the candidate model, tested inputs/results, context capacity, completion status, and generator cleanup observations, including unavailable/unknown values.
- [x] Run adapter project code only from a committed feature-branch revision synced to `home`.

## Findings

Target-HEF prompt/tool validation completed on `home` using the installed Qwen2.5-Coder HEF. No real tool was executed.

- Remote host: `home` (`edgeAI`); existing Hailo Apps checkout is clean at `891ce70` (`Release/26.03.1`), and `venv_hailo_apps` exists.
- Runtime/device: HailoRT-CLI 5.3.0; `hailortcli fw-control identify` detected a HAILO10H device with firmware 5.3.0.
- Configured model: `Qwen2.5-Coder-1.5B-Instruct` is catalogued for `agent` and `v2a_demo` on `hailo10h` with source `gen-ai-mz`. The installed `/usr/local/hailo/resources/models/hailo10h/Qwen2.5-Coder-1.5B-Instruct.hef` loaded and generated successfully. `Qwen2-VL-2B-Instruct.hef` is also installed; it was not used for this text/tool contract test.
- Prompt template: the Qwen template places the system instruction and JSON function schemas in a system prefix, serializes assistant `tool_calls` as `<tool_call>` blocks, and renders each `role=tool` message inside a user `<tool_response>` block. Structured messages preserve literal newlines. System messages and tool schemas were supplied only on fresh contexts.
- Schema-driven test: provided a single `lookup_weather` function schema and the multiline user request `What is the weather in Testville?\nUse the available tool; do not guess.` The model returned `lookup_weather` with `{"city":"Testville"}`. No external tool ran; the test supplied the synthetic result `{"city":"Testville","temperature_c":17,"condition":"light rain","source":"synthetic test fixture"}`.
- Fresh-context replay: cleared context and replayed system, user, assistant `{content:"", tool_calls:[...]}`, and tool `{content:<JSON result>}` messages. The model answered `The weather in Testville is light rain with a temperature of 17 degrees Celsius.` It did not call the tool again. A preliminary replay using the raw assistant tool-call markup as `assistant.content` and an unwrapped tool result repeated the call; the structured `tool_calls` representation and plain JSON tool content succeeded.
- Output formatting observation: a separate probe emitted `<tool_call>` with nested `<name>` and `<arguments>` XML elements rather than the template's documented JSON-object body. A later probe emitted the expected JSON object inside `<tool_call>`. Consumers must validate and robustly parse generated calls rather than assume one body shape.
- Context: HEF-fixed `max_context_capacity()` is 2,048 tokens. After the one-function schema call, usage was 271/2,048 tokens; after full transcript replay and grounded answer, usage was 328/2,048. This measures only the synthetic single-tool payload, not a full OpenClaw tool catalog.
- Completion and cleanup: both streams ended with `LLMGeneratorCompletionStatus.LOGICAL_END_OF_GENERATION`. The generator context managers exited normally. `clear_context()` and `LLM.release()` succeeded, followed by successful `VDevice.release()`.
- Adapter checkout: `/home/nyastone/hailo-apps/hailo_apps/python/gen_ai_apps/hailo_openclaw_adapter`, feature branch `tickets/01`, tested revision `0d0ba183d2da90a4a603951620162e35b2a6df52`.
- Remote project tests in `venv_hailo_apps`: `python -m pytest tests/test_adapter.py -q` -> 5 passed; `python -m pytest -q` -> 5 passed. The declared `.[dev]` dependencies were installed into the existing venv first.
- Hardware acceptance demonstrated the selected HEF's tool flow for this synthetic schema only. Do not infer support for other models or full OpenClaw tool catalogs from this result.
