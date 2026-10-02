# 01: Validate the target HEF contract on home

**What to build:** Establish a usable remote Hailo Apps runtime and verify that a configured LLM HEF can follow the conversation and native tool contract required by the adapter.

**Type:** research

**Blocked by:** None (can start immediately).

**Status:** ready-for-human

- [x] Reuse the existing Hailo Apps checkout and virtual environment when usable; check out this adapter as a project-owned app without modifying the framework or system-level code.
- [ ] Identify an available configured LLM HEF and inspect its prompt-template behavior for system instructions, multiline text, assistant call history, tool results, and fresh-context transcript replay.
- [ ] Demonstrate a harmless schema-driven call followed by a grounded response to a synthetic tool result; do not execute a real tool.
- [x] Record the candidate model, tested inputs/results, context capacity, completion status, and generator cleanup observations, including unavailable/unknown values.
- [x] Run adapter project code only from a committed feature-branch revision synced to `home`.

## Findings

Validation is paused pending model resource availability; target-HEF acceptance was not run.

- Remote host: `home` (`edgeAI`); existing Hailo Apps checkout is clean at `891ce70` (`Release/26.03.1`), and `venv_hailo_apps` exists.
- Runtime/device: HailoRT-CLI 5.3.0; `hailortcli fw-control identify` detected a HAILO10H device with firmware 5.3.0.
- Configured candidate: `Qwen2.5-Coder-1.5B-Instruct` is catalogued for `v2a_demo` on `hailo10h` with source `gen-ai-mz`.
- Availability: no Qwen/LLM HEF is installed under `/usr/local/hailo/resources`; the configured catalog entry is not a locally available HEF. No download was attempted.
- Tested inputs/results: none. System instructions, multiline text, assistant tool-call history, synthetic tool results, and fresh-context replay were not exercised. No real tool was run.
- Context capacity, completion status, and generator cleanup: unknown because no target HEF was loaded.
- Adapter checkout: `/home/nyastone/hailo-apps/hailo_apps/python/gen_ai_apps/hailo_openclaw_adapter`, feature branch `tickets/01`, tested revision `8c1b050f48388f465a5e9f7eb2ca0b14ee0fe80b`.
- Remote project tests in `venv_hailo_apps`: `python -m pytest tests/test_adapter.py -q` -> 5 passed; `python -m pytest -q` -> 5 passed. The declared `.[dev]` dependencies were installed into the existing venv first.
- Next step: make the candidate HEF available through an approved setup, then repeat prompt-contract validation before backend implementation. Do not infer tool support or capacity from catalog metadata.
