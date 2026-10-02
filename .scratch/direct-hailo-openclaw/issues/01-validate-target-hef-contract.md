# 01: Validate the target HEF contract on home

**What to build:** Establish a usable remote Hailo Apps runtime and verify that a configured LLM HEF can follow the conversation and native tool contract required by the adapter.

**Type:** research

**Blocked by:** None (can start immediately).

**Status:** ready-for-human

- [ ] Reuse the existing Hailo Apps checkout and virtual environment when usable; check out this adapter as a project-owned app without modifying the framework or system-level code.
- [ ] Identify an available configured LLM HEF and inspect its prompt-template behavior for system instructions, multiline text, assistant call history, tool results, and fresh-context transcript replay.
- [ ] Demonstrate a harmless schema-driven call followed by a grounded response to a synthetic tool result; do not execute a real tool.
- [ ] Record the selected model, tested inputs and results, context capacity for the intended schema payload, completion status, and generator cleanup observations.
- [ ] Run any adapter project code only from a committed feature-branch revision synced to `home`.

## Findings

Validation is paused pending model resource availability; target-HEF acceptance was not run.

- Remote host: `home` (`edgeAI`); existing Hailo Apps checkout is clean at `891ce70` (`Release/26.03.1`), and `venv_hailo_apps` exists.
- Runtime/device: HailoRT-CLI 5.3.0; `hailortcli fw-control identify` detected a HAILO10H device with firmware 5.3.0.
- Configured candidate: `Qwen2.5-Coder-1.5B-Instruct` is catalogued for `v2a_demo` on `hailo10h` with source `gen-ai-mz`.
- Availability: no Qwen/LLM HEF is installed under `/usr/local/hailo/resources`; the configured catalog entry is not a locally available HEF. No download was attempted.
- Tested inputs/results: none. System instructions, multiline text, assistant tool-call history, synthetic tool results, and fresh-context replay were not exercised. No real tool was run.
- Context capacity, completion status, and generator cleanup: unknown because no target HEF was loaded.
- Next step: make the candidate HEF available through an approved setup, then repeat prompt-contract validation before backend implementation. Do not infer tool support or capacity from catalog metadata.
