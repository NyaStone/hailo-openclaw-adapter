# 01: Validate the target HEF contract on home

**What to build:** Establish a usable remote Hailo Apps runtime and verify that a configured LLM HEF can follow the conversation and native tool contract required by the adapter.

**Blocked by:** None (can start immediately).

**Status:** ready-for-agent

- [ ] Reuse the existing Hailo Apps checkout and virtual environment when usable; check out this adapter as a project-owned app without modifying the framework or system-level code.
- [ ] Identify an available configured LLM HEF and inspect its prompt-template behavior for system instructions, multiline text, assistant call history, tool results, and fresh-context transcript replay.
- [ ] Demonstrate a harmless schema-driven call followed by a grounded response to a synthetic tool result; do not execute a real tool.
- [ ] Record the selected model, tested inputs and results, context capacity for the intended schema payload, completion status, and generator cleanup observations.
- [ ] Run any adapter project code only from a committed feature-branch revision synced to `home`.
