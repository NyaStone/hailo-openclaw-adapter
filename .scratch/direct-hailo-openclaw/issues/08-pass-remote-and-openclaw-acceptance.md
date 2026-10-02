# 08: Pass remote and OpenClaw acceptance

**What to build:** Validate the committed feature branch in the Hailo Apps runtime and through OpenClaw before it is eligible to merge.

**Blocked by:** 02: Expose configured model discovery; 03: Preserve conversations through the public APIs; 04: Serve requests through the direct Hailo backend; 05: Complete the non-streaming tool handshake; 06: Stream text and tool-call responses; 07: Enforce request and generation limits.

**Status:** ready-for-human

- [x] Sync the committed feature branch to the adapter checkout in Hailo Apps on `home` and record the tested commit.
- [x] Run focused project tests followed by the full project suite using the Hailo Apps environment; record commands and outcomes.
- [x] Verify model discovery through `/api/show`.
- [ ] Stream a harmless schema-driven call through OpenClaw.
- [ ] Have OpenClaw execute the approved harmless tool, submit its synthetic result, and verify a grounded final answer.
- [ ] Do not merge the feature branch to `main` unless the remote test suite and required hardware acceptance checks pass.

## Comments

### Acceptance Record

- Feature branch: `tickets/08`
- Tested commit: `05fe66a317a187a9defe31d5e0a490e83708b25f`
- Remote checkout: `~/hailo-apps/hailo_apps/python/gen_ai_apps/hailo_openclaw_adapter`, clean at the tested commit.
- Hailo Apps environment: `~/hailo-apps/venv_hailo_apps`
- `pytest --collect-only -q`: 86 tests collected.
- `pytest tests/test_adapter.py -k 'limit_failures or native_context_budget or native_terminal_status or required_and_forced_tool_choices or tool_result_replay or invalid_tool_calls' -q`: 35 passed, 51 deselected.
- `pytest -q`: 86 passed.
- `ruff check src tests`: passed.
- On HAILO10H, `/readyz` returned 200; `/api/tags` listed only `qwen2.5-coder:1.5b`; `/api/show` returned 200 with `format=hef`, `qwen2.context_length=2048`, and `completion`/`tools` capabilities.
- `openclaw --profile hailo-acceptance models list --refresh` discovered the model and reported its 2k context.
- `openclaw --profile hailo-acceptance agent --local --model ollama/qwen2.5-coder:1.5b --message 'Reply with exactly: Hailo provider connection works.' --thinking off --timeout 240 --json` was rejected before inference: OpenClaw 2026.9.5 requires at least 4,000 context tokens; the validated HEF reports 2,048.

### Blocker

The model discovery and public API checks pass, but OpenClaw refuses to start an agent turn with the truthful 2,048-token HEF profile. Consequently no streamed schema-driven call, tool execution, synthetic result replay, or grounded final answer was completed. Do not raise the configured context above the HEF's measured capacity to bypass OpenClaw's minimum. Resume acceptance only with a validated supported HEF profile of at least 4,000 tokens or an OpenClaw version that supports this model's actual context.
