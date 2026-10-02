# Direct Hailo OpenClaw Map

## Notes

- On `home` (`edgeAI`), the Hailo Apps checkout is clean at `891ce70` (`Release/26.03.1`); `venv_hailo_apps` exists.
- HailoRT-CLI 5.3.0 identifies a working HAILO10H device with firmware 5.3.0.
- Hailo Apps catalogs `Qwen2.5-Coder-1.5B-Instruct` for `v2a_demo` on `hailo10h`, but no LLM HEF is installed in `/usr/local/hailo/resources`.
- The adapter is checked out at `hailo_apps/python/gen_ai_apps/hailo_openclaw_adapter` on `home`; focused and full test runs each passed all 5 tests in `venv_hailo_apps`.

## Decisions-so-far

- [Issue 01](issues/01-validate-target-hef-contract.md): validation is paused rather than downloading an uncached HEF. The adapter suite passes remotely, but prompt behavior, tool capability, context capacity, and generator cleanup remain unvalidated pending operator setup.

## Fog

- The operator must make an approved LLM HEF available in the configured resources before hardware prompt/tool-contract validation can proceed.
- Once available, test system instructions, multiline text, assistant call history, a harmless schema-driven call, a synthetic tool result, fresh-context transcript replay, capacity, completion, and generator cleanup.