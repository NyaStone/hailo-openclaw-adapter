# Direct Hailo OpenClaw Map

## Notes

- On `home` (`edgeAI`), the Hailo Apps checkout is clean at `891ce70` (`Release/26.03.1`); `venv_hailo_apps` exists.
- HailoRT-CLI 5.3.0 identifies a working HAILO10H device with firmware 5.3.0.
- Hailo Apps catalogs and now has installed `Qwen2.5-Coder-1.5B-Instruct` for `agent` and `v2a_demo` on `hailo10h`; `Qwen2-VL-2B-Instruct` is installed as well.
- The adapter is checked out at `hailo_apps/python/gen_ai_apps/hailo_openclaw_adapter` on `home`; focused and full test runs each passed all 5 tests in `venv_hailo_apps`.
- The Qwen coder's 2,048-token context completed a schema-driven synthetic weather call and a grounded fresh-context transcript replay. Its prompt template requires structured assistant `tool_calls` and wraps plain-JSON tool results in `<tool_response>`; observed tool-call body formatting varied.

## Decisions-so-far

- [Issue 02](issues/02-expose-configured-model-discovery.md): public model IDs now resolve only to configured existing Qwen2.5-Coder HEFs; discovery reports the validated HEF format, 2,048-token context, and completion/tools capabilities, while unknown IDs fail before inference.

## Fog
- Full OpenClaw tool-catalog capacity and model capability across other Qwen/LLM HEFs remain untested; validate those before advertising broader context or model support.