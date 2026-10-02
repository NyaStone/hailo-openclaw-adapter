# Direct Hailo Migration Handoff

Date: 2026-10-02

## Goal and Scope

Replace this project's hailo-ollama HTTP proxy backend with direct Hailo
inference, using hailo-apps resource utilities and HailoRT's GenAI bindings.
Preserve the public Ollama/OpenAI interfaces and implement the complete
OpenClaw tool-call/result conversation flow.

The user initially requested research and a migration plan, not implementation.
That research is complete. No implementation files were changed, no tests were
run, and no hardware inference or tool execution was performed. This handoff
does not itself authorize starting the implementation.

The user explicitly requested this handoff at the project root, overriding the
handoff skill's default temporary-directory location.

## Existing Evidence

Read `/memories/repo/direct-hailo-research.md` with the memory tool for the
verified signatures, source locations, versions, library limitations, and
current adapter behavior. Do not repeat the earlier broad research unless
versions or the target model have changed. That artifact contains environment
details; do not copy personal paths into shared documents or logs.

Primary references:

- [Hailo user guide](https://github.com/hailo-ai/hailo-apps/blob/main/doc/user_guide/README.md).
- [Hailo developer guide](https://github.com/hailo-ai/hailo-apps/blob/main/doc/developer_guide/README.md).
- [Utilities at the inspected revision](https://github.com/hailo-ai/hailo-apps/tree/891ce701c2ebe239a5d277759eb75a30f76678a9/hailo_apps/python/gen_ai_apps/gen_ai_utils/llm_utils).
- [Minimal direct inference example](https://github.com/hailo-ai/hailo-apps/blob/891ce701c2ebe239a5d277759eb75a30f76678a9/hailo_apps/python/gen_ai_apps/simple_llm_chat/simple_llm_chat.py).
- [Reference tool agent](https://github.com/hailo-ai/hailo-apps/blob/891ce701c2ebe239a5d277759eb75a30f76678a9/hailo_apps/python/gen_ai_apps/agent_tools_example/agent.py).
- [OpenClaw Ollama documentation](https://docs.openclaw.ai/providers/ollama).

The runtime is not installed in the Windows workspace. Read-only access worked
via `ssh -o BatchMode=yes -o ConnectTimeout=10 home`; the checkout is
`~/hailo-apps/`. Prefer the installed binding and provider source over unversioned
online documentation. Remote `rg` is unavailable; use `grep` and `find`.
Avoid embedded regex-alternation pipes in Windows SSH command strings because
quote handling caused remote shell misinterpretation during research.

## Proposed Architecture

Keep FastAPI as the protocol layer. Introduce a backend abstraction that owns
model loading, generation, hardware lifetime, and recovery, plus shared internal
conversation/result/event structures for both public response formats.

Start with a single serialized hardware worker and one service process. Run
blocking native generation away from the event loop. Clear native context and
replay the complete submitted transcript for each request. This isolates clients
and satisfies the binding's fresh-context requirement for system messages and
tool schemas. Defer context caching until correctness and explicit isolation
keys are established.

OpenClaw owns tool execution. The adapter advertises model capabilities,
accepts request-specific tool schemas, returns structured calls, and consumes
tool results on subsequent requests. Do not import the example agent's local
tool execution loop or hard-code an OpenClaw tool inventory.

## Migration Plan

### 1. Validate the Model Contract Before Refactoring

On the target host, identify a configured, available LLM HEF and confirm hardware
availability without stopping unrelated services. Inspect its prompt template.
Use native `tools=` with one harmless tool and feed back a synthetic result;
there is no need to execute shell, file, or web tools for this check.

Verify system instructions, multiline content, assistant call history, native
tool-result roles, and fresh-context full-transcript replay. Inspect generated
call syntax, completion status, and generator cleanup behavior. Establish the
actual context capacity and whether the full OpenClaw schema set fits.

Acceptance: one schema-driven call followed by an answer grounded in the supplied
result, with no invented execution. A failure should discriminate between model
capability, prompt-template handling, resource setup, and protocol translation.
Do not assume bypassing hailo-ollama fixes native prompt-renderer quirks.

### 2. Introduce the Direct Backend

Replace the chat HTTP transport in `src/hailo_ollama_adapter/adapter.py` with an
injectable backend owning `VDevice`, `LLM`, and generation resources. Keep native
imports lazy so protocol tests work on Windows without hardware dependencies.
Initialize and release resources through the application lifespan.

Serialize context reset, generation, status inspection, and cleanup. Keep bounded
queue/backpressure behavior and retain worker ownership after client disconnect
until native completion or verified cancellation. An HTTP timeout or cancelled
async waiter must not falsely imply a native worker stopped. Define explicit
readiness, busy, failed, and recovery behavior with appropriate HTTP errors.

### 3. Preserve Requests and Conversation History

Replace `_build_payload()`, `normalize_messages()`, and
`assemble_messages_for_hailo()` with validated protocol-to-domain translation.
Preserve system instructions, assistant calls with empty content, call IDs,
tool-result associations, and multiline text. Reject unsupported media explicitly
for text-only models rather than silently dropping it.

Remove unconditional OpenClaw-envelope extraction and character/turn truncation.
Account for the rendered prompt, schemas, history, and output allowance within
the HEF's fixed capacity. Report overflow clearly so OpenClaw can compact/retry;
do not silently remove instructions or split tool exchanges. Map supported
generation parameters explicitly and define handling for unsupported options.

### 4. Implement Tool Parsing and Both Response Protocols

Prefer native tool schemas. Use a model-specific prompt fallback only if the
first validation demonstrates a need. Parse all supported calls, validate names
against the request's tool inventory and arguments against its JSON schemas,
and preserve IDs when replaying calls and results. Do not turn incomplete or
malformed tool output into a successful text completion.

Use shared events for text, completed tool calls, terminal status, and errors.
Ollama responses need argument objects and native `message.tool_calls`; OpenAI
responses need IDs, JSON-string arguments, indexed tool deltas, and the proper
`tool_calls` finish reason. Define enforcement of OpenAI `tool_choice`, including
none, required, and forced-name requests, rather than silently ignoring it.

Emit each complete native Ollama call exactly once across the stream. Finish
successful NDJSON streams with `done: true`; do not emit successful termination
after failure. Distinguish normal stop from output exhaustion and prevent
execution of truncated calls. OpenClaw's internal `toolUse` stop reason is
inferred from calls; it is not a required native Ollama `done_reason` value.

### 5. Replace Discovery, Configuration, and Deployment Assumptions

Create an explicit public model ID to HEF mapping. List only configured usable
models and return not-found errors for unknown IDs. Keep discovery probes
responsive while inference runs. Avoid implicit downloads during chat requests;
the library's resource resolver can download missing catalog models.

Return truthful HEF format and context metadata. Advertise `tools` in `/api/show`
only for validated model profiles; do not claim vision, thinking, or embeddings
without implementing and validating them. Native Ollama base URLs have no `/v1`.

Update `src/hailo_ollama_adapter/cli.py`, `pyproject.toml`, `setup.py`,
`requirements.txt`, and `README.md` as needed. Remove hailo-ollama startup and URL
requirements. Add model configuration, readiness, hardware concurrency, and
runtime installation guidance. Keep HailoRT installation separate from ordinary
pip dependencies and avoid unnecessary voice/UI extras for a text-only server.
Do not rename the package or command unless separately agreed.

### 6. Validate Incrementally and End to End

Replace transport-specific tests in `tests/test_adapter.py` and related fixtures
with a fake backend exercising public behavior. Keep local tests independent
of native imports. Cover discovery, text requests, system preservation, calls
and results, multiple/sequential calls, invalid names/arguments, tool choice,
unknown models, unsupported media/options, overflow, output exhaustion, and
stream framing. Also test disconnects, worker serialization, cleanup failures,
and cross-session isolation.

Replay focused cases against the target HEF, then against the installed
OpenClaw native provider. Acceptance requires discovery through `/api/show`,
a streamed call, OpenClaw execution of a harmless approved tool, result replay,
and a grounded final answer. Protocol support alone does not establish that a
small model reliably handles OpenClaw's full agent workload.

## Suggested Skills

The next agent should call the Skill tool for the relevant skills below. If that
tool is unavailable, read the skill's instructions directly.

- `codebase-design`: design the backend boundary and shared protocol model.
- `python-fact-grounded-coding`: verify runtime contracts and Python behavior.
- `tdd`: implement the migration through focused fake-backend behavior tests.
- `research`: capture new primary-source findings only when a specific unknown
  remains; reuse the existing research artifact first.
- `diagnosing-bugs`: investigate concrete native/template/cancellation failures.
- `code-review`: review implementation against the agreed plan and repo standards
  after choosing an explicit comparison point.

Project skills live under `.agents/skills/`. The Python fact-grounded skill is
provided by the installed Pylance extension; resolve its current path from the
available skill listing rather than embedding a personal filesystem path.

## Immediate Next Action

When implementation is authorized, perform milestone 1 before rewriting the
adapter. Record the chosen model, tested input/output, prompt-template findings,
capacity, and cleanup observations in a focused validation artifact. Then make
the smallest backend change supported by that evidence and run its narrow test
before expanding scope. Preserve unrelated worktree changes, including recently
installed skill files; do not commit, deploy, or stop services without permission.