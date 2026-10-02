# Direct Hailo Backend and OpenClaw Tool Handshake

Label: ready-for-agent

## Problem Statement

The adapter currently reaches Hailo inference through the Hailo-Ollama HTTP
wrapper. That extra service boundary limits the adapter's control over Hailo
resource ownership, model selection, context replay, and generation behavior.
It also prevents the adapter from completing OpenClaw's tool-call handshake:
the current translation loses request tool schemas, assistant tool calls, and
tool-result associations, so OpenClaw cannot reliably execute a model-selected
tool and provide its result back to the model.

The project is not yet integrated into the Hailo Apps monorepo where its
application utilities and lower-level Hailo interfaces are available. The
existing Hailo Apps checkout and virtual environment are already present on the
remote `home` host, but the adapter project must be checked out there as an app
before integration work and hardware validation proceed.

## Solution

Integrate this project into the existing Hailo Apps monorepo and replace its
Hailo-Ollama HTTP proxy with a direct Hailo inference backend. Keep the adapter
as the protocol boundary: it continues to expose both Ollama-compatible and
OpenAI-compatible APIs, translates requests into a complete conversation,
passes request-scoped tool schemas to inference, and returns tool calls in the
format OpenClaw expects. OpenClaw remains responsible for executing tools; the
adapter accepts tool results on later requests and supplies them to Hailo.

Before refactoring the backend, validate the target model and native tool
contract on `home`. Then implement and test behavior through the public API
using an injected fake backend. Run project code and tests only on `home`, from
the committed feature branch synced into the adapter's checkout in Hailo Apps.
Changes are limited to this project; do not modify Hailo Apps framework,
HailoRT, driver, or other system-level code. The feature branch may be committed
and tested while incomplete, but it must not be merged to `main` until remote
tests pass.

## User Stories

1. As a developer, I want the adapter checked out within the existing Hailo
   Apps monorepo, so that it can use the framework's application utilities and
   direct Hailo interfaces.
2. As a developer, I want the adapter to use the existing Hailo Apps virtual
   environment, so that its Hailo dependencies match the supported runtime.
3. As a project maintainer, I want integration changes confined to this
   project, so that the Hailo Apps framework and accelerator drivers remain
   untouched.
4. As an operator, I want setup to reuse the existing remote Hailo Apps
   checkout and environment when usable, so that setup avoids needless
   reinstallations or changes to the host.
5. As an operator, I want setup to stop with a clear report when a required
   checkout, environment, configured model, or hardware prerequisite is
   unavailable, so that no implicit downloads or system changes occur.
6. As an operator, I want to map a public model ID explicitly to a configured
   usable HEF, so that each API request selects a known inference model.
7. As an OpenClaw client, I want model discovery to list only configured usable
   models, so that onboarding does not advertise placeholders or unavailable
   models.
8. As an OpenClaw client, I want model detail responses to report truthful
   model format, context capacity, and validated capabilities, so that OpenClaw
   can make suitable requests.
9. As an OpenClaw client, I want unknown model IDs to receive a not-found error,
   so that a misspelled or unavailable model cannot silently select another
   model.
10. As an operator, I want model discovery and readiness probes to remain
    responsive during inference, so that OpenClaw setup and health checks do
    not depend on the inference worker being idle.
11. As an OpenClaw client, I want to submit a system instruction and the full
    supported conversation history, so that the model's answer follows the
    current request and prior turns.
12. As an OpenClaw client, I want multiline text and empty-content assistant
    tool-call messages preserved, so that valid conversation structure is not
    flattened or discarded.
13. As an OpenClaw client, I want tool-call IDs, names, arguments, and matching
    tool results preserved across requests, so that the model can continue from
    the result of the exact call it made.
14. As an OpenClaw client, I want to provide tool schemas per request, so that
    the model can choose only from the tools available in that conversation.
15. As an OpenClaw client, I want Hailo to return structured tool calls in both
    API formats, so that OpenClaw can execute the selected tool rather than
    interpreting the call as ordinary assistant text.
16. As an OpenClaw operator, I want the adapter to return calls without
    executing them, so that tool permissions and execution remain under
    OpenClaw's control.
17. As an OpenClaw client, I want tool results submitted in a subsequent
    request to be included in the next inference context, so that the assistant
    can ground its final answer in the actual tool output.
18. As an OpenAI-compatible client, I want `tool_choice` values for none,
    required, and a forced tool name enforced or explicitly rejected when
    unsupported, so that the requested tool policy is never silently ignored.
19. As an OpenAI-compatible client, I want streamed tool-call deltas to include
    stable call IDs, indexes, names, and JSON-string arguments, so that standard
    clients can assemble the call correctly.
20. As an Ollama-compatible client, I want streamed output to emit each
    completed tool call exactly once and end successful streams with
    `done: true`, so that OpenClaw detects calls and terminates the response
    reliably.
21. As a client, I want malformed, incomplete, truncated, unknown, or
    schema-invalid tool calls reported as failures rather than successful text
    completions, so that OpenClaw does not execute invalid or partial calls.
22. As a client, I want output exhaustion distinguished from a normal stop,
    so that an incomplete tool call cannot be mistaken for an executable one.
23. As an OpenClaw client, I want unsupported media and generation options
    rejected explicitly for a model that cannot handle them, so that request
    content and settings are not silently lost.
24. As an OpenClaw client, I want context overflow reported clearly without
    silently deleting instructions, history, or tool exchanges, so that
    OpenClaw can compact or retry the request safely.
25. As an operator, I want one serialized owner of the Hailo device and native
    context, so that simultaneous requests cannot corrupt shared inference
    state.
26. As an operator, I want native generation to run away from the async
    protocol event loop, so that probes and unrelated request handling remain
    responsive while inference blocks.
27. As an operator, I want bounded inference queueing and clear busy/readiness
    failures, so that overload is visible and memory use remains controlled.
28. As an operator, I want a disconnected or timed-out HTTP client not to imply
    that native inference has stopped, so that the device is not reused until
    completion or verified cancellation.
29. As an operator, I want native resources released during application
    shutdown and backend failures to produce a defined failed or recoverable
    state, so that resource cleanup and subsequent requests are predictable.
30. As an OpenAI-compatible client, I want non-streaming and streaming chat
    completions to carry the same text, calls, finish reason, and error meaning,
    so that choosing a response mode does not change inference semantics.
31. As an Ollama-compatible client, I want non-streaming and NDJSON chat
    responses to carry the same text, calls, terminal status, and error meaning,
    so that choosing a response mode does not change inference semantics.
32. As a developer, I want protocol tests to run without importing native
    Hailo bindings, so that API behavior can be checked independently from the
    accelerator host.
33. As a maintainer, I want both public APIs preserved during migration, so
    that existing OpenClaw and OpenAI-compatible clients continue to work.
34. As a maintainer, I want tests executed on `home` from the committed feature
    branch before merge, so that validation uses the actual Hailo Apps runtime
    and only passing changes reach `main`.

## Implementation Decisions

- Keep FastAPI as the public protocol layer and add an injectable inference
  backend that owns model loading, generation, native context, device lifetime,
  and recovery. Keep native imports lazy so fake-backend API tests do not
  require Hailo bindings.
- Integrate the adapter as a project-owned app in the existing Hailo Apps
  monorepo and use its established app/resource conventions. Preserve the
  adapter's repository ownership and confine code changes to this project;
  do not patch framework, driver, or system-level code.
- Use HailoRT GenAI bindings for inference and Hailo Apps utilities for
  project/model resource setup where applicable. Do not retain Hailo-Ollama as
  the chat-generation transport, run an example agent's local tool loop, or
  hard-code OpenClaw's tool inventory.
- Before backend refactoring, identify an available configured LLM HEF and
  validate the native prompt and tool contract. Include system instructions,
  multiline content, assistant call history, tool results, fresh-context
  transcript replay, completion status, generator cleanup, and the context
  capacity available for the intended OpenClaw schema payload. Use a harmless
  schema-driven call and a synthetic result; do not execute a real tool as part
  of this validation.
- Start with one serialized hardware worker and one service process. Run
  blocking native generation away from the event loop. Reset native context and
  replay the complete submitted transcript for each request so system messages
  and request-specific tools use fresh context and client histories remain
  isolated. Defer context caching.
- Initialize and release backend resources through application lifespan.
  Preserve worker ownership after client disconnect until native work completes
  or cancellation is verified. Define bounded queue/backpressure and explicit
  readiness, busy, failed, and recovery behavior.
- Translate API messages to a shared internal conversation model and shared
  generation events for text, complete tool calls, terminal status, and errors.
  Preserve roles, call IDs, tool associations, empty assistant content, and
  multiline text. Reject unsupported media instead of discarding it.
- Account for rendered prompt, full transcript, tool schemas, and output
  allowance against the HEF's fixed context capacity. Report overflow; do not
  silently truncate instructions/history, extract intent from OpenClaw
  envelopes, or split tool-call/result exchanges.
- Map supported generation parameters explicitly. Reject unsupported options
  when ignoring them could change requested behavior. Validate generated tool
  names against the request inventory and arguments against their JSON schemas.
  Malformed or incomplete calls, output exhaustion, and generation errors must
  not become successful executable calls.
- Keep OpenClaw as the only tool executor. Ollama responses use native
  `message.tool_calls` with argument objects; OpenAI responses use tool-call
  IDs and JSON-string arguments. Honor or explicitly reject unsupported
  `tool_choice` modes. Emit each complete call once in streams; report the
  correct tool-call finish reason for OpenAI and finish successful Ollama
  streams with `done: true`. Do not emit successful terminal framing after a
  generation failure.
- Configure an explicit public model-ID-to-HEF mapping. Do not download models
  as a side effect of chat requests. Advertise only truthful, validated model
  format, context, and capabilities; in particular, claim tool support only
  for model profiles validated for native tool use.
- Update runtime configuration, CLI, packaging, and documentation to remove
  Hailo-Ollama startup and URL requirements; document the Hailo Apps runtime,
  model mapping, readiness, and hardware concurrency. Keep HailoRT installation
  separate from ordinary Python package dependencies, avoid unrelated voice/UI
  extras, and retain the existing package and command names.
- Work on a feature branch. Committing and pushing an incomplete feature branch
  is permitted so it can be synchronized to `home` for testing. Do not merge
  the branch to `main` until the remote test suite and required hardware
  acceptance checks pass.

## Testing Decisions

- The primary automated seam is the public FastAPI boundary: send requests to
  the actual routes and assert protocol-visible responses while an injected
  fake backend returns controlled text, tool calls, exhaustion, and failures.
  This tests external behavior rather than internal helper calls and exercises
  the fewest meaningful seams across the adapter.
- Test both streaming and non-streaming Ollama and OpenAI-compatible APIs,
  including discovery, model details, unknown models, system/history
  preservation, tool schemas, assistant calls, tool-result replay, call IDs,
  multiple or sequential calls, invalid calls/arguments, `tool_choice`, error
  framing, and terminal markers.
- Test boundary failures through public behavior: unsupported media/options,
  context overflow, output exhaustion, native generation failure, busy or
  unavailable backend, queue limits, disconnect while work continues,
  serialized access, cross-session isolation, and cleanup/recovery failures.
  Verify native modules are not imported by fake-backend protocol tests.
- Existing tests provide limited prior art for exercising routes with
  `httpx.ASGITransport`, plus model discovery and HTTP error propagation. Extend
  that public-boundary style while replacing assumptions tied to the upstream
  proxy.
- Before implementation, run focused target-HEF validation on `home` to
  discriminate model capability, prompt-template handling, resource setup, and
  protocol translation. Record the model, tested inputs/results, prompt
  findings, capacity, completion status, and cleanup observations.
- After each committed feature-branch update, synchronize that branch to the
  adapter checkout on `home` and run focused tests followed by the full project
  suite using the Hailo Apps environment. Never run project code, tests,
  linters, builds, or Python commands in the local Windows workspace.
- Hardware acceptance requires model discovery through `/api/show`, a streamed
  harmless schema-driven call, OpenClaw execution of that approved tool,
  submission of its result, and a grounded final answer. Do not merge to `main`
  until this and the required remote test suite pass.

## Out of Scope

- Changes to Hailo drivers, HailoRT, kernel modules, or other host/system-level
  components.
- Replacing or modifying the Hailo Apps framework itself.
- Adapter-side execution of OpenClaw tools or a hard-coded tool catalog.
- Vision, audio, embeddings, or other capabilities not implemented and
  validated for the configured model.
- Context caching, multi-process inference, or concurrent native generations
  in the initial implementation.
- Implicit model downloads, model pulling, or stopping unrelated services.
- Merging the feature branch to `main` before the remote tests and hardware
  acceptance checks pass.

## Further Notes

- The remote Hailo Apps checkout and its virtual environment were confirmed
  present and clean during plan synthesis. Setup should verify they remain
  usable rather than assume a fresh clone is necessary.
- The configured HEF, exact public model IDs, and model-specific prompt/tool
  behavior have not yet been established for the target validation. Existing
  research indicates that HailoRT supports native tool schemas but the
  installed binding requires fresh context for tools and system messages;
  prompt-template compatibility and practical context capacity remain hardware
  validation gates.
- The existing adapter's model discovery currently falls back to a fabricated
  model entry and claims GGUF format. Those behaviors must not be carried into
  truthful direct-backend discovery.
- The feature's single automated seam is public API behavior with a fake
  backend. Hardware validation is an explicit integration gate, not a reason
  to couple routine protocol tests to accelerator availability.