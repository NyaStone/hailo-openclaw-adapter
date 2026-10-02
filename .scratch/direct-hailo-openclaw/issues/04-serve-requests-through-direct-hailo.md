# 04: Serve requests through the direct Hailo backend

**What to build:** Answer text requests through HailoRT using one serialized owner of inference resources while keeping the protocol service responsive.

**Blocked by:** 02: Expose configured model discovery; 03: Preserve conversations through the public APIs.

**Status:** claimed

- [ ] The configured public model ID selects its mapped HEF and native generation returns a text response through both public APIs.
- [ ] Native imports remain lazy for fake-backend protocol use; resources initialize and release through application lifespan.
- [ ] Blocking native generation runs away from the async event loop, and one worker serializes device and context access.
- [ ] Queueing is bounded and overload returns a clear busy response; readiness and failed states are explicit.
- [ ] Client disconnect or timeout does not release native ownership before completion or verified cancellation; backend failures and shutdown have defined cleanup behavior.
- [ ] CLI, runtime configuration, and documentation describe the direct Hailo Apps runtime, model mapping, readiness, and hardware concurrency without requiring Hailo-Ollama.
