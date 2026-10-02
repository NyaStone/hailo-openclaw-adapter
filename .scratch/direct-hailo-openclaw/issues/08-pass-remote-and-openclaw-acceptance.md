# 08: Pass remote and OpenClaw acceptance

**What to build:** Validate the committed feature branch in the Hailo Apps runtime and through OpenClaw before it is eligible to merge.

**Blocked by:** 02: Expose configured model discovery; 03: Preserve conversations through the public APIs; 04: Serve requests through the direct Hailo backend; 05: Complete the non-streaming tool handshake; 06: Stream text and tool-call responses; 07: Enforce request and generation limits.

**Status:** ready-for-agent

- [ ] Sync the committed feature branch to the adapter checkout in Hailo Apps on `home` and record the tested commit.
- [ ] Run focused project tests followed by the full project suite using the Hailo Apps environment; record commands and outcomes.
- [ ] Verify model discovery through `/api/show`, then stream a harmless schema-driven call through OpenClaw.
- [ ] Have OpenClaw execute the approved harmless tool, submit its synthetic result, and verify a grounded final answer.
- [ ] Do not merge the feature branch to `main` unless the remote test suite and required hardware acceptance checks pass.
