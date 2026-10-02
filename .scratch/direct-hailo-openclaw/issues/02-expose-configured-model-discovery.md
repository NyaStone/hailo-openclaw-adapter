# 02: Expose configured model discovery

**What to build:** Let OpenClaw discover only explicitly mapped, validated models and receive accurate details for each available model.

**Blocked by:** 01: Validate the target HEF contract on home.

**Status:** ready-for-agent

- [ ] Configure explicit public model IDs mapped to usable HEFs; chat requests do not download or pull models implicitly.
- [ ] Model listing includes only configured usable models, with no fabricated fallback entries.
- [ ] Model detail responses report truthful model format, context capacity, and only validated capabilities.
- [ ] Unknown model IDs return a not-found error instead of silently selecting another model.
- [ ] Discovery probes remain responsive while inference is active.
