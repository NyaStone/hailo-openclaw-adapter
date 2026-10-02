# Issue tracker: Local Markdown

Issues and specs for this repo live as markdown files in `.scratch/`.

## Conventions

- One feature per directory: `.scratch/<feature-slug>/`
- Spec: `.scratch/<feature-slug>/spec.md`
- Tickets: `.scratch/<feature-slug>/issues/<NN>-<slug>.md`, numbered from `01`, one file per ticket.
- Triage state is a `Status:` line near the top of each ticket; see `triage-labels.md`.
- Append discussion under `## Comments`.

When publishing, create a file under `.scratch/<feature-slug>`. When fetching a ticket, read its referenced path or issue number.

## Wayfinding

- Map: `.scratch/<effort>/map.md`, containing Notes, Decisions-so-far, and Fog.
- Child tickets: `.scratch/<effort>/issues/NN-<slug>.md`, with a `Type:` (`research`/`prototype`/`grilling`/`task`) and `Status:` (`claimed`/`resolved`) line.
- Record blockers as `Blocked by: NN, NN`; a ticket is unblocked when all listed tickets are resolved.
- The frontier is the first-numbered open, unblocked, unclaimed ticket.
- Claim by setting `Status: claimed` before work. Resolve by adding an `## Answer`, setting `Status: resolved`, and adding a gist + link context pointer to the map's Decisions-so-far.