# Direct Hailo to OpenClaw Adapter

![HailoRT](https://img.shields.io/badge/HailoRT-5.3.0-success)
![Ollama](https://img.shields.io/badge/Ollama-0.6.0-blue)
![OpenClaw](https://img.shields.io/badge/OpenClaw-2026.04.20-orange)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Platform](https://img.shields.io/badge/Platform-Raspberry%20Pi%205-red)
![License](https://img.shields.io/badge/License-MIT-yellow)


A FastAPI adapter that exposes OpenAI- and Ollama-compatible endpoints and
runs text generation directly through HailoRT GenAI in the Hailo Apps runtime.

Built and tested on **Raspberry Pi 5 with Hailo-10H running Raspberry Pi
OS (Debian 13 "Trixie")**. Requires **Python 3.10 or newer**.

<p align="center">
  <img src="docs/images/openclaw-dashboard.jpg" alt="OpenClaw dashboard chatting with qwen3 via Hailo" width="800">
  <br>
  <em>OpenClaw dashboard chatting with a Hailo-10H accelerated model</em>
</p>

---

## What this adapter does

OpenClaw talks to language-model providers using the Ollama or OpenAI
wire protocols. This adapter serves those protocols on port 11435 and maps
public model IDs to installed HEFs from the Hailo Apps `agent` catalog:

- OpenClaw -> adapter (standard Ollama `/api/tags`, `/api/show`, `/api/chat`)
- adapter -> HailoRT GenAI (structured conversation and generation options)

No Hailo-Ollama server or port-8000 service is used. The native device and
model are initialized at application startup and released at shutdown.

---

## Runtime design

The adapter keeps model discovery and protocol handling separate from the
serialized native inference worker. Native Hailo imports are lazy, so fake
backends can exercise the HTTP API without an accelerator runtime installed.

- **Framework model discovery.** Installed LLM HEFs come from Hailo Apps'
  `agent` catalog; no model is downloaded implicitly. An empty usable catalog
  fails application startup, and unknown public IDs are rejected.
- **HEF-derived limits.** `/api/show` reports the loaded HEF's native
  `LLM.max_context_capacity()`, not the base model family's advertised context
  window. Hailo Apps does not provide context limits in its model catalog.
- **Single-owner inference.** Blocking native calls run on one executor thread.
  One request can wait by default; additional work gets a clear busy response.
- **Disconnect-safe ownership.** Cancelling or timing out an HTTP waiter does
  not cancel the native worker or release device ownership before completion.
- **Readiness and failure states.** `/readyz` reports initialization and
  inference health; initialization, generation, and cleanup failures are
  surfaced rather than returned as empty assistant text.
- **Explicit request limits.** Text-only models reject media and unsupported
  generation options. The complete model-rendered transcript and tool schemas
  are tokenized with the requested output allowance before generation; overflow
  returns HTTP 413 without dropping history. The default output allowance is
  256 tokens, and reaching it is reported as output exhaustion, not a normal
  stop.
- **Enforced tool choice.** `auto`, `none`, `required`, and a forced declared
  function are validated. Required and forced choices fail instead of returning
  ordinary text or a different executable call.
- **Installable package.** You can now `pip3 install` directly from
  GitHub and get a `hailo-ollama-adapter` command, rather than cloning
  and running `uvicorn` against a loose file.

HailoRT and Hailo Apps are supplied by the target system environment and are
not installed by this package's ordinary pip dependencies.

---

## Requirements

- **Raspberry Pi 5** with **Hailo-10H** accelerator (PCIe M.2 HAT)
- **Raspberry Pi OS (Debian 13 "Trixie")**, 64-bit
- **Python 3.10+** (ships with Trixie)
- **Hailo Apps virtual environment** with compatible HailoRT GenAI bindings
- An existing supported LLM HEF; chat requests never download models
- **OpenClaw** CLI installed (`pnpm add -g openclaw@2026.04.20`)

---

## Before you start

The adapter discovers installed LLM HEFs listed by the Hailo Apps `agent`
model catalog. It does not download models. By default, each discovered model's
Hailo Apps name is its public ID; set `HAILO_MODELS` only when you want custom
public IDs. OpenClaw onboarding should happen after the adapter is serving.

### 1. Prepare the Hailo Apps runtime

Use the Hailo Apps checkout and virtual environment that provide the matching
HailoRT GenAI Python bindings. HailoRT is installed by the Hailo platform
setup, not by pip installing this adapter. Activate that environment before
installing or running the adapter.

```bash
cd ~/hailo-apps
source venv_hailo_apps/bin/activate
```

### 2. Optionally configure public model IDs

`HAILO_MODELS` optionally maps the public ID OpenClaw uses to an existing HEF
listed by Hailo Apps' `agent` catalog for the detected architecture:

```bash
export HAILO_MODELS='{"qwen2.5-coder:1.5b":"/usr/local/hailo/resources/models/hailo10h/Qwen2.5-Coder-1.5B-Instruct.hef"}'
```

### 3. Install OpenClaw 2026.04.20

Later OpenClaw releases introduced breaking changes in the concurrency
handling and auth-profile schema. Pin to `2026.04.20` for a stable setup:

```bash
# npm
npm install -g openclaw@2026.04.20

# pnpm (recommended - fewer native-build hiccups)
pnpm add -g openclaw@2026.04.20
```

Verify:

```bash
openclaw --version
# should print 2026.04.20
```

If `pnpm` complains about missing channel plugin dependencies after
install, add them explicitly:

```bash
pnpm add -g @larksuiteoapi/node-sdk @buape/carbon grammy \
  @grammyjs/runner @grammyjs/transformer-throttler \
  @slack/web-api @slack/bolt @slack/logger nostr-tools

pnpm approve-builds -g
```

---

## Installation

Install the adapter into the Hailo Apps virtual environment so its HailoRT
bindings remain available:

```bash
cd ~/hailo-apps
source venv_hailo_apps/bin/activate

python -m pip install git+https://github.com/tishyk/hailo-ollama-openclaw-adapter.git
```

This installs the `hailo-ollama-adapter` command into that environment.

For a specific release:

```bash
python -m pip install git+https://github.com/tishyk/hailo-ollama-openclaw-adapter.git@2026.04.20
```

### Clone for development

If you want to modify the adapter or run tests:

```bash
git clone https://github.com/tishyk/hailo-ollama-openclaw-adapter.git
cd hailo-ollama-openclaw-adapter

source ~/hailo-apps/venv_hailo_apps/bin/activate
python -m pip install -e ".[dev]"
```

`-e` is editable mode - code changes take effect without reinstall.
`[dev]` adds `ruff`, `pytest`, `pytest-asyncio`, and `httpx` for tests.

---

## Running the adapter

At startup, the adapter discovers installed LLM HEFs from the Hailo Apps
`agent` catalog for the connected architecture. It loads those HEFs and reports
their actual context capacities from HailoRT. Startup fails if there are no
usable installed HEFs. `HAILO_MODELS` is optional and can map custom public IDs
to paths for cataloged HEFs; mapped models load during startup and are released
at shutdown.

```bash
export HAILO_MODELS='{"qwen2.5-coder:1.5b":"/usr/local/hailo/resources/models/hailo10h/Qwen2.5-Coder-1.5B-Instruct.hef"}'
```

Set `HAILO_MODELS` only if custom public IDs are needed. Set
`HAILO_QUEUE_SIZE` if desired, then run any of these (they're all equivalent):

```bash
source ~/hailo-apps/venv_hailo_apps/bin/activate

# Simplest
hailo-ollama-adapter

# Python module form
python -m hailo_ollama_adapter

# Raw uvicorn (for custom uvicorn flags)
uvicorn hailo_ollama_adapter.adapter:app --host 0.0.0.0 --port 11435
```

All three default to binding `0.0.0.0:11435` with a 240-second keep-alive.

Leave the terminal open while using OpenClaw. Press `Ctrl+C` to stop the
adapter when done.

### CLI options

```bash
hailo-ollama-adapter --help

  --host HOST                        default: 0.0.0.0
  --port PORT                        default: 11435
  --timeout-keep-alive SECONDS       default: 240
  --limit-concurrency N              default: 2 HTTP connections
  --queue-size N                     override HAILO_QUEUE_SIZE (default: 1)
  --log-level LEVEL                  default: info
  --reload                           auto-reload on source changes (dev)
```

### Verify it's working

In a second terminal:

```bash
# Should list installed HEFs from the Hailo Apps agent catalog
curl -s http://127.0.0.1:11435/api/tags | python3 -m json.tool

# Native device/model initialization must succeed for readiness
curl -i http://127.0.0.1:11435/readyz
```

When `HAILO_MODELS` is set, the ID on the left is what OpenClaw sees and must
send on chat requests. Without it, model IDs are the Hailo Apps catalog names.
Only installed models listed for the Hailo Apps `agent` app are exposed.
`/api/tags/refresh` re-reports the models loaded at startup; restart after
installing another HEF:

```bash
curl -s -X POST http://127.0.0.1:11435/api/tags/refresh
```

---

## Configuring OpenClaw

Before running onboarding, make sure:

- The adapter is running inside the Hailo Apps virtual environment
- `HAILO_MODELS` maps a custom public ID to a cataloged installed HEF (optional)
- `GET /readyz` returns `200`

Open a **second terminal** and run OpenClaw's interactive onboarding.
This wires up the gateway, the adapter endpoint, and your default model.

```bash
openclaw onboard --install-daemon
```

### Step 1 - Accept the security prompt and pick QuickStart

Type `Yes` to accept the personal-by-default prompt, then choose
**QuickStart** as the setup mode. If an existing config is detected,
select **Update values**.

<p align="center">
  <img src="docs/images/setup-quickstart.jpg" alt="OpenClaw QuickStart setup" width="800">
</p>

### Step 2 - Pick Ollama as the model provider

Arrow down to **Ollama (Cloud and local open models)** and hit Enter.

<p align="center">
  <img src="docs/images/setup-provider-ollama.jpg" alt="Selecting Ollama as provider" width="700">
</p>

### Step 3 - Point to the adapter and pick a default model

- **Ollama mode**: `Local only`
- **Ollama base URL**: `http://127.0.0.1:11435` (the adapter)
- **Default model**: pick a discovered Hailo Apps model ID.

<p align="center">
  <img src="docs/images/setup-model-picker.jpg" alt="Model picker with live Hailo models" width="900">
</p>

The right-hand pane shows the adapter serving `/api/tags` and `/api/show`
requests as OpenClaw probes it. 200 OKs everywhere = healthy.

### Step 4 - Finish onboarding and open the dashboard

When onboarding finishes, you'll see the completion screen with the
dashboard link.

<p align="center">
  <img src="docs/images/setup-complete.jpg" alt="Onboarding complete screen" width="700">
</p>

Open the dashboard:

```bash
openclaw dashboard
```

### Step 5 - Chat with your Hailo-accelerated model

The dashboard opens in your browser. Start chatting with the public model ID
you configured, or the discovered Hailo Apps model name when no override was
configured.

<p align="center">
  <img src="docs/images/openclaw-dashboard.jpg" alt="OpenClaw dashboard in action" width="900">
</p>

### Daily use

Once configured, each time you want to use OpenClaw with Hailo:

```bash
# 1. Activate the Hailo Apps environment and start the adapter
cd ~/hailo-apps
source venv_hailo_apps/bin/activate
hailo-ollama-adapter

# 2. Terminal 2: open the dashboard
openclaw dashboard
```

Press `Ctrl+C` in the adapter terminal when you're done.

---

## Endpoints

| Method | Path                              | Purpose                                    |
|--------|-----------------------------------|--------------------------------------------|
| GET    | `/api/tags`                       | Installed Hailo Apps agent HEF list        |
| POST   | `/api/tags/refresh`               | Re-list HEFs loaded at startup             |
| POST   | `/api/show`                       | Ollama model details                       |
| GET    | `/readyz`                         | Native backend readiness                   |
| POST   | `/api/chat`                       | Ollama chat endpoint                       |
| POST   | `/chat/completions`               | OpenAI-compatible chat endpoint            |
| POST   | `/v1/chat/completions`            | OpenAI-compatible chat (alt path)          |
| POST   | `/api/chat/completions`           | OpenAI-compatible chat (alt path)          |

All chat endpoints honor the public `model` ID in the request body. The
adapter resolves it to the configured local HEF path for inference and keeps
the public ID in client-facing responses.

---

## Configuration knobs

Optionally map custom public IDs to cataloged HEF paths with `HAILO_MODELS`.
`HAILO_QUEUE_SIZE` controls how many requests may wait behind the active
generation (default 1).
Inference itself is always serialized because the Hailo device and model
context have one owner. Request deadlines are 180 seconds; a timed-out client
does not cancel native work or make the device available early.

```bash
HAILO_MODELS='{"qwen2.5-coder:1.5b":"/path/to/Qwen2.5-Coder-1.5B-Instruct.hef"}'
HAILO_QUEUE_SIZE=1
```

The queue rejects excess requests with `503`; the backend reports initialization
or native-generation failures at `/readyz`. Model discovery remains available
while inference runs.

---

## Troubleshooting

**OpenClaw dashboard shows no models** - Check that Hailo Apps detects the
connected architecture, that the model is listed for its `agent` app, and that
the corresponding HEF exists in the Hailo Apps resources directory. If
`HAILO_MODELS` is set, confirm it points to an installed cataloged HEF. Restart
the adapter after correcting the catalog or installing the model.

**`/readyz` returns `503`** - Check the adapter log for a HailoRT/Hailo Apps
initialization or generation failure, verify the Hailo Apps catalog contains
an installed LLM HEF and check `HAILO_QUEUE_SIZE` is at least 1. Restart
the adapter after correcting the failure; hardware inference is not retried
automatically.

**`[Bootstrap pending]` scaffolding keeps appearing in replies** - Delete
`~/.openclaw/workspace/BOOTSTRAP.md` after onboarding. OpenClaw treats
the file's presence as "bootstrap pending".

**Small model (1.7B) can't hold persona or remember names well** - This
is the model's capability limit, not an adapter bug. Use a larger Hailo
model or switch to a cloud provider for agent-heavy work, keep the Hailo
model for quick chat.

---

## Development

```bash
# Lint
ruff check src/

# Auto-fix style issues
ruff check --fix src/

# Run tests
pytest
```

The configured Ruff rule sets are:
`E`, `W`, `F`, `I`, `N`, `UP`, `B`, `C4`, `SIM`, `G`.

---

## Versioning

When bumping the version, update **three** files in lockstep:

- `pyproject.toml` -> `version = "X.Y.Z"`
- `setup.py` -> `version="X.Y.Z"`
- `src/hailo_ollama_adapter/__init__.py` -> `__version__ = "X.Y.Z"`

Tag the release:

```bash
git tag -a vX.Y.Z -m "Release X.Y.Z"
git push origin vX.Y.Z
```

---

## License

MIT - see [LICENSE](LICENSE).

---

## Acknowledgments

- [Hailo](https://hailo.ai/) for the Hailo-10H accelerator, HailoRT, and
  Hailo Apps runtime
- [OpenClaw](https://openclaw.ai/) for the local-first agent framework
- [FastAPI](https://fastapi.tiangolo.com/) and
  [uvicorn](https://www.uvicorn.org/) for the HTTP stack
