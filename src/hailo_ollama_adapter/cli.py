"""Command-line entry point for the adapter."""

from __future__ import annotations

import argparse
import os

import uvicorn


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hailo-ollama-adapter",
        description="Direct HailoRT server for Ollama- and OpenAI-compatible APIs.",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Interface to bind (default: 0.0.0.0).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=11435,
        help="Port to listen on (default: 11435).",
    )
    parser.add_argument(
        "--timeout-keep-alive",
        type=int,
        default=240,
        help="HTTP keep-alive timeout in seconds (default: 240).",
    )
    parser.add_argument(
        "--limit-concurrency",
        type=int,
        default=2,
        help="Max concurrent HTTP connections; inference remains serialized (default: 2).",
    )
    parser.add_argument(
        "--queue-size",
        type=int,
        default=None,
        help="Waiting inference requests (default: HAILO_QUEUE_SIZE or 1).",
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["critical", "error", "warning", "info", "debug", "trace"],
        help="Uvicorn log level (default: info).",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload on source changes (development only).",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.queue_size is not None and args.queue_size < 1:
        parser.error("--queue-size must be at least 1")
    if args.queue_size is not None:
        os.environ["HAILO_QUEUE_SIZE"] = str(args.queue_size)
    uvicorn.run(
        "hailo_ollama_adapter.adapter:app",
        host=args.host,
        port=args.port,
        timeout_keep_alive=args.timeout_keep_alive,
        limit_concurrency=args.limit_concurrency,
        log_level=args.log_level,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
