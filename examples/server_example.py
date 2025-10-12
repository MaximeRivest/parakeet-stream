#!/usr/bin/env python3
"""
Example: Parakeet WebSocket Server

Start a transcription server that accepts audio streams from clients.

Usage:
    python examples/server_example.py [--host HOST] [--port PORT] [--config CONFIG] [--device DEVICE]

Examples:
    # Basic server on localhost
    python examples/server_example.py

    # Server on all interfaces with GPU
    python examples/server_example.py --host 0.0.0.0 --port 8765 --device cuda

    # High quality preset
    python examples/server_example.py --config high_quality --device cuda
"""

import argparse
import logging
from parakeet_stream import ParakeetServer


def main():
    parser = argparse.ArgumentParser(description="Start Parakeet WebSocket transcription server")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port to listen on (default: 8765)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="balanced",
        choices=["maximum_quality", "high_quality", "balanced", "low_latency", "realtime", "ultra_realtime"],
        help="Parakeet quality preset (default: balanced)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device to run on (default: cpu)"
    )
    parser.add_argument(
        "--chunk-secs",
        type=float,
        default=None,
        help="Optional: Override chunk duration in seconds"
    )
    parser.add_argument(
        "--left-context-secs",
        type=float,
        default=None,
        help="Optional: Override left context duration in seconds"
    )
    parser.add_argument(
        "--right-context-secs",
        type=float,
        default=None,
        help="Optional: Override right context duration in seconds"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create and start server
    server = ParakeetServer(
        host=args.host,
        port=args.port,
        parakeet_config=args.config,
        device=args.device,
        chunk_secs=args.chunk_secs,
        left_context_secs=args.left_context_secs,
        right_context_secs=args.right_context_secs,
    )

    server.start()


if __name__ == "__main__":
    main()
