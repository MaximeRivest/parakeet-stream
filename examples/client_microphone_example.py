#!/usr/bin/env python3
"""
Example: Stream audio from microphone to Parakeet server

Stream live microphone audio to a Parakeet transcription server and
display transcriptions in real-time.

Usage:
    python examples/client_microphone_example.py [--server URL] [--duration SECONDS]

Examples:
    # Connect to local server
    python examples/client_microphone_example.py

    # Connect to remote server, record for 60 seconds
    python examples/client_microphone_example.py --server ws://192.168.1.100:8765 --duration 60

    # Infinite recording (Ctrl+C to stop)
    python examples/client_microphone_example.py --duration 0
"""

import argparse
import logging
from parakeet_stream import ParakeetClient


def main():
    parser = argparse.ArgumentParser(description="Stream microphone audio to Parakeet server")
    parser.add_argument(
        "--server",
        type=str,
        default="ws://localhost:8765",
        help="WebSocket server URL (default: ws://localhost:8765)"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="Recording duration in seconds (0 = infinite, default: 30)"
    )
    parser.add_argument(
        "--chunk-duration",
        type=float,
        default=1.0,
        help="Duration of each audio chunk to send (default: 1.0)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress transcription output (show only summary)"
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create client
    client = ParakeetClient(args.server)

    # Determine duration (None = infinite if 0)
    duration = None if args.duration == 0 else args.duration

    # Stream from microphone
    print(f"Connecting to {args.server}...")
    print()

    try:
        for segment in client.stream_microphone(
            duration=duration,
            chunk_duration=args.chunk_duration,
            verbose=not args.quiet
        ):
            # Segments are already printed if verbose=True
            pass

    except KeyboardInterrupt:
        print("\n✓ Stopped by user")


if __name__ == "__main__":
    main()
