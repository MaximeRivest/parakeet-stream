#!/usr/bin/env python3
"""
Example: Stream audio file to Parakeet server

Stream an audio file to a Parakeet transcription server and
display the transcription.

Usage:
    python examples/client_file_example.py <audio_file> [--server URL] [--realtime]

Examples:
    # Transcribe local file
    python examples/client_file_example.py audio.wav

    # Connect to remote server
    python examples/client_file_example.py audio.wav --server ws://192.168.1.100:8765

    # Simulate real-time streaming
    python examples/client_file_example.py audio.wav --realtime
"""

import argparse
import logging
from parakeet_stream import ParakeetClient


def main():
    parser = argparse.ArgumentParser(description="Stream audio file to Parakeet server")
    parser.add_argument(
        "audio_file",
        type=str,
        help="Path to audio file"
    )
    parser.add_argument(
        "--server",
        type=str,
        default="ws://localhost:8765",
        help="WebSocket server URL (default: ws://localhost:8765)"
    )
    parser.add_argument(
        "--chunk-duration",
        type=float,
        default=2.0,
        help="Duration of each audio chunk to send (default: 2.0)"
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Simulate real-time streaming (sleep between chunks)"
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

    # Stream file
    print(f"Connecting to {args.server}...")
    print()

    segments = []
    for segment in client.stream_file(
        audio_path=args.audio_file,
        chunk_duration=args.chunk_duration,
        realtime=args.realtime,
        verbose=not args.quiet
    ):
        segments.append(segment['text'])

    # Print full transcript
    if not args.quiet:
        print("\n" + "=" * 60)
        print("FULL TRANSCRIPT:")
        print("=" * 60)
        print(" ".join(segments))


if __name__ == "__main__":
    main()
