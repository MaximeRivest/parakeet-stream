#!/usr/bin/env python3
"""
Live microphone transcription with overlapping window strategy.

This example shows how to use the OverlappingWindowStrategy for
live microphone transcription to improve quality at word boundaries.
"""
import argparse
import time

from parakeet_stream import Parakeet, OverlappingWindowStrategy


def main():
    parser = argparse.ArgumentParser(
        description="Live microphone transcription with overlapping windows",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--chunk",
        type=float,
        default=5.0,
        help="Chunk duration in seconds (default: 5.0)"
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=2.0,
        help="Overlap in seconds (default: 2.0)"
    )
    parser.add_argument(
        "--trim-words",
        type=int,
        default=2,
        help="Number of words to trim from edges (default: 2)"
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Speed factor (1.0=normal, 2.0=2x speed) (default: 1.0)"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output file to save transcript"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print transcriptions to console"
    )
    parser.add_argument(
        "--no-strategy",
        action="store_true",
        help="Disable strategy (use standard transcription)"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("LIVE MICROPHONE TRANSCRIPTION WITH OVERLAPPING WINDOW STRATEGY")
    print("=" * 70)
    print()

    # Initialize transcriber
    print("Loading Parakeet model...")
    pk = Parakeet()
    print()

    # Create strategy if not disabled
    strategy = None
    if not args.no_strategy:
        strategy = OverlappingWindowStrategy(
            chunk_duration=args.chunk,
            overlap=args.overlap,
            trim_words=args.trim_words,
            speed_factor=args.speed
        )

        print("Strategy Configuration:")
        print(f"  Chunk duration: {args.chunk}s")
        print(f"  Overlap: {args.overlap}s (on each side)")
        print(f"  Trim words: {args.trim_words} (from start/end)")
        print(f"  Speed factor: {args.speed}x")
        print()
        print("How it works:")
        print(f"  • Records {args.chunk}s chunks with {args.overlap}s overlap")
        print(f"  • Removes first/last {args.trim_words} words (may be cut off)")
        print("  • Merges clean middle sections")
        print()
    else:
        print("Using standard transcription (no strategy)")
        print()

    if args.output:
        print(f"Saving transcript to: {args.output}")
        print()

    print("─" * 70)
    print("Starting live transcription...")
    print("Speak into your microphone. Press Ctrl+C to stop.")
    print("─" * 70)
    print()

    try:
        # Start live transcription
        live = pk.listen(
            output=args.output,
            verbose=args.verbose,
            strategy=strategy
        )

        # Keep running until user stops
        while live.is_running:
            time.sleep(0.5)

            # Show periodic updates if not verbose
            if not args.verbose:
                stats = live.transcript.stats
                print(
                    f"\r[{stats['duration']:.1f}s] "
                    f"Segments: {stats['segments']} | "
                    f"Words: {stats['words']}",
                    end="",
                    flush=True
                )

    except KeyboardInterrupt:
        print("\n\nStopping...")
        live.stop()

    # Show final results
    print("\n")
    print("=" * 70)
    print("FINAL TRANSCRIPT")
    print("=" * 70)
    print()

    if live.transcript.segments:
        for i, segment in enumerate(live.transcript.segments, 1):
            print(f"{i}. [{segment.start_time:.1f}s - {segment.end_time:.1f}s]")
            print(f"   {segment.text}")
            if segment.confidence:
                print(f"   Confidence: {segment.confidence:.1%}")
            print()
    else:
        print("No transcription recorded.")
        print()

    # Statistics
    stats = live.transcript.stats
    print("─" * 70)
    print("STATISTICS")
    print("─" * 70)
    print(f"Total duration: {stats['duration']:.1f}s")
    print(f"Total segments: {stats['segments']}")
    print(f"Total words: {stats['words']}")
    if stats['avg_confidence'] > 0:
        print(f"Avg confidence: {stats['avg_confidence']:.1%}")
    print()

    if args.output:
        print(f"✓ Transcript saved to: {args.output}")


if __name__ == "__main__":
    main()
