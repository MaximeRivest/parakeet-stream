#!/usr/bin/env python3
"""
Compare different transcription strategies with Parakeet Stream.

This example demonstrates the overlapping window strategy which:
1. Chunks audio with large overlaps
2. Transcribes each chunk independently
3. Removes first/last N words (which may be cut off at boundaries)
4. Merges the clean middle sections

This can improve quality by avoiding word boundary issues.
"""
import time
from pathlib import Path

from parakeet_stream import Parakeet, OverlappingWindowStrategy


def main():
    print("=" * 70)
    print("TRANSCRIPTION STRATEGY COMPARISON")
    print("=" * 70)
    print()

    # Audio file to test
    audio_file = "2086-149220-0033.wav"
    if not Path(audio_file).exists():
        print(f"Error: Audio file '{audio_file}' not found")
        print("Please provide a path to a WAV file to test")
        return

    # Initialize transcriber
    print("Loading Parakeet model...")
    pk = Parakeet()
    print()

    # Strategy 1: Standard transcription (default)
    print("─" * 70)
    print("1. STANDARD TRANSCRIPTION")
    print("─" * 70)
    start = time.time()
    result_standard = pk.transcribe(audio_file)
    time_standard = time.time() - start

    print(f"Text: {result_standard.text}")
    print(f"Confidence: {result_standard.confidence:.2%}" if result_standard.confidence else "Confidence: N/A")
    print(f"Time: {time_standard:.2f}s")
    print()

    # Strategy 2: Overlapping windows (better quality at boundaries)
    print("─" * 70)
    print("2. OVERLAPPING WINDOW STRATEGY")
    print("─" * 70)
    print("Config: 5s chunks, 2s overlap, trim 2 words from edges")

    strategy = OverlappingWindowStrategy(
        chunk_duration=5.0,      # 5-second chunks
        overlap=2.0,              # 2 seconds overlap on each side
        trim_words=2,             # Remove first/last 2 words
        speed_factor=1.0          # Normal speed (can use 2.0 for 2x faster)
    )

    start = time.time()

    # Load audio manually to use with strategy
    import torch
    from parakeet_stream.utils import load_audio
    audio = load_audio(audio_file, pk.sample_rate)

    # Process with strategy
    segments = strategy.process_stream(
        audio.numpy(),
        pk,
        pk.sample_rate
    )

    time_overlap = time.time() - start

    # Combine segments
    result_text = " ".join(seg.text for seg in segments)

    print(f"Text: {result_text}")
    print(f"Segments: {len(segments)}")
    print(f"Time: {time_overlap:.2f}s")
    print()

    # Strategy 3: With speed factor (experimental)
    print("─" * 70)
    print("3. OVERLAPPING WINDOW + 1.5x SPEED (Experimental)")
    print("─" * 70)
    print("Config: Same as above but with 1.5x audio speed")
    print("Note: Speed > 1.0 may reduce quality but process faster")

    strategy_fast = OverlappingWindowStrategy(
        chunk_duration=5.0,
        overlap=2.0,
        trim_words=2,
        speed_factor=1.5  # 1.5x speed
    )

    start = time.time()
    segments_fast = strategy_fast.process_stream(
        audio.numpy(),
        pk,
        pk.sample_rate
    )
    time_fast = time.time() - start

    result_fast = " ".join(seg.text for seg in segments_fast)

    print(f"Text: {result_fast}")
    print(f"Segments: {len(segments_fast)}")
    print(f"Time: {time_fast:.2f}s")
    print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Standard:          {time_standard:.2f}s")
    print(f"Overlapping:       {time_overlap:.2f}s ({time_overlap/time_standard:.2f}x)")
    print(f"Overlapping 1.5x:  {time_fast:.2f}s ({time_fast/time_standard:.2f}x)")
    print()

    print("When to use each strategy:")
    print()
    print("• Standard: Fast, works well for continuous speech")
    print("• Overlapping: Better quality at word boundaries, good for")
    print("              choppy audio or when accuracy is critical")
    print("• Speed factor: Experimental - may reduce quality but faster")
    print()


if __name__ == "__main__":
    main()
