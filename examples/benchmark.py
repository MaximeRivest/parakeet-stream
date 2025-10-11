"""
Parameterized testing script for parakeet-stream library

This script allows testing different streaming configurations matching
the NVIDIA reference implementation.

Recommended settings from NVIDIA:
- Long file transcription: 10-10-5 (10s left, 10s chunk, 5s right) - similar to offline
- Streaming with 4s latency: 10-2-2 (usually similar or better than 10-0.16-3.84)
"""
import argparse
import time
from pathlib import Path

from parakeet_stream import StreamingTranscriber, TranscriberConfig


def test_configuration(
    audio_file: str,
    left_context_secs: float = 10.0,
    chunk_secs: float = 2.0,
    right_context_secs: float = 2.0,
    device: str = "cpu",
    verbose: bool = True,
):
    """Test a specific streaming configuration.

    Args:
        audio_file: Path to audio file
        left_context_secs: Left context window in seconds
        chunk_secs: Chunk size in seconds (affects latency)
        right_context_secs: Right context window in seconds (affects latency)
        device: Device to run on (cpu, cuda, mps)
        verbose: Whether to print detailed output

    Returns:
        Dictionary with results
    """
    # Calculate theoretical latency
    theoretical_latency = chunk_secs + right_context_secs

    if verbose:
        print("=" * 70)
        print(f"Testing Configuration: {left_context_secs}-{chunk_secs}-{right_context_secs}")
        print("=" * 70)
        print(f"  Left context:   {left_context_secs}s")
        print(f"  Chunk size:     {chunk_secs}s")
        print(f"  Right context:  {right_context_secs}s")
        print(f"  Theoretical latency: {theoretical_latency}s")
        print(f"  Device: {device}")
        print("-" * 70)

    # Create configuration
    config = TranscriberConfig(
        model_name="nvidia/parakeet-tdt-0.6b-v3",
        device=device,
        chunk_secs=chunk_secs,
        left_context_secs=left_context_secs,
        right_context_secs=right_context_secs,
        streaming=True,
    )

    # Initialize transcriber
    transcriber = StreamingTranscriber(config=config)

    # Test streaming
    start_time = time.time()
    chunks = []
    final_text = ""

    for chunk in transcriber.stream(audio_file):
        chunks.append(chunk)
        if chunk.is_final:
            final_text = chunk.text

    elapsed_time = time.time() - start_time

    # Calculate stats
    num_chunks = len(chunks)

    if verbose:
        print(f"\nResults:")
        print(f"  Total chunks: {num_chunks}")
        print(f"  Elapsed time: {elapsed_time:.2f}s")
        print(f"  Final transcription:")
        print(f"    '{final_text}'")
        print("=" * 70)
        print()

    return {
        "config": f"{left_context_secs}-{chunk_secs}-{right_context_secs}",
        "theoretical_latency": theoretical_latency,
        "num_chunks": num_chunks,
        "elapsed_time": elapsed_time,
        "transcription": final_text,
    }


def run_benchmark(audio_file: str, device: str = "cpu"):
    """Run benchmark with multiple configurations.

    Args:
        audio_file: Path to audio file
        device: Device to run on
    """
    print("\n" + "=" * 70)
    print("PARAKEET STREAM - CONFIGURATION BENCHMARK")
    print("=" * 70)
    print(f"Audio file: {audio_file}")
    print(f"Device: {device}")
    print()

    # Test configurations (left-chunk-right)
    configurations = [
        # Long file transcription (high quality)
        {"left_context_secs": 10.0, "chunk_secs": 10.0, "right_context_secs": 5.0,
         "name": "Long file (10-10-5)"},

        # Balanced streaming (recommended)
        {"left_context_secs": 10.0, "chunk_secs": 2.0, "right_context_secs": 2.0,
         "name": "Streaming 4s latency (10-2-2)"},

        # Low latency streaming
        {"left_context_secs": 10.0, "chunk_secs": 1.0, "right_context_secs": 1.0,
         "name": "Low latency 2s (10-1-1)"},

        # Very low latency
        {"left_context_secs": 10.0, "chunk_secs": 0.5, "right_context_secs": 0.5,
         "name": "Very low latency 1s (10-0.5-0.5)"},
    ]

    results = []
    for i, cfg in enumerate(configurations, 1):
        print(f"\n[{i}/{len(configurations)}] {cfg['name']}")
        result = test_configuration(
            audio_file=audio_file,
            left_context_secs=cfg["left_context_secs"],
            chunk_secs=cfg["chunk_secs"],
            right_context_secs=cfg["right_context_secs"],
            device=device,
            verbose=True,
        )
        results.append({**result, "name": cfg["name"]})

    # Print summary
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"{'Configuration':<30} {'Latency':<12} {'Chunks':<10} {'Time (s)':<10}")
    print("-" * 70)
    for result in results:
        print(
            f"{result['name']:<30} "
            f"{result['theoretical_latency']:<12.2f} "
            f"{result['num_chunks']:<10} "
            f"{result['elapsed_time']:<10.2f}"
        )
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Test parakeet-stream with different configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run benchmark with all configurations
  python test_parameterized.py --audio 2086-149220-0033.wav --benchmark

  # Test specific configuration (left-chunk-right)
  python test_parameterized.py --audio 2086-149220-0033.wav --left 10 --chunk 2 --right 2

  # Test on GPU
  python test_parameterized.py --audio 2086-149220-0033.wav --device cuda --benchmark
        """
    )

    parser.add_argument(
        "--audio",
        type=str,
        required=True,
        help="Path to audio file"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark with multiple configurations"
    )
    parser.add_argument(
        "--left",
        type=float,
        default=10.0,
        help="Left context in seconds (default: 10.0)"
    )
    parser.add_argument(
        "--chunk",
        type=float,
        default=2.0,
        help="Chunk size in seconds (default: 2.0)"
    )
    parser.add_argument(
        "--right",
        type=float,
        default=2.0,
        help="Right context in seconds (default: 2.0)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device to run on (default: cpu)"
    )

    args = parser.parse_args()

    # Check if audio file exists
    if not Path(args.audio).exists():
        print(f"Error: Audio file not found: {args.audio}")
        return 1

    if args.benchmark:
        run_benchmark(args.audio, args.device)
    else:
        test_configuration(
            audio_file=args.audio,
            left_context_secs=args.left,
            chunk_secs=args.chunk,
            right_context_secs=args.right,
            device=args.device,
            verbose=True,
        )

    return 0


if __name__ == "__main__":
    exit(main())
