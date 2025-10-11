"""
Example of streaming transcription with Parakeet Stream
"""
from parakeet_stream import StreamingTranscriber, TranscriberConfig


def main():
    # Configure for streaming with custom chunk size
    config = TranscriberConfig(
        model_name="nvidia/parakeet-tdt-0.6b-v3",
        device="cpu",  # Use "cuda" for GPU
        chunk_secs=2.0,  # 2-second chunks for low latency
        left_context_secs=10.0,  # 10 seconds of left context for quality
        right_context_secs=2.0,  # 2 seconds of right context
        streaming=True,
    )

    transcriber = StreamingTranscriber(config=config)

    # Stream transcription results
    audio_file = "audio.wav"
    print("Streaming transcription:")
    print("-" * 50)

    for chunk in transcriber.stream(audio_file):
        # Print incremental results
        print(f"[{chunk.timestamp_start:.2f}s - {chunk.timestamp_end:.2f}s]: {chunk.text}")

        if chunk.is_final:
            print("-" * 50)
            print("Final transcription:")
            print(chunk.text)


if __name__ == "__main__":
    main()
