"""
Simple example of transcribing an audio file with Parakeet Stream
"""
from parakeet_stream import StreamingTranscriber


def main():
    # Initialize transcriber (model will be loaded on first use)
    transcriber = StreamingTranscriber(
        model_name="nvidia/parakeet-tdt-0.6b-v3",
        device="cpu",  # Use "cuda" for GPU
    )

    # Transcribe a single file
    # Use the sample audio file from the repository, or provide your own
    audio_file = "2086-149220-0033.wav"  # Update this to your audio file path
    result = transcriber.transcribe(audio_file)

    print(f"Transcription: {result.text}")


if __name__ == "__main__":
    main()
