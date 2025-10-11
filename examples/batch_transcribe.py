"""
Example of batch transcription with timestamps
"""
from pathlib import Path

from parakeet_stream import StreamingTranscriber
from parakeet_stream.utils import format_timestamp


def main():
    # Initialize transcriber with timestamp support
    transcriber = StreamingTranscriber(
        model_name="nvidia/parakeet-tdt-0.6b-v3",
        device="cpu",
    )

    # Get all audio files in a directory
    audio_dir = Path("audio_files")
    audio_files = list(audio_dir.glob("*.wav"))

    if not audio_files:
        print(f"No audio files found in {audio_dir}")
        return

    # Transcribe all files
    results = transcriber.transcribe_batch(
        audio_files,
        timestamps=True,
        show_progress=True,
    )

    # Print results with timestamps
    for audio_file, result in zip(audio_files, results):
        print(f"\n{audio_file.name}:")
        print(f"Transcription: {result.text}")

        if result.timestamps:
            print("Word-level timestamps:")
            for word_info in result.timestamps[:10]:  # Show first 10 words
                start = format_timestamp(word_info['start'])
                end = format_timestamp(word_info['end'])
                word = word_info['word']
                print(f"  [{start} - {end}]: {word}")


if __name__ == "__main__":
    main()
