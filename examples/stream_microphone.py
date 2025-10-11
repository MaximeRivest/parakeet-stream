#!/usr/bin/env python3
"""
Real-time microphone transcription with Parakeet Stream

This script captures audio from your microphone and transcribes it in real-time,
displaying the results on screen and optionally saving to a file.

Requirements:
    pip install sounddevice

Usage:
    python stream_microphone.py
    python stream_microphone.py --output transcription.txt
    python stream_microphone.py --device cuda --chunk 1.0
"""
import argparse
import datetime
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import torch

from parakeet_stream import StreamingTranscriber, TranscriberConfig


class MicrophoneStreamer:
    """Real-time microphone transcription with buffering."""

    def __init__(
        self,
        transcriber: StreamingTranscriber,
        sample_rate: int = 16000,
        chunk_duration: float = 2.0,
        output_file: str = None,
    ):
        """Initialize microphone streamer.

        Args:
            transcriber: StreamingTranscriber instance
            sample_rate: Audio sample rate in Hz
            chunk_duration: Duration of audio chunks to accumulate before transcribing
            output_file: Optional file path to save transcriptions
        """
        self.transcriber = transcriber
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_samples = int(sample_rate * chunk_duration)
        self.output_file = output_file

        # Audio buffer queue
        self.audio_queue = queue.Queue()
        self.is_running = False

        # Accumulator for audio chunks
        self.audio_buffer = []
        self.buffer_lock = threading.Lock()

        # Transcription state
        self.last_transcription = ""
        self.full_transcription = []

    def audio_callback(self, indata, frames, time_info, status):
        """Callback for sounddevice to capture audio.

        Args:
            indata: Input audio data
            frames: Number of frames
            time_info: Time information
            status: Status flags
        """
        if status:
            print(f"Audio callback status: {status}", file=sys.stderr)

        # Copy audio data and add to queue
        audio_chunk = indata.copy().flatten()
        self.audio_queue.put(audio_chunk)

    def process_audio_stream(self):
        """Process audio chunks from the queue and transcribe them."""
        print("Starting audio processing thread...")

        while self.is_running:
            try:
                # Get audio chunk from queue (with timeout to check is_running)
                audio_chunk = self.audio_queue.get(timeout=0.1)

                with self.buffer_lock:
                    self.audio_buffer.append(audio_chunk)
                    current_length = sum(len(chunk) for chunk in self.audio_buffer)

                    # When we have enough audio, transcribe it
                    if current_length >= self.chunk_samples:
                        # Concatenate all chunks
                        audio_array = np.concatenate(self.audio_buffer)

                        # Take exactly chunk_samples
                        audio_to_process = audio_array[:self.chunk_samples]
                        remainder = audio_array[self.chunk_samples:]

                        # Reset buffer with remainder
                        self.audio_buffer = [remainder] if len(remainder) > 0 else []

                        # Transcribe
                        self.transcribe_chunk(audio_to_process)

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error processing audio: {e}", file=sys.stderr)

    def transcribe_chunk(self, audio_array: np.ndarray):
        """Transcribe a chunk of audio.

        Args:
            audio_array: Audio data as numpy array
        """
        try:
            # Convert to tensor
            audio_tensor = torch.from_numpy(audio_array).float()

            # Transcribe using the simple transcribe method (faster for short chunks)
            result = self.transcriber.transcribe(audio_tensor)

            # Update transcription
            new_text = result.text.strip()

            if new_text and new_text != self.last_transcription:
                timestamp = datetime.datetime.now().strftime("%H:%M:%S")

                # Display on screen
                print(f"[{timestamp}] {new_text}")

                # Save to file if specified
                if self.output_file:
                    with open(self.output_file, 'a', encoding='utf-8') as f:
                        f.write(f"[{timestamp}] {new_text}\n")

                # Update state
                self.last_transcription = new_text
                self.full_transcription.append((timestamp, new_text))

        except Exception as e:
            print(f"Error transcribing: {e}", file=sys.stderr)

    def start(self):
        """Start microphone streaming and transcription."""
        print("=" * 70)
        print("REAL-TIME MICROPHONE TRANSCRIPTION")
        print("=" * 70)
        print(f"Sample rate: {self.sample_rate} Hz")
        print(f"Chunk duration: {self.chunk_duration}s")
        print(f"Device: {self.transcriber.config.device}")
        if self.output_file:
            print(f"Output file: {self.output_file}")
        print("-" * 70)
        print("Press Ctrl+C to stop...")
        print("=" * 70)
        print()

        self.is_running = True

        # Start audio processing thread
        processing_thread = threading.Thread(target=self.process_audio_stream, daemon=True)
        processing_thread.start()

        try:
            # Start audio stream
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype='float32',
                callback=self.audio_callback,
                blocksize=int(self.sample_rate * 0.1),  # 100ms blocks
            ):
                print("🎤 Listening... (speak into your microphone)")
                print()

                # Keep running until interrupted
                while self.is_running:
                    time.sleep(0.1)

        except KeyboardInterrupt:
            print("\n\n" + "=" * 70)
            print("Stopping...")
        finally:
            self.stop()

    def stop(self):
        """Stop microphone streaming."""
        self.is_running = False
        print("=" * 70)
        print("TRANSCRIPTION COMPLETE")
        print("=" * 70)

        if self.full_transcription:
            print("\nFull transcription:")
            print("-" * 70)
            for timestamp, text in self.full_transcription:
                print(f"[{timestamp}] {text}")

        if self.output_file:
            print(f"\nTranscription saved to: {self.output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Real-time microphone transcription with Parakeet Stream",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output file to save transcriptions (default: screen only)"
    )
    parser.add_argument(
        "--chunk",
        type=float,
        default=2.0,
        help="Chunk duration in seconds (default: 2.0)"
    )
    parser.add_argument(
        "--left",
        type=float,
        default=10.0,
        help="Left context in seconds (default: 10.0)"
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
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio devices and exit"
    )

    args = parser.parse_args()

    # List audio devices if requested
    if args.list_devices:
        print("Available audio devices:")
        print(sd.query_devices())
        return 0

    # Clear output file if it exists
    if args.output and Path(args.output).exists():
        response = input(f"File '{args.output}' exists. Overwrite? [y/N]: ")
        if response.lower() != 'y':
            print("Aborted.")
            return 1
        Path(args.output).unlink()

    # Create configuration
    config = TranscriberConfig(
        model_name="nvidia/parakeet-tdt-0.6b-v3",
        device=args.device,
        chunk_secs=args.chunk,
        left_context_secs=args.left,
        right_context_secs=args.right,
    )

    print("Initializing transcriber (this may take a moment)...")
    transcriber = StreamingTranscriber(config=config)

    # Create and start streamer
    streamer = MicrophoneStreamer(
        transcriber=transcriber,
        sample_rate=16000,
        chunk_duration=args.chunk,
        output_file=args.output,
    )

    try:
        streamer.start()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
