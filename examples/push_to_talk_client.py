#!/usr/bin/env python3
"""
Push-to-talk transcription client.

Records audio while you hold a key, then sends it all at once for transcription.
Perfect for keyboard shortcuts!

Usage:
    # Interactive mode
    python push_to_talk_client.py

    # One-shot mode (for keyboard shortcuts)
    python push_to_talk_client.py --oneshot

    # Copy to clipboard
    python push_to_talk_client.py --oneshot --clipboard
"""

import asyncio
import json
import sys
from typing import Optional

import numpy as np
from parakeet_stream import Microphone


class PushToTalkClient:
    """Simple client: record → send → get transcription."""

    def __init__(self, server_url: str = 'ws://192.168.2.24:8765'):
        self.server_url = server_url

    async def transcribe(self, audio: np.ndarray) -> str:
        """Send audio to server and get transcription."""
        import websockets

        async with websockets.connect(self.server_url) as websocket:
            # Wait for handshake
            handshake = await websocket.recv()
            handshake_data = json.loads(handshake)

            if handshake_data.get("type") != "ready":
                raise RuntimeError(f"Server not ready: {handshake_data}")

            # Send audio
            audio_int16 = (audio * 32768.0).astype(np.int16)
            await websocket.send(audio_int16.tobytes())

            # Send flush
            await websocket.send(json.dumps({"type": "flush"}))

            # Collect results
            text_parts = []
            async for message in websocket:
                data = json.loads(message)
                if data.get("type") == "segment":
                    text_parts.append(data.get("text", ""))
                elif data.get("type") == "flushed":
                    break

            return " ".join(text_parts).strip()

    def record_and_transcribe(self, duration: Optional[float] = None) -> str:
        """
        Record audio and transcribe it.

        Args:
            duration: Recording duration. If None, records until Enter is pressed.

        Returns:
            Transcribed text
        """
        mic = Microphone()

        if duration:
            # Fixed duration
            print(f"🎤 Recording for {duration}s...")
            clip = mic.record(duration=duration)
        else:
            # Record until Enter (with safety limit)
            print("🎤 Recording... Press Enter to stop.")
            print("   (Auto-stops at 9.5 minutes for safety)")
            import sounddevice as sd
            import threading
            import time

            recording = []
            sample_rate = 16000
            start_time = time.time()
            stop_event = threading.Event()
            max_duration = 9.5 * 60  # 9.5 minutes

            def callback(indata, frames, time, status):
                recording.append(indata.copy())

            # Start recording
            stream = sd.InputStream(
                samplerate=sample_rate,
                channels=1,
                callback=callback
            )
            stream.start()

            # Monitor thread for duration limit
            def monitor_duration():
                while not stop_event.is_set():
                    elapsed = time.time() - start_time
                    if elapsed >= max_duration:
                        print(f"\n⚠️  Max duration ({max_duration/60:.1f} min) reached! Auto-stopping...")
                        stop_event.set()
                        break
                    time.sleep(0.5)

            monitor_thread = threading.Thread(target=monitor_duration, daemon=True)
            monitor_thread.start()

            # Wait for Enter or timeout
            try:
                import select
                import sys
                # Use select for timeout on Unix systems
                if hasattr(select, 'select'):
                    while not stop_event.is_set():
                        ready, _, _ = select.select([sys.stdin], [], [], 0.5)
                        if ready:
                            sys.stdin.readline()
                            break
                else:
                    # Fallback for Windows
                    input()
            except KeyboardInterrupt:
                pass
            finally:
                stop_event.set()
                stream.stop()
                stream.close()

            # Combine recording
            if recording:
                audio = np.concatenate(recording, axis=0).flatten().astype(np.float32)
            else:
                return ""

            print(f"✓ Recorded {len(audio)/sample_rate:.1f}s")

        # Transcribe
        print("⏳ Transcribing...")
        if duration:
            text = asyncio.run(self.transcribe(clip.data))
        else:
            text = asyncio.run(self.transcribe(audio))

        return text


def oneshot_mode(server_url: str, clipboard: bool = False):
    """
    One-shot mode for keyboard shortcuts.

    Records until Enter, transcribes, prints result.
    If clipboard=True, copies to clipboard.
    """
    client = PushToTalkClient(server_url)

    try:
        text = client.record_and_transcribe()

        if not text:
            print("(no speech detected)", file=sys.stderr)
            sys.exit(1)

        # Print result
        print(text)

        # Copy to clipboard if requested
        if clipboard:
            try:
                import pyperclip
                pyperclip.copy(text)
                print("✓ Copied to clipboard", file=sys.stderr)
            except ImportError:
                print("⚠ pyperclip not installed (pip install pyperclip)", file=sys.stderr)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def interactive_mode(server_url: str):
    """Interactive mode - keep recording on demand."""
    client = PushToTalkClient(server_url)

    print("🎤 Push-to-Talk Transcription")
    print(f"   Server: {server_url}")
    print("   Press Enter to start recording, Enter again to stop")
    print("   Ctrl+C to quit\n")

    try:
        while True:
            input("Press Enter to start recording...")
            text = client.record_and_transcribe()

            if text:
                print(f"\n📝 Transcription:\n{text}\n")
            else:
                print("(no speech detected)\n")

    except KeyboardInterrupt:
        print("\n✓ Goodbye!")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Push-to-talk transcription client")
    parser.add_argument(
        "--server",
        type=str,
        default="ws://192.168.2.24:8765",
        help="WebSocket server URL"
    )
    parser.add_argument(
        "--oneshot",
        action="store_true",
        help="One-shot mode (record once, transcribe, exit)"
    )
    parser.add_argument(
        "--clipboard",
        action="store_true",
        help="Copy result to clipboard (requires pyperclip)"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Fixed recording duration in seconds (default: press Enter to stop)"
    )

    args = parser.parse_args()

    if args.oneshot:
        oneshot_mode(args.server, args.clipboard)
    else:
        interactive_mode(args.server)


if __name__ == "__main__":
    main()
