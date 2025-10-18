#!/usr/bin/env python3
"""
System-wide hotkey transcription service.
Press Alt+W to start/stop recording. Transcription appears in panel status bar.
"""

import asyncio
import threading
from pynput import keyboard
from pynput.keyboard import Key, KeyCode
import panelstatus as ps
from examples.push_to_talk_client import PushToTalkClient


class HotkeyTranscriber:
    def __init__(self, server_url='ws://192.168.2.24:8765', auto_paste=False):
        self.client = PushToTalkClient(server_url)
        self.recording = False
        self.recording_task = None
        self.loop = None
        self.auto_paste = auto_paste

    def start(self):
        """Start the hotkey listener"""
        # Create event loop for async operations
        self.loop = asyncio.new_event_loop()

        # Start event loop in background thread
        threading.Thread(target=self._run_event_loop, daemon=True).start()

        # Set up hotkey listener
        hotkey = keyboard.HotKey(
            keyboard.HotKey.parse('<alt>+w'),
            self._on_hotkey
        )

        with keyboard.Listener(
            on_press=lambda k: hotkey.press(self._normalize_key(k)),
            on_release=lambda k: hotkey.release(self._normalize_key(k))
        ) as listener:
            ps.status.set("🎤 Ready (Alt+W to record)", color="green")
            print("🎤 Hotkey service running. Press Alt+W to start/stop recording.")
            print("Press Ctrl+C to exit.")
            listener.join()

    def _normalize_key(self, key):
        """Normalize key for hotkey matching"""
        if isinstance(key, KeyCode):
            return key
        return key

    def _run_event_loop(self):
        """Run asyncio event loop in background thread"""
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def _on_hotkey(self):
        """Handle Alt+W hotkey press"""
        if not self.recording:
            # Start recording
            self.recording = True
            asyncio.run_coroutine_threadsafe(self._record(), self.loop)
        else:
            # Stop recording
            self.recording = False

    async def _record(self):
        """Record and transcribe audio"""
        try:
            ps.status.set("🔴 Recording... (Alt+W to stop)", color="red")

            # Start recording
            import sounddevice as sd
            import numpy as np
            import time

            recording = []
            sample_rate = 16000
            start_time = time.time()

            # Safety limits
            MAX_DURATION = 9.5 * 60  # 9.5 minutes (safe margin before 10min limit)
            WARN_DURATION = 8 * 60   # Warn at 8 minutes

            def callback(indata, frames, time, status):
                recording.append(indata.copy())

            # Start recording stream
            stream = sd.InputStream(
                samplerate=sample_rate,
                channels=1,
                callback=callback
            )
            stream.start()

            # Record until stopped or max duration reached
            while self.recording:
                elapsed = time.time() - start_time

                # Auto-stop at max duration
                if elapsed >= MAX_DURATION:
                    ps.status.set("⚠️  Max duration reached! Stopping...", color="yellow")
                    self.recording = False
                    break

                # Warn when approaching limit
                if elapsed >= WARN_DURATION and elapsed < WARN_DURATION + 1:
                    remaining = int(MAX_DURATION - elapsed)
                    ps.status.set(f"⚠️  {remaining//60}:{remaining%60:02d} left! (Alt+W to stop)", color="yellow")

                await asyncio.sleep(0.1)

            # Stop recording
            stream.stop()
            stream.close()

            # Convert to audio array
            if not recording:
                ps.status.set("🎤 Ready (Alt+W to record)", color="green")
                return

            audio = np.concatenate(recording, axis=0).flatten().astype(np.float32)
            duration = len(audio) / sample_rate

            # For very long recordings, chunk and transcribe progressively
            CHUNK_DURATION = 8 * 60  # 8 minutes per chunk (safe margin)
            chunk_size = int(CHUNK_DURATION * sample_rate)

            if len(audio) > chunk_size:
                ps.status.set(f"⏳ Transcribing {duration:.1f}s in chunks...", color="yellow")

                # Split into chunks and transcribe each
                text_parts = []
                num_chunks = (len(audio) + chunk_size - 1) // chunk_size

                for i in range(0, len(audio), chunk_size):
                    chunk = audio[i:i + chunk_size]
                    chunk_num = i // chunk_size + 1
                    ps.status.set(f"⏳ Transcribing chunk {chunk_num}/{num_chunks}...", color="yellow")

                    chunk_text = await self.client.transcribe(chunk)
                    if chunk_text:
                        text_parts.append(chunk_text)

                text = " ".join(text_parts)
            else:
                ps.status.set(f"⏳ Transcribing {duration:.1f}s...", color="yellow")
                # Transcribe normally
                text = await self.client.transcribe(audio)

            # Display result
            if text:
                ps.status.scroll(text, color="blue")

                # Copy to clipboard
                try:
                    import pyperclip
                    pyperclip.copy(text)
                except ImportError:
                    pass  # Silently skip if pyperclip not installed

                # Auto-paste if enabled
                if self.auto_paste:
                    try:
                        import subprocess
                        import time
                        # Small delay to let clipboard settle
                        time.sleep(0.1)

                        # Detect if we're in a terminal
                        is_terminal = False
                        try:
                            # Get active window PID
                            result = subprocess.run(
                                ['xdotool', 'getactivewindow', 'getwindowpid'],
                                capture_output=True,
                                text=True,
                                check=True
                            )
                            window_pid = result.stdout.strip()

                            # Get process tree for this window
                            result = subprocess.run(
                                ['ps', '-o', 'comm=', '-p', window_pid],
                                capture_output=True,
                                text=True
                            )
                            process_name = result.stdout.strip().lower()

                            # Also check child processes
                            result = subprocess.run(
                                ['pgrep', '-P', window_pid],
                                capture_output=True,
                                text=True
                            )
                            child_pids = result.stdout.strip().split('\n')

                            all_processes = [process_name]
                            for pid in child_pids:
                                if pid:
                                    result = subprocess.run(
                                        ['ps', '-o', 'comm=', '-p', pid],
                                        capture_output=True,
                                        text=True
                                    )
                                    all_processes.append(result.stdout.strip().lower())

                            # Common terminal/shell process names
                            terminal_keywords = [
                                'terminal', 'konsole', 'xterm', 'rxvt', 'urxvt',
                                'terminator', 'tilix', 'alacritty', 'kitty', 'wezterm',
                                'st', 'ghostty', 'bash', 'zsh', 'fish', 'sh', 'tmux',
                                'screen', 'csh', 'tcsh', 'ksh', 'nvim', 'vim', 'emacs'
                            ]

                            is_terminal = any(
                                any(keyword in proc for keyword in terminal_keywords)
                                for proc in all_processes
                            )
                        except Exception:
                            pass

                        # Use appropriate paste shortcut
                        if is_terminal:
                            subprocess.run(['xdotool', 'key', 'ctrl+shift+v'], check=False)
                        else:
                            subprocess.run(['xdotool', 'key', 'ctrl+v'], check=False)

                    except Exception:
                        pass  # Silently skip if paste fails
            else:
                ps.status.set("🎤 Ready (Alt+W to record)", color="green")

        except Exception as e:
            ps.status.set(f"❌ Error: {str(e)}", color="red")
            import traceback
            traceback.print_exc()

        self.recording = False


def main():
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="System-wide hotkey transcription")
    parser.add_argument(
        "--server",
        type=str,
        default="ws://192.168.2.24:8765",
        help="WebSocket server URL"
    )
    parser.add_argument(
        "--auto-paste",
        action="store_true",
        help="Automatically paste transcription with Ctrl+Shift+V"
    )

    args = parser.parse_args()

    transcriber = HotkeyTranscriber(args.server, auto_paste=args.auto_paste)

    try:
        transcriber.start()
    except KeyboardInterrupt:
        ps.status.set("", color=None)  # Clear status
        print("\n👋 Goodbye!")


if __name__ == '__main__':
    main()
