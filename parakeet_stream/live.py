"""
Live transcription with background recording.
"""
import queue
import threading
import time
from pathlib import Path
from typing import Optional, TYPE_CHECKING, Union

import numpy as np

from parakeet_stream.display import RichRepr, format_duration
from parakeet_stream.microphone import Microphone
from parakeet_stream.transcript import Segment, TranscriptBuffer

if TYPE_CHECKING:
    from parakeet_stream.parakeet import Parakeet


class LiveTranscriber(RichRepr):
    """
    Background live transcription from microphone.

    Records audio continuously in background thread and transcribes in chunks.

    Attributes:
        transcriber: Parakeet transcriber instance
        microphone: Microphone to record from
        transcript: TranscriptBuffer accumulating results
    """

    def __init__(
        self,
        transcriber: 'Parakeet',
        microphone: Optional[Microphone] = None,
        output: Optional[Union[str, Path]] = None,
        chunk_duration: float = 2.0,
        verbose: bool = False,
        strategy: Optional['TranscriptionStrategy'] = None
    ):
        """
        Initialize live transcriber.

        Args:
            transcriber: Parakeet instance for transcription
            microphone: Microphone to use (default: auto-select)
            output: Optional file path to save transcript
            chunk_duration: Duration of audio chunks in seconds
            verbose: Whether to print transcriptions to console (default: False)
            strategy: Transcription strategy (default: None = standard streaming)
        """
        self.transcriber = transcriber
        self.microphone = microphone or Microphone()
        self.output_file = output
        self.chunk_duration = chunk_duration
        self.verbose = verbose
        self.strategy = strategy

        self.transcript = TranscriptBuffer()
        self._running = False
        self._paused = False
        self._thread = None
        self._start_time = None

    def start(self):
        """Start live transcription in background thread."""
        if self._running:
            raise RuntimeError("Already running. Call stop() first.")

        self._running = True
        self._paused = False
        self._start_time = time.time()
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()

        if self.verbose:
            print(f"🎤 Listening on: {self.microphone.name}")
            print("   (Press Ctrl+C or call .stop() to end)")

    def _listen_loop(self):
        """Background loop that records and transcribes audio."""
        try:
            import sounddevice as sd
        except ImportError:
            print("Error: sounddevice is required for live transcription")
            self._running = False
            return

        audio_queue = queue.Queue()

        def audio_callback(indata, frames, time_info, status):
            """Callback for audio input stream."""
            if status and self.verbose:
                print(f"⚠ Audio warning: {status}")
            if not self._paused:
                audio_queue.put(indata.copy())

        chunk_samples = int(self.microphone.sample_rate * self.chunk_duration)

        try:
            with sd.InputStream(
                samplerate=self.microphone.sample_rate,
                channels=1,
                dtype='float32',
                callback=audio_callback,
                blocksize=chunk_samples,
                device=self.microphone.device
            ):
                buffer = []
                while self._running:
                    if self._paused:
                        time.sleep(0.1)
                        continue

                    try:
                        chunk = audio_queue.get(timeout=0.1)
                        buffer.append(chunk)

                        # When we have enough audio, transcribe
                        total_samples = sum(len(c) for c in buffer)
                        if total_samples >= chunk_samples:
                            audio_data = np.concatenate(buffer).flatten()
                            buffer = []

                            # Transcribe using strategy or default method
                            try:
                                if self.strategy:
                                    # Use custom strategy
                                    segments = self.strategy.process_stream(
                                        audio_data,
                                        self.transcriber,
                                        self.microphone.sample_rate
                                    )

                                    # Add all segments from strategy
                                    for segment in segments:
                                        if segment.text.strip():
                                            self.transcript.append(segment)

                                            # Print to console if verbose
                                            if self.verbose:
                                                elapsed = time.time() - self._start_time
                                                print(f"[{format_duration(elapsed)}] {segment.text}")

                                            # Write to file if specified
                                            if self.output_file:
                                                with open(self.output_file, 'a') as f:
                                                    f.write(f"{segment.text}\n")
                                else:
                                    # Use default transcribe method
                                    result = self.transcriber.transcribe(audio_data, _quiet=True)

                                    if result.text.strip():
                                        # Calculate timestamps
                                        elapsed = time.time() - self._start_time
                                        segment = Segment(
                                            text=result.text.strip(),
                                            start_time=elapsed - self.chunk_duration,
                                            end_time=elapsed,
                                            confidence=result.confidence
                                        )
                                        self.transcript.append(segment)

                                        # Print to console if verbose
                                        if self.verbose:
                                            print(f"[{format_duration(elapsed)}] {result.text}")

                                        # Write to file if specified
                                        if self.output_file:
                                            with open(self.output_file, 'a') as f:
                                                f.write(f"{result.text}\n")

                            except Exception as e:
                                if self.verbose:
                                    print(f"⚠ Transcription error: {e}")

                    except queue.Empty:
                        continue

        except Exception as e:
            if self.verbose:
                print(f"⚠ Audio stream error: {e}")
        finally:
            self._running = False

    def pause(self):
        """Pause transcription (audio continues recording but not transcribed)."""
        if not self._running:
            raise RuntimeError("Not running. Call start() first.")
        self._paused = True
        if self.verbose:
            print("⏸  Paused")

    def resume(self):
        """Resume transcription."""
        if not self._running:
            raise RuntimeError("Not running. Call start() first.")
        self._paused = False
        if self.verbose:
            print("▶  Resumed")

    def stop(self):
        """Stop live transcription."""
        if not self._running:
            return

        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)

        if self.verbose:
            stats = self.transcript.stats
            print(f"\n✓ Stopped")
            print(f"  Segments: {stats['segments']}")
            print(f"  Duration: {format_duration(stats['duration'])}")
            print(f"  Words: {stats['words']}")

    @property
    def text(self) -> str:
        """
        Current full transcript text.

        Returns:
            Complete transcribed text
        """
        return self.transcript.text

    @property
    def is_running(self) -> bool:
        """Whether transcription is currently running."""
        return self._running

    @property
    def is_paused(self) -> bool:
        """Whether transcription is currently paused."""
        return self._paused

    @property
    def elapsed(self) -> float:
        """
        Elapsed time since start in seconds.

        Returns:
            Elapsed seconds
        """
        if self._start_time:
            return time.time() - self._start_time
        return 0.0

    def __repr__(self) -> str:
        """String representation for Python REPL."""
        status = "running" if self._running else "stopped"
        if self._paused:
            status = "paused"
        return (
            f"LiveTranscriber(status='{status}', "
            f"segments={len(self.transcript)})"
        )

    def _repr_pretty_(self, p, cycle):
        """IPython pretty print representation."""
        if cycle:
            p.text('LiveTranscriber(...)')
            return

        if self._running:
            status = "🟢 Running" if not self._paused else "⏸  Paused"
        else:
            status = "⚪ Stopped"

        lines = [
            f"🎤 LiveTranscriber ({status})",
            f"   Microphone: {self.microphone.name}",
            f"   Duration: {format_duration(self.elapsed)}",
            f"   Segments: {len(self.transcript)}",
        ]

        if self.transcript.segments:
            last_seg = self.transcript.segments[-1]
            lines.append(f"   Last: \"{last_seg.text[:50]}{'...' if len(last_seg.text) > 50 else ''}\"")

        p.text('\n'.join(lines))

    def _repr_html_(self) -> str:
        """Jupyter HTML representation."""
        if self._running:
            status_icon = "🟢" if not self._paused else "⏸"
            status_text = "Running" if not self._paused else "Paused"
        else:
            status_icon = "⚪"
            status_text = "Stopped"

        html_parts = [
            '<div style="border: 1px solid #ccc; padding: 12px; '
            'border-radius: 5px; background-color: #f9f9f9;">',
            f'<h4 style="margin-top: 0;">{status_icon} LiveTranscriber</h4>',
            '<table style="border-collapse: collapse; width: 100%;">',
            f'<tr><td style="padding: 4px; font-weight: bold;">Status:</td>'
            f'<td style="padding: 4px;">{status_text}</td></tr>',
            f'<tr><td style="padding: 4px; font-weight: bold;">Microphone:</td>'
            f'<td style="padding: 4px;">{self.microphone.name}</td></tr>',
            f'<tr><td style="padding: 4px; font-weight: bold;">Duration:</td>'
            f'<td style="padding: 4px;">{format_duration(self.elapsed)}</td></tr>',
            f'<tr><td style="padding: 4px; font-weight: bold;">Segments:</td>'
            f'<td style="padding: 4px;">{len(self.transcript)}</td></tr>',
            '</table>',
        ]

        if self.transcript.segments:
            last_seg = self.transcript.segments[-1]
            escaped_text = (
                last_seg.text.replace('&', '&amp;')
                .replace('<', '&lt;')
                .replace('>', '&gt;')
            )
            html_parts.append(
                f'<p style="margin-top: 12px; font-family: monospace; '
                f'font-size: 0.9em; color: #666;">'
                f'Last: "{escaped_text}"</p>'
            )

        html_parts.append('</div>')

        return ''.join(html_parts)
