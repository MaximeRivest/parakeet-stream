"""
AudioClip class for recorded audio.
"""
from pathlib import Path
from typing import Union

import numpy as np

from parakeet_stream.display import RichRepr, format_duration, format_sample_rate


class AudioClip(RichRepr):
    """
    Wrapper for recorded audio clip.

    Provides playback and save functionality with rich display in REPL/Jupyter.

    Attributes:
        data: Audio data as numpy array (mono)
        sample_rate: Sample rate in Hz
    """

    def __init__(self, data: np.ndarray, sample_rate: int):
        """
        Create audio clip from numpy array.

        Args:
            data: Audio samples as numpy array (mono, float32 or float64)
            sample_rate: Sample rate in Hz (e.g., 16000, 44100)
        """
        self.data = data
        self.sample_rate = sample_rate

    @property
    def duration(self) -> float:
        """
        Duration of audio clip in seconds.

        Returns:
            Duration in seconds
        """
        return len(self.data) / self.sample_rate

    @property
    def num_samples(self) -> int:
        """
        Number of audio samples.

        Returns:
            Sample count
        """
        return len(self.data)

    def play(self):
        """
        Play audio clip through default audio device.

        Requires sounddevice library.

        Raises:
            ImportError: If sounddevice is not installed
        """
        try:
            import sounddevice as sd
        except ImportError:
            raise ImportError(
                "sounddevice is required for audio playback. "
                "Install with: pip install sounddevice"
            )

        print(f"🔊 Playing {format_duration(self.duration)}...")
        sd.play(self.data, self.sample_rate)
        sd.wait()
        print("✓ Playback complete")

    def save(self, path: Union[str, Path]):
        """
        Save audio clip to file.

        Supports WAV format. File extension determines format.

        Args:
            path: Output file path (e.g., "recording.wav")

        Raises:
            ImportError: If soundfile is not installed
        """
        try:
            import soundfile as sf
        except ImportError:
            raise ImportError(
                "soundfile is required for saving audio. "
                "Install with: pip install soundfile"
            )

        path = Path(path)
        sf.write(str(path), self.data, self.sample_rate)
        print(f"✓ Saved to {path} ({format_duration(self.duration)})")

    def __repr__(self) -> str:
        """
        String representation for Python REPL.

        Returns:
            Compact string with duration and sample rate
        """
        return (
            f"AudioClip(duration={self.duration:.1f}s, "
            f"sample_rate={self.sample_rate}Hz)"
        )

    def _repr_pretty_(self, p, cycle):
        """
        IPython pretty print representation.

        Args:
            p: IPython printer object
            cycle: Whether there's a circular reference
        """
        if cycle:
            p.text('AudioClip(...)')
            return

        lines = [
            f"🔊 AudioClip",
            f"   Duration: {format_duration(self.duration)}",
            f"   Sample Rate: {format_sample_rate(self.sample_rate)}",
            f"   Samples: {self.num_samples:,}",
        ]
        p.text('\n'.join(lines))

    def _repr_html_(self) -> str:
        """
        Jupyter HTML representation.

        Returns:
            HTML string for Jupyter display
        """
        return f"""
        <div style="border: 1px solid #ccc; padding: 12px; border-radius: 5px; background-color: #f9f9f9;">
            <h4 style="margin-top: 0;">🔊 AudioClip</h4>
            <table style="border-collapse: collapse; width: 100%;">
                <tr>
                    <td style="padding: 4px; font-weight: bold;">Duration:</td>
                    <td style="padding: 4px;">{format_duration(self.duration)}</td>
                </tr>
                <tr>
                    <td style="padding: 4px; font-weight: bold;">Sample Rate:</td>
                    <td style="padding: 4px;">{format_sample_rate(self.sample_rate)}</td>
                </tr>
                <tr>
                    <td style="padding: 4px; font-weight: bold;">Samples:</td>
                    <td style="padding: 4px;">{self.num_samples:,}</td>
                </tr>
            </table>
        </div>
        """

    def to_tensor(self):
        """
        Convert audio data to PyTorch tensor.

        Returns:
            Torch tensor with audio data

        Raises:
            ImportError: If torch is not installed
        """
        try:
            import torch
        except ImportError:
            raise ImportError(
                "torch is required for tensor conversion. "
                "Install with: pip install torch"
            )

        return torch.from_numpy(self.data)
