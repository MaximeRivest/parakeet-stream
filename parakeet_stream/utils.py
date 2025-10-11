"""
Utility functions for audio processing
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
import torch


def make_divisible_by(num: int, factor: int) -> int:
    """Make num divisible by factor.

    Args:
        num: Number to make divisible
        factor: Factor to divide by

    Returns:
        Largest number <= num that is divisible by factor
    """
    return (num // factor) * factor


@dataclass
class ContextSize:
    """Context size for buffered streaming.

    This is a simple dataclass to hold left, chunk, and right context sizes.
    Compatible with NeMo's streaming utilities.

    Attributes:
        left: Left context size in samples or frames
        chunk: Chunk size in samples or frames
        right: Right context size in samples or frames
    """
    left: int
    chunk: int
    right: int

    def total(self) -> int:
        """Return total context size."""
        return self.left + self.chunk + self.right

    def subsample(self, factor: int) -> "ContextSize":
        """Subsample context size by factor.

        Args:
            factor: Subsampling factor

        Returns:
            New ContextSize with subsampled values
        """
        return ContextSize(
            left=self.left // factor,
            chunk=self.chunk // factor,
            right=self.right // factor,
        )


def load_audio(
    audio_path: Union[str, Path],
    sample_rate: int = 16000,
) -> torch.Tensor:
    """Load audio file and resample to target sample rate.

    Args:
        audio_path: Path to audio file
        sample_rate: Target sample rate

    Returns:
        Audio tensor with shape [samples]
    """
    try:
        import soundfile as sf
    except ImportError:
        try:
            import librosa
            audio, sr = librosa.load(str(audio_path), sr=sample_rate, mono=True)
            return torch.from_numpy(audio).float()
        except ImportError:
            raise ImportError(
                "Either soundfile or librosa is required for audio loading. "
                "Install with: pip install soundfile"
            )

    # Try soundfile first (faster)
    audio, sr = sf.read(str(audio_path), dtype='float32')

    # Convert to mono if stereo
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Resample if needed
    if sr != sample_rate:
        try:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
        except ImportError:
            raise ImportError(
                "librosa is required for resampling. Install with: pip install librosa"
            )

    return torch.from_numpy(audio).float()


def format_timestamp(seconds: float) -> str:
    """Format timestamp in seconds to HH:MM:SS.mmm format.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted timestamp string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"
