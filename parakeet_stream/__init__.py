"""
Parakeet Stream - Simple, powerful streaming transcription with Parakeet TDT

A modern, REPL-friendly Python API for real-time speech transcription with:
- Beautiful displays in Python REPL, IPython, and Jupyter notebooks
- 6 quality presets for instant quality/latency tuning
- Microphone support with device discovery
- Live transcription with background recording
- Fluent, chainable configuration API

Quick Start:
    >>> from parakeet_stream import Parakeet
    >>>
    >>> # Simple transcription
    >>> pk = Parakeet()
    >>> result = pk.transcribe("audio.wav")
    >>> print(result.text)
    >>>
    >>> # Live transcription
    >>> live = pk.listen()
    >>> # Speak into microphone...
    >>> live.stop()
    >>> print(live.transcript.text)
"""

__version__ = "0.3.0"

# Core transcription
from parakeet_stream.parakeet import Parakeet, StreamChunk, TranscriptionResult

# Audio configuration
from parakeet_stream.audio_config import AudioConfig, ConfigPresets

# Transcription results
from parakeet_stream.transcript import TranscriptResult, TranscriptBuffer, Segment

# Audio input
from parakeet_stream.microphone import Microphone, MicrophoneTestResult, TEST_PHRASES
from parakeet_stream.audio_clip import AudioClip

# Live transcription
from parakeet_stream.live import LiveTranscriber

# Legacy API (for backwards compatibility)
from parakeet_stream.config import TranscriberConfig
from parakeet_stream.transcriber import StreamingTranscriber

__all__ = [
    # Core API (recommended)
    "Parakeet",
    "AudioConfig",
    "ConfigPresets",

    # Results
    "TranscriptResult",
    "TranscriptBuffer",
    "Segment",

    # Audio input
    "Microphone",
    "MicrophoneTestResult",
    "AudioClip",

    # Live transcription
    "LiveTranscriber",

    # Streaming
    "StreamChunk",

    # Legacy API (TranscriptionResult is deprecated, use TranscriptResult)
    "TranscriptionResult",
    "StreamingTranscriber",
    "TranscriberConfig",
]
