"""
Parakeet Stream - Simple, powerful streaming transcription with Parakeet TDT

A modern, REPL-friendly Python API for real-time speech transcription with:
- Beautiful displays in Python REPL, IPython, and Jupyter notebooks
- 6 quality presets for instant quality/latency tuning
- Microphone support with device discovery
- Live transcription with background recording
- Client-server architecture for remote transcription
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
    >>>
    >>> # Client-server setup
    >>> # Server:
    >>> from parakeet_stream import ParakeetServer
    >>> server = ParakeetServer(host='0.0.0.0', port=8765)
    >>> server.start()
    >>>
    >>> # Client:
    >>> from parakeet_stream import ParakeetClient
    >>> client = ParakeetClient('ws://localhost:8765')
    >>> for segment in client.stream_microphone():
    ...     print(segment['text'])
"""

__version__ = "0.6.0"

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

# Transcription strategies
from parakeet_stream.strategies import (
    TranscriptionStrategy,
    DefaultStrategy,
    OverlappingWindowStrategy,
    ConsensusStrategy,
)

# Client-Server architecture
try:
    from parakeet_stream.server import ParakeetServer
    from parakeet_stream.client import ParakeetClient, ParakeetConnection
    _CLIENT_SERVER_AVAILABLE = True
except ImportError:
    _CLIENT_SERVER_AVAILABLE = False
    ParakeetServer = None
    ParakeetClient = None
    ParakeetConnection = None

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

    # Transcription strategies
    "TranscriptionStrategy",
    "DefaultStrategy",
    "OverlappingWindowStrategy",
    "ConsensusStrategy",

    # Client-Server
    "ParakeetServer",
    "ParakeetClient",
    "ParakeetConnection",

    # Legacy API (TranscriptionResult is deprecated, use TranscriptResult)
    "TranscriptionResult",
    "StreamingTranscriber",
    "TranscriberConfig",
]
