"""
Configuration for Parakeet Stream transcriber
"""
from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class TranscriberConfig:
    """Configuration for StreamingTranscriber.

    Attributes:
        model_name: Name of the pretrained model (default: nvidia/parakeet-tdt-0.6b-v3)
        device: Device to run inference on (default: "cpu")
        compute_dtype: Compute dtype for inference (default: "float32")
        chunk_secs: Chunk length in seconds for streaming (default: 2.0)
        left_context_secs: Left context in seconds (default: 10.0)
        right_context_secs: Right context in seconds (default: 2.0)
        batch_size: Batch size for processing multiple files (default: 1)
        timestamps: Whether to include timestamps in output (default: False)
        sample_rate: Audio sample rate in Hz (default: 16000)
        streaming: Whether to use streaming mode (default: False)
        decoding_strategy: Decoding strategy to use (default: "greedy_batch")
    """

    model_name: str = "nvidia/parakeet-tdt-0.6b-v3"
    device: str = "cpu"
    compute_dtype: Literal["float32", "float16", "bfloat16"] = "float32"

    # Streaming parameters
    chunk_secs: float = 2.0
    left_context_secs: float = 10.0
    right_context_secs: float = 2.0

    # Processing parameters
    batch_size: int = 1
    timestamps: bool = False
    sample_rate: int = 16000
    streaming: bool = False

    # Decoding parameters
    decoding_strategy: str = "greedy_batch"

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.chunk_secs <= 0:
            raise ValueError(f"chunk_secs must be positive, got {self.chunk_secs}")
        if self.left_context_secs < 0:
            raise ValueError(f"left_context_secs must be non-negative, got {self.left_context_secs}")
        if self.right_context_secs < 0:
            raise ValueError(f"right_context_secs must be non-negative, got {self.right_context_secs}")
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {self.sample_rate}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")

        # Validate device
        valid_devices = ["cpu", "cuda", "mps"]
        if not any(self.device.startswith(d) for d in valid_devices):
            raise ValueError(f"device must be one of {valid_devices}, got {self.device}")
