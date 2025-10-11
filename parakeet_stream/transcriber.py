"""
Main streaming transcriber implementation using Parakeet TDT models
"""
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, List, Optional, Union

import numpy as np
import torch
from tqdm.auto import tqdm

from parakeet_stream.config import TranscriberConfig
from parakeet_stream.transcript import TranscriptResult
from parakeet_stream.utils import ContextSize, load_audio, make_divisible_by


# Deprecated: kept for backward compatibility
@dataclass
class TranscriptionResult:
    """
    DEPRECATED: Use TranscriptResult from parakeet_stream.transcript instead.

    Legacy result class kept for backward compatibility.

    Attributes:
        text: Transcribed text
        timestamps: Optional word-level timestamps if enabled
        confidence: Optional confidence score
    """

    text: str
    timestamps: Optional[List[dict]] = None
    confidence: Optional[float] = None


@dataclass
class StreamChunk:
    """Chunk of streaming transcription result.

    Attributes:
        text: Partial transcribed text for this chunk
        is_final: Whether this is the final chunk
        timestamp_start: Start time in seconds
        timestamp_end: End time in seconds
    """

    text: str
    is_final: bool = False
    timestamp_start: Optional[float] = None
    timestamp_end: Optional[float] = None


class StreamingTranscriber:
    """Simple declarative API for streaming transcription with Parakeet TDT models.

    Example:
        >>> # Simple file transcription
        >>> transcriber = StreamingTranscriber()
        >>> result = transcriber.transcribe("audio.wav")
        >>> print(result.text)

        >>> # Streaming transcription
        >>> transcriber = StreamingTranscriber(streaming=True)
        >>> for chunk in transcriber.stream("audio.wav"):
        ...     print(chunk.text, end="", flush=True)
    """

    def __init__(
        self,
        model_name: str = "nvidia/parakeet-tdt-0.6b-v3",
        device: str = "cpu",
        config: Optional[TranscriberConfig] = None,
        streaming: bool = False,
    ):
        """Initialize the streaming transcriber.

        Args:
            model_name: Name of the pretrained model
            device: Device to run inference on (cpu, cuda, mps)
            config: Optional TranscriberConfig for advanced configuration
            streaming: Whether to enable streaming mode by default
        """
        if config is None:
            config = TranscriberConfig(model_name=model_name, device=device, streaming=streaming)
        else:
            # Override config with explicit parameters
            config.model_name = model_name
            config.device = device
            config.streaming = streaming

        self.config = config
        self.model = None
        self._initialized = False

    def _initialize_model(self):
        """Lazy initialization of the model."""
        if self._initialized:
            return

        try:
            import nemo.collections.asr as nemo_asr
            from omegaconf import OmegaConf, open_dict
        except ImportError:
            raise ImportError(
                "NeMo toolkit is required. Install with: pip install nemo_toolkit[asr]"
            )

        print(f"Loading model {self.config.model_name}...")
        self.model = nemo_asr.models.ASRModel.from_pretrained(
            model_name=self.config.model_name
        )

        # Move model to device and set dtype
        self.model = self.model.to(self.config.device)
        if self.config.compute_dtype == "float16":
            self.model = self.model.half()
        elif self.config.compute_dtype == "bfloat16":
            self.model = self.model.to(torch.bfloat16)

        # Configure for streaming
        model_cfg = copy.deepcopy(self.model._cfg)
        OmegaConf.set_struct(model_cfg.preprocessor, False)
        model_cfg.preprocessor.dither = 0.0
        model_cfg.preprocessor.pad_to = 0
        OmegaConf.set_struct(model_cfg.preprocessor, True)

        self.model.preprocessor.featurizer.dither = 0.0
        self.model.preprocessor.featurizer.pad_to = 0
        self.model.eval()

        # Setup decoding for streaming if needed
        if hasattr(self.model, 'change_decoding_strategy'):
            from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTDecodingConfig

            decoding_cfg = RNNTDecodingConfig(
                strategy=self.config.decoding_strategy,
                preserve_alignments=False,
                fused_batch_size=-1,
            )
            # Convert to OmegaConf DictConfig for modification
            decoding_cfg = OmegaConf.structured(decoding_cfg)
            with open_dict(decoding_cfg):
                decoding_cfg.greedy.loop_labels = True
                decoding_cfg.tdt_include_token_duration = self.config.timestamps

            self.model.change_decoding_strategy(decoding_cfg)

        # Store model configuration
        self.sample_rate = model_cfg.preprocessor['sample_rate']
        self.feature_stride_sec = model_cfg.preprocessor['window_stride']
        self.features_per_sec = 1.0 / self.feature_stride_sec
        self.encoder_subsampling_factor = self.model.encoder.subsampling_factor

        # Calculate context sizes
        self._compute_context_sizes()

        self._initialized = True
        print(f"Model loaded successfully on {self.config.device}")

    def _compute_context_sizes(self):
        """Compute context sizes for streaming."""
        features_frame2audio_samples = make_divisible_by(
            int(self.sample_rate * self.feature_stride_sec),
            factor=self.encoder_subsampling_factor
        )
        self.encoder_frame2audio_samples = features_frame2audio_samples * self.encoder_subsampling_factor

        self.context_encoder_frames = ContextSize(
            left=int(self.config.left_context_secs * self.features_per_sec / self.encoder_subsampling_factor),
            chunk=int(self.config.chunk_secs * self.features_per_sec / self.encoder_subsampling_factor),
            right=int(self.config.right_context_secs * self.features_per_sec / self.encoder_subsampling_factor),
        )

        self.context_samples = ContextSize(
            left=self.context_encoder_frames.left * self.encoder_subsampling_factor * features_frame2audio_samples,
            chunk=self.context_encoder_frames.chunk * self.encoder_subsampling_factor * features_frame2audio_samples,
            right=self.context_encoder_frames.right * self.encoder_subsampling_factor * features_frame2audio_samples,
        )

    def transcribe(
        self,
        audio: Union[str, Path, np.ndarray, torch.Tensor],
        timestamps: Optional[bool] = None,
    ) -> TranscriptionResult:
        """Transcribe audio file or array.

        Args:
            audio: Path to audio file or audio array
            timestamps: Whether to include timestamps (overrides config)

        Returns:
            TranscriptionResult with transcribed text
        """
        self._initialize_model()

        if timestamps is None:
            timestamps = self.config.timestamps

        # Load audio if path is provided
        if isinstance(audio, (str, Path)):
            audio = load_audio(audio, self.sample_rate)

        # Convert to tensor if needed
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio)

        # Calculate duration before transcription
        duration = len(audio) / self.sample_rate if self.sample_rate else None

        # Suppress NeMo's internal deprecation warnings
        import logging

        # Mark the deprecated function as already warned in NeMo's decorator tracker
        try:
            from nemo.utils.decorators.deprecated import _PRINTED_WARNING
            _PRINTED_WARNING['_transcribe_output_processing'] = True
        except (ImportError, KeyError):
            pass

        # Transcribe using model's built-in method with return_hypotheses=True
        output = self.model.transcribe([audio], timestamps=timestamps, return_hypotheses=True)

        # Extract results from Hypothesis object
        hyp = output[0]

        # Handle case where output might be nested (list of lists)
        if isinstance(hyp, list) and len(hyp) > 0:
            hyp = hyp[0]

        # Get text from Hypothesis
        if hasattr(hyp, 'text'):
            text = hyp.text
        elif hasattr(hyp, '__getitem__') and hasattr(hyp[0], 'text'):
            # Another level of nesting
            text = hyp[0].text
        else:
            text = str(hyp)

        # Extract confidence if available
        confidence = None
        if hasattr(hyp, 'score'):
            confidence = float(hyp.score)
        elif hasattr(hyp, 'confidence'):
            confidence = float(hyp.confidence)

        # Extract timestamps if requested
        timestamp_list = None
        if timestamps and hasattr(hyp, 'timestamp'):
            timestamp_list = hyp.timestamp.get('word', [])

        # Return new rich TranscriptResult
        return TranscriptResult(
            text=text,
            confidence=confidence,
            duration=duration,
            timestamps=timestamp_list,
        )

    def stream(
        self,
        audio: Union[str, Path, np.ndarray, torch.Tensor],
    ) -> Generator[StreamChunk, None, None]:
        """Stream transcription results as they become available.

        Args:
            audio: Path to audio file or audio array

        Yields:
            StreamChunk objects with partial transcription results
        """
        self._initialize_model()

        # Load audio if path is provided
        if isinstance(audio, (str, Path)):
            audio = load_audio(audio, self.sample_rate)

        # Convert to tensor if needed
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio)

        # Ensure audio is on correct device
        audio = audio.to(self.config.device)

        # Add batch dimension if needed
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        yield from self._stream_buffered(audio)

    def _stream_buffered(self, audio_batch: torch.Tensor) -> Generator[StreamChunk, None, None]:
        """Internal method for buffered streaming transcription.

        Args:
            audio_batch: Audio tensor with shape [batch, samples]

        Yields:
            StreamChunk objects with transcription results
        """
        from nemo.collections.asr.parts.utils.rnnt_utils import batched_hyps_to_hypotheses
        from nemo.collections.asr.parts.utils.streaming_utils import StreamingBatchedAudioBuffer

        batch_size = audio_batch.shape[0]
        device = audio_batch.device
        audio_batch_lengths = torch.tensor([audio_batch.shape[1]] * batch_size, device=device, dtype=torch.long)

        # Initialize buffer with NeMo's proper streaming buffer
        buffer = StreamingBatchedAudioBuffer(
            batch_size=batch_size,
            context_samples=self.context_samples,
            dtype=audio_batch.dtype,
            device=device,
        )

        state = None
        current_batched_hyps = None
        left_sample = 0
        right_sample = min(self.context_samples.chunk + self.context_samples.right, audio_batch.shape[1])
        rest_audio_lengths = audio_batch_lengths.clone()

        with torch.no_grad(), torch.inference_mode():
            while left_sample < audio_batch.shape[1]:
                # Add samples to buffer
                chunk_length = min(right_sample, audio_batch.shape[1]) - left_sample
                is_last_chunk_batch = chunk_length >= rest_audio_lengths
                is_last_chunk = right_sample >= audio_batch.shape[1]

                chunk_lengths_batch = torch.where(
                    is_last_chunk_batch,
                    rest_audio_lengths,
                    torch.full_like(rest_audio_lengths, fill_value=chunk_length),
                )

                buffer.add_audio_batch_(
                    audio_batch[:, left_sample:right_sample],
                    audio_lengths=chunk_lengths_batch,
                    is_last_chunk=is_last_chunk,
                    is_last_chunk_batch=is_last_chunk_batch,
                )

                # Get encoder output using full buffer [left-chunk-right]
                encoder_output, encoder_output_len = self.model(
                    input_signal=buffer.samples,
                    input_signal_length=buffer.context_size_batch.total(),
                )
                encoder_output = encoder_output.transpose(1, 2)  # [B, T, C]

                # Remove extra context from encoder_output (leave only frames corresponding to the chunk)
                encoder_context = buffer.context_size.subsample(factor=self.encoder_frame2audio_samples)
                encoder_context_batch = buffer.context_size_batch.subsample(factor=self.encoder_frame2audio_samples)
                # Remove left context
                encoder_output = encoder_output[:, encoder_context.left:]

                # Decode only chunk frames
                decoding_computer = self.model.decoding.decoding.decoding_computer
                chunk_batched_hyps, _, state = decoding_computer(
                    x=encoder_output,
                    out_len=encoder_context_batch.chunk,
                    prev_batched_state=state,
                )

                # Merge hyps with previous hyps
                if current_batched_hyps is None:
                    current_batched_hyps = chunk_batched_hyps
                else:
                    current_batched_hyps.merge_(chunk_batched_hyps)

                # Convert to text and yield
                hyps = batched_hyps_to_hypotheses(current_batched_hyps, None, batch_size=batch_size)
                text = self.model.tokenizer.ids_to_text(hyps[0].y_sequence.tolist())

                timestamp_start = left_sample / self.sample_rate
                timestamp_end = right_sample / self.sample_rate

                yield StreamChunk(
                    text=text,
                    is_final=is_last_chunk,
                    timestamp_start=timestamp_start,
                    timestamp_end=timestamp_end,
                )

                # Move to next sample
                rest_audio_lengths -= chunk_lengths_batch
                left_sample = right_sample
                right_sample = min(right_sample + self.context_samples.chunk, audio_batch.shape[1])

    def transcribe_batch(
        self,
        audio_files: List[Union[str, Path]],
        timestamps: Optional[bool] = None,
        show_progress: bool = True,
    ) -> List[TranscriptionResult]:
        """Transcribe multiple audio files in batch.

        Args:
            audio_files: List of paths to audio files
            timestamps: Whether to include timestamps
            show_progress: Whether to show progress bar

        Returns:
            List of TranscriptionResult objects
        """
        self._initialize_model()

        if timestamps is None:
            timestamps = self.config.timestamps

        results = []
        iterator = tqdm(audio_files, desc="Transcribing") if show_progress else audio_files

        for audio_file in iterator:
            result = self.transcribe(audio_file, timestamps=timestamps)
            results.append(result)

        return results
