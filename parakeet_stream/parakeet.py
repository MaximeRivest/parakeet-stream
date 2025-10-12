"""
Parakeet - Main interface for streaming transcription with Parakeet TDT models
"""
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, List, Optional, Union

import numpy as np
import torch
from tqdm.auto import tqdm

from parakeet_stream.audio_config import AudioConfig, ConfigPresets
from parakeet_stream.config import TranscriberConfig
from parakeet_stream.display import RichRepr
from parakeet_stream.transcript import TranscriptResult
from parakeet_stream.utils import ContextSize, load_audio, make_divisible_by


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


class Parakeet(RichRepr):
    """
    Main interface for audio transcription with Parakeet TDT models.

    Provides a simple, REPL-friendly API for transcribing audio with configurable
    quality/latency tradeoffs. Models are loaded eagerly by default with a progress
    bar.

    Examples:
        >>> # Simple transcription (loads model with progress bar)
        >>> pk = Parakeet()
        >>> result = pk.transcribe("audio.wav")
        >>> print(result.text)

        >>> # Quality experimentation
        >>> pk.with_quality('high').transcribe("audio.wav")
        >>> pk.with_latency('low').transcribe("audio.wav")

        >>> # Streaming transcription
        >>> for chunk in pk.stream("audio.wav"):
        ...     print(chunk.text)

    Attributes:
        model_name: Name of the pretrained model
        device: Device for inference (cpu, cuda, mps)
        configs: Access to ConfigPresets
    """

    def __init__(
        self,
        model_name: str = "nvidia/parakeet-tdt-0.6b-v3",
        device: str = "cpu",
        config: Optional[Union[str, AudioConfig]] = None,
        lazy: bool = False,
    ):
        """
        Initialize Parakeet transcriber.

        Args:
            model_name: Name of pretrained model (default: parakeet-tdt-0.6b-v3)
            device: Device for inference - 'cpu', 'cuda', or 'mps' (default: cpu)
            config: Audio configuration preset name or AudioConfig object
                   (default: 'balanced')
            lazy: If False (default), load model immediately with progress bar.
                  If True, delay loading until first transcription.

        Examples:
            >>> pk = Parakeet()  # Loads immediately with progress
            >>> pk = Parakeet(device='cuda')  # Use GPU
            >>> pk = Parakeet(config='high_quality')  # High quality preset
            >>> pk = Parakeet(lazy=True)  # Delay loading
        """
        self.model_name = model_name
        self.device = device

        # Resolve audio config
        if config is None:
            self._audio_config = ConfigPresets.BALANCED
        elif isinstance(config, str):
            self._audio_config = ConfigPresets.get(config)
        elif isinstance(config, AudioConfig):
            self._audio_config = config
        else:
            raise TypeError(f"config must be str or AudioConfig, got {type(config)}")

        # Create internal TranscriberConfig for compatibility
        self._transcriber_config = self._create_transcriber_config()

        # Model state
        self.model = None
        self._initialized = False
        self.sample_rate = None

        # Load model immediately unless lazy
        if not lazy:
            self.load()

    @property
    def configs(self):
        """Access to configuration presets."""
        return ConfigPresets

    @property
    def config(self):
        """Current audio configuration."""
        return self._audio_config

    def _create_transcriber_config(self) -> TranscriberConfig:
        """Create TranscriberConfig from AudioConfig for internal compatibility."""
        return TranscriberConfig(
            model_name=self.model_name,
            device=self.device,
            chunk_secs=self._audio_config.chunk_secs,
            left_context_secs=self._audio_config.left_context_secs,
            right_context_secs=self._audio_config.right_context_secs,
            streaming=True,
        )

    def load(self):
        """
        Load the model with progress bar.

        This is called automatically during __init__ unless lazy=True.
        You can call it manually if you used lazy=True.

        The model takes approximately 3-5 minutes to load on first run
        (downloads from HuggingFace). Subsequent runs are faster (loads from cache).
        """
        if self._initialized:
            return

        # Suppress verbose NeMo logging and SyntaxWarnings during model loading
        import logging
        import warnings

        # Save current warning filters
        original_filters = warnings.filters[:]

        # Suppress ALL warnings during import and loading
        warnings.filterwarnings('ignore')

        try:
            import nemo.collections.asr as nemo_asr
            from omegaconf import OmegaConf, open_dict
        except ImportError:
            warnings.filters[:] = original_filters
            raise ImportError(
                "NeMo toolkit is required. Install with: pip install nemo_toolkit[asr]"
            )

        # Save current log levels
        nemo_logger = logging.getLogger('nemo_logger')
        root_logger = logging.getLogger()
        original_nemo_level = nemo_logger.level
        original_root_level = root_logger.level

        # Suppress NeMo logs
        nemo_logger.setLevel(logging.ERROR)
        root_logger.setLevel(logging.ERROR)

        try:
            print(f"\nLoading {self.model_name} on {self.device}...")

            # Use tqdm for progress indication
            with tqdm(
                total=5,
                desc="Loading model",
                bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}',
                ncols=80,
                leave=True
            ) as pbar:
                pbar.set_description("Loading model")
                self.model = nemo_asr.models.ASRModel.from_pretrained(
                    model_name=self.model_name
                )
                pbar.update(1)

                pbar.set_description("Moving to device")
                self.model = self.model.to(self.device)
                if self._transcriber_config.compute_dtype == "float16":
                    self.model = self.model.half()
                elif self._transcriber_config.compute_dtype == "bfloat16":
                    self.model = self.model.to(torch.bfloat16)
                pbar.update(1)

                pbar.set_description("Configuring streaming")
                # Configure for streaming
                model_cfg = copy.deepcopy(self.model._cfg)
                OmegaConf.set_struct(model_cfg.preprocessor, False)
                model_cfg.preprocessor.dither = 0.0
                model_cfg.preprocessor.pad_to = 0
                OmegaConf.set_struct(model_cfg.preprocessor, True)

                self.model.preprocessor.featurizer.dither = 0.0
                self.model.preprocessor.featurizer.pad_to = 0
                self.model.eval()
                pbar.update(1)

                pbar.set_description("Setting up decoder")
                # Setup decoding for streaming if needed
                if hasattr(self.model, 'change_decoding_strategy'):
                    from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTDecodingConfig

                    decoding_cfg = RNNTDecodingConfig(
                        strategy=self._transcriber_config.decoding_strategy,
                        preserve_alignments=False,
                        fused_batch_size=-1,
                    )
                    # Convert to OmegaConf DictConfig for modification
                    decoding_cfg = OmegaConf.structured(decoding_cfg)
                    with open_dict(decoding_cfg):
                        decoding_cfg.greedy.loop_labels = True
                        decoding_cfg.tdt_include_token_duration = self._transcriber_config.timestamps

                    self.model.change_decoding_strategy(decoding_cfg)
                pbar.update(1)

                pbar.set_description("Computing context")
                # Store model configuration
                self.sample_rate = model_cfg.preprocessor['sample_rate']
                self.feature_stride_sec = model_cfg.preprocessor['window_stride']
                self.features_per_sec = 1.0 / self.feature_stride_sec
                self.encoder_subsampling_factor = self.model.encoder.subsampling_factor

                # Calculate context sizes
                self._compute_context_sizes()
                pbar.update(1)

            self._initialized = True
            print(f"✓ Ready! ({self.model_name} on {self.device})\n")

        finally:
            # Restore log levels
            nemo_logger.setLevel(original_nemo_level)
            root_logger.setLevel(original_root_level)
            # Restore warning filters
            warnings.filters[:] = original_filters

    def _initialize_model(self):
        """
        Legacy method for backwards compatibility.
        Use load() instead.
        """
        self.load()

    def _compute_context_sizes(self):
        """Compute context sizes for streaming."""
        features_frame2audio_samples = make_divisible_by(
            int(self.sample_rate * self.feature_stride_sec),
            factor=self.encoder_subsampling_factor
        )
        self.encoder_frame2audio_samples = features_frame2audio_samples * self.encoder_subsampling_factor

        self.context_encoder_frames = ContextSize(
            left=int(self._audio_config.left_context_secs * self.features_per_sec / self.encoder_subsampling_factor),
            chunk=int(self._audio_config.chunk_secs * self.features_per_sec / self.encoder_subsampling_factor),
            right=int(self._audio_config.right_context_secs * self.features_per_sec / self.encoder_subsampling_factor),
        )

        self.context_samples = ContextSize(
            left=self.context_encoder_frames.left * self.encoder_subsampling_factor * features_frame2audio_samples,
            chunk=self.context_encoder_frames.chunk * self.encoder_subsampling_factor * features_frame2audio_samples,
            right=self.context_encoder_frames.right * self.encoder_subsampling_factor * features_frame2audio_samples,
        )

    def with_config(self, config: Union[str, AudioConfig]) -> 'Parakeet':
        """
        Set audio configuration (chainable, does NOT reload model).

        Args:
            config: Preset name (e.g., 'balanced') or AudioConfig object

        Returns:
            self (for method chaining)

        Examples:
            >>> pk.with_config('high_quality').transcribe("audio.wav")
            >>> pk.with_config(ConfigPresets.LOW_LATENCY).transcribe("audio.wav")
        """
        if isinstance(config, str):
            self._audio_config = ConfigPresets.get(config)
        elif isinstance(config, AudioConfig):
            self._audio_config = config
        else:
            raise TypeError(f"config must be str or AudioConfig, got {type(config)}")

        # Update internal transcriber config
        self._transcriber_config = self._create_transcriber_config()

        # Recompute context sizes if model is loaded
        if self._initialized:
            self._compute_context_sizes()

        return self

    def with_quality(self, level: str) -> 'Parakeet':
        """
        Set quality level (chainable, does NOT reload model).

        Args:
            level: Quality level - 'max', 'high', 'good', 'low', or 'realtime'

        Returns:
            self (for method chaining)

        Examples:
            >>> pk.with_quality('max').transcribe("audio.wav")
            >>> pk.with_quality('high').transcribe("audio.wav")
        """
        return self.with_config(ConfigPresets.by_quality(level))

    def with_latency(self, level: str) -> 'Parakeet':
        """
        Set latency level (chainable, does NOT reload model).

        Args:
            level: Latency level - 'high', 'medium', 'low', or 'realtime'

        Returns:
            self (for method chaining)

        Examples:
            >>> pk.with_latency('low').transcribe("audio.wav")
            >>> pk.with_latency('realtime').transcribe("audio.wav")
        """
        return self.with_config(ConfigPresets.by_latency(level))

    def with_params(
        self,
        chunk_secs: Optional[float] = None,
        left_context_secs: Optional[float] = None,
        right_context_secs: Optional[float] = None
    ) -> 'Parakeet':
        """
        Set custom parameters (chainable, does NOT reload model).

        Args:
            chunk_secs: Chunk size in seconds (affects latency)
            left_context_secs: Left context window size (improves quality)
            right_context_secs: Right context window size (affects latency)

        Returns:
            self (for method chaining)

        Examples:
            >>> pk.with_params(chunk_secs=1.5, right_context_secs=1.0).transcribe("audio.wav")
            >>> pk.with_params(left_context_secs=20.0).transcribe("audio.wav")
        """
        # Create custom config with current values as defaults
        custom_config = AudioConfig(
            name="custom",
            chunk_secs=chunk_secs if chunk_secs is not None else self._audio_config.chunk_secs,
            left_context_secs=left_context_secs if left_context_secs is not None else self._audio_config.left_context_secs,
            right_context_secs=right_context_secs if right_context_secs is not None else self._audio_config.right_context_secs,
        )
        return self.with_config(custom_config)

    def __repr__(self) -> str:
        """
        Rich string representation for Python REPL.

        Returns:
            Formatted string with model info, config, and status
        """
        status = "ready" if self._initialized else "not loaded"
        return (
            f"Parakeet(model='{self.model_name}', device='{self.device}', "
            f"config='{self._audio_config.name}', status='{status}')"
        )

    def _repr_pretty_(self, p, cycle):
        """
        IPython pretty print representation.

        Displays a multi-line, formatted representation with quality indicators,
        latency info, and status.

        Args:
            p: IPython printer object
            cycle: Whether there's a circular reference
        """
        if cycle:
            p.text('Parakeet(...)')
            return

        status_icon = "✓" if self._initialized else "○"
        status_text = "Ready" if self._initialized else "Not loaded"

        lines = [
            f"Parakeet(model='{self.model_name}', device='{self.device}')",
            f"  Quality: {self._audio_config.quality_indicator} ({self._audio_config.name})",
            f"  Latency: ~{self._audio_config.latency:.1f}s",
            f"  Status: {status_icon} {status_text}",
        ]
        p.text('\n'.join(lines))

    def _repr_html_(self) -> str:
        """
        Jupyter HTML representation.

        Returns a styled HTML table with model information, configuration,
        and status.

        Returns:
            HTML string for Jupyter display
        """
        status_icon = "✅" if self._initialized else "⚪"
        status_text = "Ready" if self._initialized else "Not loaded"

        return f"""
        <div style="border: 1px solid #ccc; padding: 12px; border-radius: 5px; background-color: #f9f9f9;">
            <h4 style="margin-top: 0;">{status_icon} Parakeet</h4>
            <table style="border-collapse: collapse; width: 100%;">
                <tr>
                    <td style="padding: 4px; font-weight: bold;">Model:</td>
                    <td style="padding: 4px;">{self.model_name}</td>
                </tr>
                <tr>
                    <td style="padding: 4px; font-weight: bold;">Device:</td>
                    <td style="padding: 4px;">{self.device}</td>
                </tr>
                <tr>
                    <td style="padding: 4px; font-weight: bold;">Config:</td>
                    <td style="padding: 4px;">{self._audio_config.name}</td>
                </tr>
                <tr>
                    <td style="padding: 4px; font-weight: bold;">Quality:</td>
                    <td style="padding: 4px;">{self._audio_config.quality_indicator}</td>
                </tr>
                <tr>
                    <td style="padding: 4px; font-weight: bold;">Latency:</td>
                    <td style="padding: 4px;">~{self._audio_config.latency:.1f}s</td>
                </tr>
                <tr>
                    <td style="padding: 4px; font-weight: bold;">Status:</td>
                    <td style="padding: 4px;">{status_text}</td>
                </tr>
            </table>
        </div>
        """

    def transcribe(
        self,
        audio: Union[str, Path, np.ndarray, torch.Tensor],
        timestamps: Optional[bool] = None,
        _quiet: bool = False,
    ) -> TranscriptResult:
        """
        Transcribe audio file or array.

        Args:
            audio: Path to audio file or audio array
            timestamps: Whether to include timestamps (overrides config)
            _quiet: Internal parameter to suppress progress bars (used by LiveTranscriber)

        Returns:
            TranscriptResult with transcribed text, confidence, and metadata
        """
        self._initialize_model()

        if timestamps is None:
            timestamps = self._transcriber_config.timestamps

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
        import warnings

        # Mark the deprecated function as already warned in NeMo's decorator tracker
        # This prevents the deprecation warning from being shown
        try:
            from nemo.utils.decorators.deprecated import _PRINTED_WARNING
            _PRINTED_WARNING['_transcribe_output_processing'] = True
        except (ImportError, KeyError):
            pass  # If this fails, fall back to logger suppression

        # Get the NeMo logger and save its level + handler levels
        nemo_logger = logging.getLogger('nemo_logger')
        original_level = nemo_logger.level
        original_handler_levels = [(h, h.level) for h in nemo_logger.handlers]

        # Suppress progress bars and warnings if quiet mode
        if _quiet:
            # Redirect tqdm output to devnull
            from io import StringIO
            from contextlib import redirect_stderr, redirect_stdout

            devnull = StringIO()

            try:
                warnings.filterwarnings('ignore')
                nemo_logger.setLevel(logging.ERROR)
                for handler in nemo_logger.handlers:
                    handler.setLevel(logging.ERROR)
                with redirect_stderr(devnull), redirect_stdout(devnull):
                    output = self.model.transcribe([audio], timestamps=timestamps, return_hypotheses=True)
            finally:
                nemo_logger.setLevel(original_level)
                for handler, level in original_handler_levels:
                    handler.setLevel(level)
                warnings.filterwarnings('default')
        else:
            # Transcribe with return_hypotheses=True (the recommended approach)
            output = self.model.transcribe([audio], timestamps=timestamps, return_hypotheses=True)

        # Extract results from Hypothesis object
        # When return_hypotheses=True, output is a list of Hypothesis objects
        # output[0] is the Hypothesis for the first audio in the batch
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
            # Fallback if structure is unexpected
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
        audio = audio.to(self.device)

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
    ) -> List[TranscriptResult]:
        """Transcribe multiple audio files in batch.

        Args:
            audio_files: List of paths to audio files
            timestamps: Whether to include timestamps
            show_progress: Whether to show progress bar

        Returns:
            List of TranscriptResult objects
        """
        self._initialize_model()

        if timestamps is None:
            timestamps = self._transcriber_config.timestamps

        results = []
        iterator = tqdm(audio_files, desc="Transcribing") if show_progress else audio_files

        for audio_file in iterator:
            result = self.transcribe(audio_file, timestamps=timestamps)
            results.append(result)

        return results

    def listen(
        self,
        microphone: Optional['Microphone'] = None,
        output: Optional[Union[str, Path]] = None,
        chunk_duration: Optional[float] = None,
        verbose: bool = False,
        strategy: Optional['TranscriptionStrategy'] = None
    ) -> 'LiveTranscriber':
        """
        Start live transcription from microphone.

        Creates and starts a LiveTranscriber that records and transcribes
        audio continuously in the background.

        Args:
            microphone: Microphone to use (auto-detected if None)
            output: File path to save transcript (optional)
            chunk_duration: Duration of chunks to process (uses config default if None)
            verbose: Whether to print transcriptions to console (default: False)
            strategy: Transcription strategy to use (default: None = standard streaming)

        Returns:
            LiveTranscriber object (already started)

        Examples:
            >>> pk = Parakeet()
            >>> live = pk.listen()  # Silent mode (default)
            >>> # Speak into microphone...
            >>> live.text  # Get current transcript
            >>> live.stop()

            >>> # Verbose mode - prints transcriptions
            >>> live = pk.listen(verbose=True)
            🎤 Listening on: Built-in Microphone
               (Press Ctrl+C or call .stop() to end)
            [2.5s] Hello world
            [4.6s] This is a test

            >>> # Save to file
            >>> live = pk.listen(output="transcript.txt")

            >>> # Use overlapping window strategy
            >>> from parakeet_stream.strategies import OverlappingWindowStrategy
            >>> strategy = OverlappingWindowStrategy(chunk_duration=5.0, overlap=2.0)
            >>> live = pk.listen(strategy=strategy)

        Raises:
            RuntimeError: If model is not loaded
        """
        if not self._initialized:
            raise RuntimeError("Model not loaded. Call .load() first or use lazy=False")

        # Import here to avoid circular dependency
        from parakeet_stream.live import LiveTranscriber
        from parakeet_stream.microphone import Microphone

        # Use config's chunk_secs if not specified
        chunk_duration = chunk_duration or self._audio_config.chunk_secs

        # Create microphone if not provided
        if microphone is None:
            microphone = Microphone(sample_rate=self.sample_rate)

        live = LiveTranscriber(
            transcriber=self,
            microphone=microphone,
            output=output,
            chunk_duration=chunk_duration,
            verbose=verbose,
            strategy=strategy
        )
        live.start()
        return live
