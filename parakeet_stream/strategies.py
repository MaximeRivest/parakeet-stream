"""
Transcription strategies for different quality/speed tradeoffs.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import torch

from parakeet_stream.transcript import Segment


@dataclass
class TranscriptionChunk:
    """A chunk of audio with transcription result."""
    audio: np.ndarray
    start_time: float
    end_time: float
    text: str
    confidence: Optional[float] = None
    words: Optional[List[str]] = None


class TranscriptionStrategy(ABC):
    """Base class for transcription strategies."""

    @abstractmethod
    def process_stream(
        self,
        audio_stream: np.ndarray,
        transcriber,
        sample_rate: int
    ) -> List[Segment]:
        """
        Process audio stream and return segments.

        Args:
            audio_stream: Audio data as numpy array
            transcriber: Parakeet transcriber instance
            sample_rate: Sample rate in Hz

        Returns:
            List of transcription segments
        """
        pass


class DefaultStrategy(TranscriptionStrategy):
    """Default streaming strategy using NeMo's buffered streaming."""

    def process_stream(
        self,
        audio_stream: np.ndarray,
        transcriber,
        sample_rate: int
    ) -> List[Segment]:
        """Use the standard streaming approach."""
        # This uses the existing Parakeet.stream() method
        segments = []
        for chunk in transcriber.stream(audio_stream):
            if chunk.text.strip():
                segments.append(Segment(
                    text=chunk.text,
                    start_time=chunk.timestamp_start,
                    end_time=chunk.timestamp_end,
                    confidence=None
                ))
        return segments


class OverlappingWindowStrategy(TranscriptionStrategy):
    """
    Overlapping window strategy with word trimming.

    This strategy:
    1. Chunks audio with large overlaps
    2. Transcribes each chunk
    3. Removes first/last N words (which may be cut off)
    4. Keeps only the clean middle section
    5. Merges all middle sections for final transcript

    Example with 3s chunks, 1s overlap, trim 2 words:
        Chunk 1: [0s -------- 3s]
        Chunk 2:     [2s -------- 5s]
        Chunk 3:         [4s -------- 7s]

        After trimming edges, keep only:
        Chunk 1: [0.5s - 2.5s]  (middle)
        Chunk 2: [2.5s - 4.5s]  (middle)
        Chunk 3: [4.5s - 6.5s]  (middle)
    """

    def __init__(
        self,
        chunk_duration: float = 5.0,
        overlap: float = 2.0,
        trim_words: int = 2,
        speed_factor: float = 1.0
    ):
        """
        Initialize overlapping window strategy.

        Args:
            chunk_duration: Duration of each chunk in seconds
            overlap: Overlap between chunks in seconds (on each side)
            trim_words: Number of words to trim from start/end of each chunk
            speed_factor: Speed multiplier for audio (1.0 = normal, 2.0 = 2x speed)
                         Note: >1.0 may reduce quality but process faster
        """
        self.chunk_duration = chunk_duration
        self.overlap = overlap
        self.trim_words = trim_words
        self.speed_factor = speed_factor

        if overlap * 2 >= chunk_duration:
            raise ValueError(
                f"Overlap ({overlap}s x 2 = {overlap*2}s) must be less than "
                f"chunk_duration ({chunk_duration}s)"
            )

    def _change_speed(
        self,
        audio: np.ndarray,
        factor: float,
        sample_rate: int
    ) -> np.ndarray:
        """
        Change audio playback speed.

        Args:
            audio: Audio array
            factor: Speed factor (2.0 = 2x faster)
            sample_rate: Sample rate

        Returns:
            Speed-adjusted audio
        """
        if factor == 1.0:
            return audio

        try:
            import librosa
            return librosa.effects.time_stretch(audio, rate=factor)
        except ImportError:
            # Fallback: simple resampling (lower quality)
            from scipy import signal
            new_length = int(len(audio) / factor)
            return signal.resample(audio, new_length)

    def _split_into_words(self, text: str) -> List[str]:
        """Split text into words, preserving punctuation."""
        return text.split()

    def _trim_edges(
        self,
        text: str,
        trim_start: int,
        trim_end: int
    ) -> tuple[str, int, int]:
        """
        Trim words from start and end of text.

        Args:
            text: Text to trim
            trim_start: Number of words to trim from start
            trim_end: Number of words to trim from end

        Returns:
            Tuple of (trimmed_text, words_removed_start, words_removed_end)
        """
        words = self._split_into_words(text)

        if len(words) <= trim_start + trim_end:
            # Not enough words to trim
            return "", 0, 0

        trimmed = words[trim_start:-trim_end] if trim_end > 0 else words[trim_start:]
        return " ".join(trimmed), trim_start, trim_end

    def _estimate_word_timing(
        self,
        text: str,
        start_time: float,
        end_time: float,
        words_trimmed_start: int
    ) -> float:
        """
        Estimate when trimmed text actually starts.

        Args:
            text: Original full text
            start_time: Chunk start time
            end_time: Chunk end time
            words_trimmed_start: How many words were trimmed from start

        Returns:
            Estimated start time for trimmed text
        """
        words = self._split_into_words(text)
        if not words:
            return start_time

        # Assume uniform word distribution
        duration = end_time - start_time
        time_per_word = duration / len(words)

        return start_time + (words_trimmed_start * time_per_word)

    def process_stream(
        self,
        audio_stream: np.ndarray,
        transcriber,
        sample_rate: int
    ) -> List[Segment]:
        """
        Process audio with overlapping windows.

        Args:
            audio_stream: Full audio stream
            transcriber: Parakeet transcriber
            sample_rate: Sample rate in Hz

        Returns:
            List of cleaned, non-overlapping segments
        """
        duration = len(audio_stream) / sample_rate
        chunk_samples = int(self.chunk_duration * sample_rate)
        step_samples = int((self.chunk_duration - 2 * self.overlap) * sample_rate)

        chunks = []
        current_pos = 0

        # Split into overlapping chunks
        while current_pos < len(audio_stream):
            end_pos = min(current_pos + chunk_samples, len(audio_stream))
            chunk_audio = audio_stream[current_pos:end_pos]

            # Apply speed change if requested
            if self.speed_factor != 1.0:
                chunk_audio = self._change_speed(
                    chunk_audio,
                    self.speed_factor,
                    sample_rate
                )

            # Calculate timing
            start_time = current_pos / sample_rate
            end_time = end_pos / sample_rate

            # Transcribe chunk
            result = transcriber.transcribe(chunk_audio, _quiet=True)

            # Skip empty transcriptions
            if not result.text.strip():
                current_pos += step_samples
                continue

            # Note: NeMo returns negative log-likelihood scores for confidence
            # So we don't filter on confidence here (would need to convert to probability first)

            chunks.append(TranscriptionChunk(
                audio=chunk_audio,
                start_time=start_time,
                end_time=end_time,
                text=result.text,
                confidence=result.confidence,
                words=self._split_into_words(result.text)
            ))

            current_pos += step_samples

            # Break if we've reached the end
            if end_pos >= len(audio_stream):
                break

        # Trim edges and create segments
        segments = []

        for i, chunk in enumerate(chunks):
            # First chunk: only trim end
            # Last chunk: only trim start
            # Middle chunks: trim both
            trim_start = 0 if i == 0 else self.trim_words
            trim_end = 0 if i == len(chunks) - 1 else self.trim_words

            trimmed_text, words_removed_start, words_removed_end = self._trim_edges(
                chunk.text,
                trim_start,
                trim_end
            )

            if not trimmed_text.strip():
                continue

            # Estimate actual timing for trimmed text
            segment_start = self._estimate_word_timing(
                chunk.text,
                chunk.start_time,
                chunk.end_time,
                words_removed_start
            )

            # Estimate end time
            words_in_trimmed = len(self._split_into_words(trimmed_text))
            total_words = len(chunk.words) if chunk.words else 1
            duration_ratio = words_in_trimmed / total_words
            segment_duration = (chunk.end_time - chunk.start_time) * duration_ratio
            segment_end = segment_start + segment_duration

            segments.append(Segment(
                text=trimmed_text,
                start_time=segment_start,
                end_time=segment_end,
                confidence=chunk.confidence
            ))

        return segments

    def __repr__(self) -> str:
        return (
            f"OverlappingWindowStrategy(chunk={self.chunk_duration}s, "
            f"overlap={self.overlap}s, trim_words={self.trim_words}, "
            f"speed={self.speed_factor}x)"
        )


class ConsensusStrategy(OverlappingWindowStrategy):
    """
    Advanced strategy that transcribes overlapping sections multiple times
    and uses consensus/voting to pick the best words.

    This further improves quality by:
    1. Transcribing the same audio segment from multiple chunks
    2. Comparing the overlapping transcriptions
    3. Using word-level confidence or voting to pick best version
    """

    def __init__(
        self,
        chunk_duration: float = 5.0,
        overlap: float = 2.0,
        trim_words: int = 1,
        min_agreement: float = 0.5
    ):
        """
        Initialize consensus strategy.

        Args:
            chunk_duration: Duration of each chunk
            overlap: Large overlap for consensus
            trim_words: Words to trim from edges
            min_agreement: Minimum agreement ratio for consensus (0.5 = majority)
        """
        super().__init__(
            chunk_duration=chunk_duration,
            overlap=overlap,
            trim_words=trim_words
        )
        self.min_agreement = min_agreement

    def _find_consensus_words(
        self,
        alternatives: List[List[str]]
    ) -> List[str]:
        """
        Find consensus among multiple transcriptions of the same segment.

        Args:
            alternatives: List of word lists from different transcriptions

        Returns:
            Consensus word list
        """
        if not alternatives:
            return []
        if len(alternatives) == 1:
            return alternatives[0]

        # Align and vote on each position
        max_len = max(len(alt) for alt in alternatives)
        consensus = []

        for pos in range(max_len):
            # Get words at this position from all alternatives
            words_at_pos = []
            for alt in alternatives:
                if pos < len(alt):
                    words_at_pos.append(alt[pos].lower())

            if not words_at_pos:
                continue

            # Vote: pick most common word
            from collections import Counter
            word_counts = Counter(words_at_pos)
            most_common, count = word_counts.most_common(1)[0]

            # Check if it meets minimum agreement
            if count / len(words_at_pos) >= self.min_agreement:
                # Use original case from first occurrence
                for alt in alternatives:
                    if pos < len(alt) and alt[pos].lower() == most_common:
                        consensus.append(alt[pos])
                        break
            else:
                # No consensus, use first alternative
                consensus.append(alternatives[0][pos])

        return consensus

    def __repr__(self) -> str:
        return (
            f"ConsensusStrategy(chunk={self.chunk_duration}s, "
            f"overlap={self.overlap}s, min_agreement={self.min_agreement})"
        )
