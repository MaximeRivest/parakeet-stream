"""
Tests for transcription strategies.
"""
import numpy as np
import pytest

from parakeet_stream.strategies import (
    DefaultStrategy,
    OverlappingWindowStrategy,
    ConsensusStrategy,
    TranscriptionChunk,
)


class TestOverlappingWindowStrategy:
    """Tests for OverlappingWindowStrategy."""

    def test_init_valid_params(self):
        """Test initialization with valid parameters."""
        strategy = OverlappingWindowStrategy(
            chunk_duration=5.0,
            overlap=2.0,
            trim_words=2
        )
        assert strategy.chunk_duration == 5.0
        assert strategy.overlap == 2.0
        assert strategy.trim_words == 2

    def test_init_invalid_overlap(self):
        """Test that overlap must be less than half of chunk duration."""
        with pytest.raises(ValueError, match="must be less than"):
            OverlappingWindowStrategy(
                chunk_duration=4.0,
                overlap=2.5,  # 2.5 * 2 = 5.0 > 4.0
                trim_words=2
            )

    def test_split_into_words(self):
        """Test word splitting."""
        strategy = OverlappingWindowStrategy()
        words = strategy._split_into_words("Hello world this is a test")
        assert words == ["Hello", "world", "this", "is", "a", "test"]

    def test_trim_edges(self):
        """Test edge trimming."""
        strategy = OverlappingWindowStrategy()

        # Trim 2 from start, 2 from end
        text, start_count, end_count = strategy._trim_edges(
            "one two three four five six",
            trim_start=2,
            trim_end=2
        )
        assert text == "three four"
        assert start_count == 2
        assert end_count == 2

    def test_trim_edges_not_enough_words(self):
        """Test trimming when there aren't enough words."""
        strategy = OverlappingWindowStrategy()

        # Only 3 words, can't trim 2 from each side
        text, start_count, end_count = strategy._trim_edges(
            "one two three",
            trim_start=2,
            trim_end=2
        )
        assert text == ""
        assert start_count == 0
        assert end_count == 0

    def test_trim_edges_no_end_trim(self):
        """Test trimming only from start."""
        strategy = OverlappingWindowStrategy()

        text, start_count, end_count = strategy._trim_edges(
            "one two three four five",
            trim_start=2,
            trim_end=0
        )
        assert text == "three four five"
        assert start_count == 2
        assert end_count == 0

    def test_estimate_word_timing(self):
        """Test word timing estimation."""
        strategy = OverlappingWindowStrategy()

        # 5 words over 10 seconds = 2 seconds per word
        # Trimming 2 words from start = 4 seconds offset
        start_time = strategy._estimate_word_timing(
            text="one two three four five",
            start_time=0.0,
            end_time=10.0,
            words_trimmed_start=2
        )
        assert start_time == pytest.approx(4.0, rel=0.1)

    def test_change_speed_no_change(self):
        """Test speed change with factor 1.0 (no change)."""
        strategy = OverlappingWindowStrategy()

        audio = np.random.randn(16000)
        result = strategy._change_speed(audio, 1.0, 16000)

        # Should be identical
        assert np.array_equal(audio, result)

    def test_repr(self):
        """Test string representation."""
        strategy = OverlappingWindowStrategy(
            chunk_duration=5.0,
            overlap=2.0,
            trim_words=2,
            speed_factor=1.5
        )
        repr_str = repr(strategy)
        assert "OverlappingWindowStrategy" in repr_str
        assert "5.0" in repr_str
        assert "2.0" in repr_str
        assert "1.5" in repr_str


class TestConsensusStrategy:
    """Tests for ConsensusStrategy."""

    def test_init(self):
        """Test initialization."""
        strategy = ConsensusStrategy(
            chunk_duration=5.0,
            overlap=2.0,
            min_agreement=0.6
        )
        assert strategy.chunk_duration == 5.0
        assert strategy.overlap == 2.0
        assert strategy.min_agreement == 0.6

    def test_find_consensus_single(self):
        """Test consensus with single alternative."""
        strategy = ConsensusStrategy()
        alternatives = [["Hello", "world"]]
        consensus = strategy._find_consensus_words(alternatives)
        assert consensus == ["Hello", "world"]

    def test_find_consensus_agreement(self):
        """Test consensus with agreement."""
        strategy = ConsensusStrategy(min_agreement=0.5)

        alternatives = [
            ["Hello", "world", "this"],
            ["Hello", "world", "that"],
            ["Hello", "world", "this"],
        ]

        consensus = strategy._find_consensus_words(alternatives)

        # First two words should agree (3/3)
        assert consensus[0] == "Hello"
        assert consensus[1] == "world"

        # Third word: "this" appears 2/3 times (66% > 50%)
        assert consensus[2] == "this"

    def test_find_consensus_no_alternatives(self):
        """Test consensus with no alternatives."""
        strategy = ConsensusStrategy()
        alternatives = []
        consensus = strategy._find_consensus_words(alternatives)
        assert consensus == []

    def test_repr(self):
        """Test string representation."""
        strategy = ConsensusStrategy(
            chunk_duration=5.0,
            overlap=2.0,
            min_agreement=0.6
        )
        repr_str = repr(strategy)
        assert "ConsensusStrategy" in repr_str
        assert "5.0" in repr_str
        assert "2.0" in repr_str
        assert "0.6" in repr_str


class TestTranscriptionChunk:
    """Tests for TranscriptionChunk dataclass."""

    def test_creation(self):
        """Test chunk creation."""
        audio = np.random.randn(16000)
        chunk = TranscriptionChunk(
            audio=audio,
            start_time=0.0,
            end_time=1.0,
            text="Hello world",
            confidence=0.95,
            words=["Hello", "world"]
        )

        assert len(chunk.audio) == 16000
        assert chunk.start_time == 0.0
        assert chunk.end_time == 1.0
        assert chunk.text == "Hello world"
        assert chunk.confidence == 0.95
        assert chunk.words == ["Hello", "world"]


# Integration tests (slower, require model)
@pytest.mark.slow
class TestStrategyIntegration:
    """Integration tests with actual Parakeet model."""

    def test_overlapping_strategy_with_audio(self):
        """Test overlapping strategy with real audio."""
        from parakeet_stream import Parakeet

        pk = Parakeet()
        strategy = OverlappingWindowStrategy(
            chunk_duration=3.0,
            overlap=1.0,
            trim_words=1
        )

        # Create test audio (3 seconds of noise)
        audio = np.random.randn(16000 * 3)

        # Process
        segments = strategy.process_stream(audio, pk, 16000)

        # Should produce at least one segment
        assert isinstance(segments, list)
        # May be empty if audio is just noise, which is fine

    def test_default_strategy(self):
        """Test default strategy."""
        from parakeet_stream import Parakeet

        pk = Parakeet()
        strategy = DefaultStrategy()

        # Create test audio
        audio = np.random.randn(16000 * 3)

        # Process
        segments = strategy.process_stream(audio, pk, 16000)

        assert isinstance(segments, list)
