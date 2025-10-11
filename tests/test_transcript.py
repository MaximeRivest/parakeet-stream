"""
Tests for TranscriptResult, Segment, and TranscriptBuffer classes.
"""
import pytest
from pathlib import Path
from parakeet_stream.transcript import TranscriptResult, Segment, TranscriptBuffer


class TestTranscriptResultCreation:
    """Tests for TranscriptResult creation and basic properties"""

    def test_minimal_creation(self):
        """Test creating result with only text"""
        result = TranscriptResult(text="Hello world")
        assert result.text == "Hello world"
        assert result.confidence is None
        assert result.duration is None
        assert result.timestamps is None

    def test_creation_with_confidence(self):
        """Test creating result with confidence"""
        result = TranscriptResult(text="Hello", confidence=0.95)
        assert result.text == "Hello"
        assert result.confidence == 0.95

    def test_creation_with_duration(self):
        """Test creating result with duration"""
        result = TranscriptResult(text="Hello", duration=3.5)
        assert result.text == "Hello"
        assert result.duration == 3.5

    def test_creation_with_timestamps(self):
        """Test creating result with timestamps"""
        timestamps = [
            {"word": "hello", "start": 0.0, "end": 0.5},
            {"word": "world", "start": 0.5, "end": 1.0},
        ]
        result = TranscriptResult(text="hello world", timestamps=timestamps)
        assert result.text == "hello world"
        assert result.timestamps == timestamps

    def test_creation_with_all_fields(self):
        """Test creating result with all fields"""
        timestamps = [{"word": "test", "start": 0.0, "end": 0.5}]
        result = TranscriptResult(
            text="test",
            confidence=0.92,
            duration=2.3,
            timestamps=timestamps
        )
        assert result.text == "test"
        assert result.confidence == 0.92
        assert result.duration == 2.3
        assert result.timestamps == timestamps


class TestTranscriptResultProperties:
    """Tests for computed properties"""

    def test_word_count_simple(self):
        """Test word_count with simple text"""
        result = TranscriptResult(text="hello world")
        assert result.word_count == 2

    def test_word_count_single_word(self):
        """Test word_count with single word"""
        result = TranscriptResult(text="hello")
        assert result.word_count == 1

    def test_word_count_empty(self):
        """Test word_count with empty text"""
        result = TranscriptResult(text="")
        # Empty string split() returns [] which has length 0
        assert result.word_count == 0

    def test_word_count_multiple_spaces(self):
        """Test word_count with multiple spaces"""
        result = TranscriptResult(text="hello  world  test")
        # Multiple spaces create empty strings in split
        assert result.word_count >= 3

    def test_has_timestamps_true(self):
        """Test has_timestamps when timestamps exist"""
        timestamps = [{"word": "test", "start": 0.0, "end": 0.5}]
        result = TranscriptResult(text="test", timestamps=timestamps)
        assert result.has_timestamps is True

    def test_has_timestamps_false_none(self):
        """Test has_timestamps when timestamps is None"""
        result = TranscriptResult(text="test")
        assert result.has_timestamps is False

    def test_has_timestamps_false_empty(self):
        """Test has_timestamps when timestamps is empty list"""
        result = TranscriptResult(text="test", timestamps=[])
        assert result.has_timestamps is False


class TestTranscriptResultRepr:
    """Tests for __repr__ method"""

    def test_repr_basic(self):
        """Test __repr__ with basic text"""
        result = TranscriptResult(text="Hello world")
        repr_str = repr(result)

        assert "TranscriptResult" in repr_str
        assert "Hello world" in repr_str

    def test_repr_with_confidence(self):
        """Test __repr__ includes confidence"""
        result = TranscriptResult(text="Hello", confidence=0.95)
        repr_str = repr(result)

        assert "confidence=0.95" in repr_str

    def test_repr_with_duration(self):
        """Test __repr__ includes duration"""
        result = TranscriptResult(text="Hello", duration=3.5)
        repr_str = repr(result)

        assert "duration=3.5s" in repr_str

    def test_repr_truncates_long_text(self):
        """Test __repr__ truncates very long text"""
        long_text = "a" * 100
        result = TranscriptResult(text=long_text)
        repr_str = repr(result)

        # Should be truncated with ...
        assert len(repr_str) < len(long_text) + 50
        assert "..." in repr_str

    def test_repr_all_fields(self):
        """Test __repr__ with all fields"""
        result = TranscriptResult(
            text="Test",
            confidence=0.88,
            duration=2.0,
            timestamps=[{"word": "test"}]
        )
        repr_str = repr(result)

        assert "Test" in repr_str
        assert "confidence=0.88" in repr_str
        assert "duration=2.0s" in repr_str


class TestTranscriptResultReprPretty:
    """Tests for _repr_pretty_ method (IPython)"""

    def test_repr_pretty_basic(self):
        """Test _repr_pretty_ basic output"""
        result = TranscriptResult(text="Hello world")

        class MockPrinter:
            def __init__(self):
                self.text_content = ""

            def text(self, content):
                self.text_content = content

        printer = MockPrinter()
        result._repr_pretty_(printer, cycle=False)

        assert "📝" in printer.text_content
        assert "Hello world" in printer.text_content

    def test_repr_pretty_with_confidence(self):
        """Test _repr_pretty_ shows confidence"""
        result = TranscriptResult(text="Test", confidence=0.95)

        class MockPrinter:
            def __init__(self):
                self.text_content = ""

            def text(self, content):
                self.text_content = content

        printer = MockPrinter()
        result._repr_pretty_(printer, cycle=False)

        assert "Confidence:" in printer.text_content
        assert "95%" in printer.text_content
        assert "●" in printer.text_content

    def test_repr_pretty_with_duration(self):
        """Test _repr_pretty_ shows duration"""
        result = TranscriptResult(text="Test", duration=5.5)

        class MockPrinter:
            def __init__(self):
                self.text_content = ""

            def text(self, content):
                self.text_content = content

        printer = MockPrinter()
        result._repr_pretty_(printer, cycle=False)

        assert "Duration:" in printer.text_content
        assert "5.5s" in printer.text_content

    def test_repr_pretty_with_timestamps(self):
        """Test _repr_pretty_ shows word count from timestamps"""
        timestamps = [
            {"word": "hello"},
            {"word": "world"},
            {"word": "test"},
        ]
        result = TranscriptResult(text="hello world test", timestamps=timestamps)

        class MockPrinter:
            def __init__(self):
                self.text_content = ""

            def text(self, content):
                self.text_content = content

        printer = MockPrinter()
        result._repr_pretty_(printer, cycle=False)

        assert "Words: 3" in printer.text_content

    def test_repr_pretty_cycle(self):
        """Test _repr_pretty_ handles circular reference"""
        result = TranscriptResult(text="Test")

        class MockPrinter:
            def __init__(self):
                self.text_content = ""

            def text(self, content):
                self.text_content = content

        printer = MockPrinter()
        result._repr_pretty_(printer, cycle=True)

        assert printer.text_content == "TranscriptResult(...)"


class TestTranscriptResultReprHtml:
    """Tests for _repr_html_ method (Jupyter)"""

    def test_repr_html_basic(self):
        """Test _repr_html_ returns HTML"""
        result = TranscriptResult(text="Hello world")
        html = result._repr_html_()

        assert isinstance(html, str)
        assert "<div" in html
        assert "<table" in html
        assert "Hello world" in html

    def test_repr_html_has_emoji(self):
        """Test _repr_html_ includes emoji"""
        result = TranscriptResult(text="Test")
        html = result._repr_html_()

        assert "📝" in html

    def test_repr_html_escapes_text(self):
        """Test _repr_html_ escapes HTML characters"""
        result = TranscriptResult(text="<script>alert('xss')</script>")
        html = result._repr_html_()

        assert "<script>" not in html
        assert "&lt;script&gt;" in html

    def test_repr_html_with_confidence(self):
        """Test _repr_html_ shows confidence"""
        result = TranscriptResult(text="Test", confidence=0.87)
        html = result._repr_html_()

        assert "Confidence:" in html
        assert "87%" in html
        assert "●" in html

    def test_repr_html_with_duration(self):
        """Test _repr_html_ shows duration"""
        result = TranscriptResult(text="Test", duration=12.5)
        html = result._repr_html_()

        assert "Duration:" in html
        assert "12.5s" in html

    def test_repr_html_with_timestamps(self):
        """Test _repr_html_ shows word count"""
        timestamps = [{"word": "a"}, {"word": "b"}]
        result = TranscriptResult(text="a b", timestamps=timestamps)
        html = result._repr_html_()

        assert "Words:" in html
        assert "2" in html


class TestTranscriptResultIntegration:
    """Integration tests for TranscriptResult"""

    def test_all_display_methods_consistent(self):
        """Test all display methods show consistent info"""
        result = TranscriptResult(
            text="Integration test",
            confidence=0.92,
            duration=3.5,
        )

        # Get all representations
        repr_str = repr(result)

        class MockPrinter:
            def __init__(self):
                self.text_content = ""

            def text(self, content):
                self.text_content = content

        printer = MockPrinter()
        result._repr_pretty_(printer, cycle=False)
        pretty_str = printer.text_content

        html_str = result._repr_html_()

        # All should mention the text
        assert "Integration test" in repr_str
        assert "Integration test" in pretty_str
        assert "Integration test" in html_str

        # All should show confidence
        assert "0.92" in repr_str
        assert "92%" in pretty_str
        assert "92%" in html_str

        # All should show duration
        assert "3.5s" in repr_str
        assert "3.5s" in pretty_str
        assert "3.5s" in html_str

    def test_realistic_transcription(self):
        """Test with realistic transcription data"""
        result = TranscriptResult(
            text="This is a realistic transcription of spoken audio.",
            confidence=0.94,
            duration=4.8,
            timestamps=[
                {"word": "This", "start": 0.0, "end": 0.2},
                {"word": "is", "start": 0.2, "end": 0.4},
                {"word": "a", "start": 0.4, "end": 0.5},
                {"word": "realistic", "start": 0.5, "end": 1.2},
                {"word": "transcription", "start": 1.2, "end": 2.0},
            ]
        )

        assert result.word_count == 8
        assert result.has_timestamps is True
        assert result.confidence > 0.9
        assert result.duration < 5.0

        # Check repr works
        repr_str = repr(result)
        assert "TranscriptResult" in repr_str


class TestSegment:
    """Tests for Segment dataclass"""

    def test_segment_creation(self):
        """Test creating a segment"""
        seg = Segment(text="Hello", start_time=0.0, end_time=1.0)
        assert seg.text == "Hello"
        assert seg.start_time == 0.0
        assert seg.end_time == 1.0
        assert seg.confidence is None

    def test_segment_with_confidence(self):
        """Test segment with confidence"""
        seg = Segment(text="Test", start_time=1.0, end_time=2.0, confidence=0.95)
        assert seg.confidence == 0.95

    def test_segment_duration_property(self):
        """Test duration property"""
        seg = Segment(text="Test", start_time=1.5, end_time=4.5)
        assert seg.duration == 3.0


class TestTranscriptBufferCreation:
    """Tests for TranscriptBuffer creation"""

    def test_empty_buffer(self):
        """Test creating empty buffer"""
        buffer = TranscriptBuffer()
        assert len(buffer) == 0
        assert buffer.text == ""

    def test_buffer_stats_empty(self):
        """Test stats on empty buffer"""
        buffer = TranscriptBuffer()
        stats = buffer.stats
        assert stats['segments'] == 0
        assert stats['duration'] == 0.0
        assert stats['words'] == 0


class TestTranscriptBufferAppend:
    """Tests for appending segments"""

    def test_append_single_segment(self):
        """Test appending one segment"""
        buffer = TranscriptBuffer()
        seg = Segment(text="Hello", start_time=0.0, end_time=1.0, confidence=0.95)
        buffer.append(seg)

        assert len(buffer) == 1
        assert buffer.text == "Hello"

    def test_append_multiple_segments(self):
        """Test appending multiple segments"""
        buffer = TranscriptBuffer()
        buffer.append(Segment("Hello", 0.0, 1.0))
        buffer.append(Segment("world", 1.0, 2.0))

        assert len(buffer) == 2
        assert buffer.text == "Hello world"

    def test_append_joins_with_space(self):
        """Test that segments are joined with spaces"""
        buffer = TranscriptBuffer()
        buffer.append(Segment("First", 0.0, 1.0))
        buffer.append(Segment("second", 1.0, 2.0))
        buffer.append(Segment("third", 2.0, 3.0))

        assert buffer.text == "First second third"


class TestTranscriptBufferAccess:
    """Tests for accessing segments"""

    def test_getitem(self):
        """Test accessing segment by index"""
        buffer = TranscriptBuffer()
        seg1 = Segment("First", 0.0, 1.0)
        seg2 = Segment("Second", 1.0, 2.0)
        buffer.append(seg1)
        buffer.append(seg2)

        assert buffer[0] == seg1
        assert buffer[1] == seg2

    def test_segments_property(self):
        """Test segments property returns copy"""
        buffer = TranscriptBuffer()
        buffer.append(Segment("Test", 0.0, 1.0))

        segs = buffer.segments
        assert len(segs) == 1
        # Modifying copy shouldn't affect buffer
        segs.append(Segment("Extra", 1.0, 2.0))
        assert len(buffer) == 1

    def test_head(self):
        """Test head method"""
        buffer = TranscriptBuffer()
        for i in range(10):
            buffer.append(Segment(f"Seg{i}", float(i), float(i+1)))

        head = buffer.head(3)
        assert len(head) == 3
        assert head[0].text == "Seg0"
        assert head[2].text == "Seg2"

    def test_tail(self):
        """Test tail method"""
        buffer = TranscriptBuffer()
        for i in range(10):
            buffer.append(Segment(f"Seg{i}", float(i), float(i+1)))

        tail = buffer.tail(3)
        assert len(tail) == 3
        assert tail[0].text == "Seg7"
        assert tail[2].text == "Seg9"


class TestTranscriptBufferStats:
    """Tests for stats calculation"""

    def test_stats_with_segments(self):
        """Test stats calculation"""
        buffer = TranscriptBuffer()
        buffer.append(Segment("Hello world", 0.0, 2.0, confidence=0.95))
        buffer.append(Segment("Test phrase", 2.0, 4.0, confidence=0.93))

        stats = buffer.stats
        assert stats['segments'] == 2
        assert stats['duration'] == 4.0
        assert stats['words'] == 4  # "Hello world Test phrase"
        assert 0.93 < stats['avg_confidence'] < 0.95

    def test_stats_without_confidence(self):
        """Test stats when no confidence provided"""
        buffer = TranscriptBuffer()
        buffer.append(Segment("Test", 0.0, 1.0))

        stats = buffer.stats
        assert stats['avg_confidence'] == 0.0


class TestTranscriptBufferSave:
    """Tests for saving buffer"""

    def test_save_to_file(self, tmp_path):
        """Test saving buffer to JSON file"""
        buffer = TranscriptBuffer()
        buffer.append(Segment("Test", 0.0, 1.0, confidence=0.95))

        output_path = tmp_path / "transcript.json"
        buffer.save(output_path)

        assert output_path.exists()

    def test_save_preserves_data(self, tmp_path):
        """Test that saved data can be loaded"""
        import json

        buffer = TranscriptBuffer()
        buffer.append(Segment("First", 0.0, 1.0))
        buffer.append(Segment("Second", 1.0, 2.0))

        output_path = tmp_path / "transcript.json"
        buffer.save(output_path)

        # Load and verify
        with open(output_path) as f:
            data = json.load(f)

        assert data['text'] == "First Second"
        assert len(data['segments']) == 2
        assert data['stats']['segments'] == 2


class TestTranscriptBufferDisplay:
    """Tests for display methods"""

    def test_repr_basic(self):
        """Test __repr__"""
        buffer = TranscriptBuffer()
        buffer.append(Segment("Test", 0.0, 1.0))

        repr_str = repr(buffer)
        assert "TranscriptBuffer" in repr_str
        assert "segments=1" in repr_str

    def test_repr_pretty_basic(self):
        """Test _repr_pretty_"""
        buffer = TranscriptBuffer()
        buffer.append(Segment("Test", 0.0, 1.0))

        class MockPrinter:
            def __init__(self):
                self.text_content = ""

            def text(self, content):
                self.text_content = content

        printer = MockPrinter()
        buffer._repr_pretty_(printer, cycle=False)

        assert "📄" in printer.text_content
        assert "TranscriptBuffer" in printer.text_content
        assert "Latest:" in printer.text_content

    def test_repr_html_basic(self):
        """Test _repr_html_"""
        buffer = TranscriptBuffer()
        buffer.append(Segment("Test", 0.0, 1.0))

        html = buffer._repr_html_()

        assert "<div" in html
        assert "TranscriptBuffer" in html
        assert "Segments:" in html
