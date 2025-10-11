"""
Tests for display helpers.
"""
import pytest
from parakeet_stream.display import (
    format_duration,
    format_confidence,
    create_progress_bar,
    format_file_size,
    format_sample_rate,
    format_timestamp,
    truncate_text,
    table_row,
    RichRepr,
)


class TestFormatDuration:
    """Tests for format_duration"""

    def test_seconds_only(self):
        assert format_duration(5.2) == "5.2s"
        assert format_duration(45.0) == "45.0s"

    def test_minutes_and_seconds(self):
        assert format_duration(65.5) == "1m 5.5s"
        assert format_duration(125.0) == "2m 5.0s"

    def test_hours_minutes_seconds(self):
        assert format_duration(3665) == "1h 1m 5.0s"
        assert format_duration(7325.3) == "2h 2m 5.3s"

    def test_zero(self):
        assert format_duration(0) == "0.0s"

    def test_negative(self):
        assert format_duration(-10) == "0s"

    def test_large_duration(self):
        result = format_duration(10000)
        assert "h" in result


class TestFormatConfidence:
    """Tests for format_confidence"""

    def test_perfect_confidence(self):
        result = format_confidence(1.0)
        assert "100%" in result
        assert "●●●●●" in result

    def test_high_confidence(self):
        result = format_confidence(0.95)
        assert "95%" in result
        assert "●●●●●" in result

    def test_medium_confidence(self):
        result = format_confidence(0.73)
        assert "73%" in result
        assert "●●●●○" in result  # 0.73 rounds to 4/5

    def test_low_confidence(self):
        result = format_confidence(0.25)
        assert "25%" in result
        assert "○○○" in result or "●○○○○" in result

    def test_none_confidence(self):
        result = format_confidence(None)
        assert result == "N/A"

    def test_confidence_bounds(self):
        # Test edge cases
        result0 = format_confidence(0.0)
        assert "0%" in result0

        result1 = format_confidence(1.0)
        assert "100%" in result1


class TestCreateProgressBar:
    """Tests for create_progress_bar"""

    def test_half_progress(self):
        result = create_progress_bar(50, 100, width=20)
        assert "[" in result
        assert "]" in result
        assert "50%" in result

    def test_full_progress(self):
        result = create_progress_bar(100, 100, width=10)
        assert "100%" in result

    def test_zero_progress(self):
        result = create_progress_bar(0, 100, width=10)
        assert "0%" in result

    def test_no_percentage(self):
        result = create_progress_bar(50, 100, width=10, show_percentage=False)
        assert "%" not in result
        assert "[" in result

    def test_custom_width(self):
        result = create_progress_bar(75, 100, width=40)
        assert len(result) > 40  # Bar + brackets + percentage

    def test_zero_total(self):
        result = create_progress_bar(10, 0, width=10)
        assert "0%" in result

    def test_over_100_percent(self):
        result = create_progress_bar(150, 100, width=10)
        assert "100%" in result  # Should clamp to 100%


class TestFormatFileSize:
    """Tests for format_file_size"""

    def test_bytes(self):
        assert format_file_size(500) == "500 B"

    def test_kilobytes(self):
        assert format_file_size(1024) == "1.0 KB"
        assert format_file_size(2048) == "2.0 KB"

    def test_megabytes(self):
        result = format_file_size(1536000)
        assert "MB" in result
        assert "1." in result

    def test_gigabytes(self):
        result = format_file_size(2 * 1024 * 1024 * 1024)
        assert "GB" in result


class TestFormatSampleRate:
    """Tests for format_sample_rate"""

    def test_common_rates(self):
        assert format_sample_rate(16000) == "16 kHz"
        assert format_sample_rate(44100) == "44.1 kHz"
        assert format_sample_rate(48000) == "48 kHz"

    def test_low_rate(self):
        assert format_sample_rate(8000) == "8 kHz"

    def test_below_1khz(self):
        result = format_sample_rate(500)
        assert "Hz" in result
        assert "kHz" not in result


class TestFormatTimestamp:
    """Tests for format_timestamp"""

    def test_seconds_only(self):
        assert format_timestamp(12.5) == "00:12.5"

    def test_minutes_and_seconds(self):
        assert format_timestamp(83.2) == "01:23.2"

    def test_zero(self):
        result = format_timestamp(0)
        assert result.startswith("00:")


class TestTruncateText:
    """Tests for truncate_text"""

    def test_short_text(self):
        text = "Short"
        assert truncate_text(text, 10) == "Short"

    def test_long_text(self):
        text = "This is a very long sentence"
        result = truncate_text(text, 10)
        assert len(result) == 10
        assert result.endswith("...")

    def test_exact_length(self):
        text = "Exactly10!"
        assert truncate_text(text, 10) == "Exactly10!"

    def test_custom_suffix(self):
        text = "Long text here"
        result = truncate_text(text, 8, suffix="…")
        assert result.endswith("…")


class TestTableRow:
    """Tests for table_row"""

    def test_basic_row(self):
        result = table_row(['A', 'B', 'C'])
        assert 'A' in result
        assert 'B' in result
        assert 'C' in result

    def test_custom_widths(self):
        result = table_row(['A', 'B', 'C'], widths=[10, 10, 10])
        # Should be padded to width
        assert len(result) >= 30


class TestRichRepr:
    """Tests for RichRepr mixin"""

    def test_rich_repr_mixin(self):
        class TestClass(RichRepr):
            def __repr__(self):
                return "TestClass()"

        obj = TestClass()
        assert repr(obj) == "TestClass()"

    def test_rich_repr_pretty(self):
        class TestClass(RichRepr):
            def __repr__(self):
                return "TestClass()"

        obj = TestClass()

        # Mock IPython printer
        class MockPrinter:
            def __init__(self):
                self.text_value = None

            def text(self, value):
                self.text_value = value

        printer = MockPrinter()
        obj._repr_pretty_(printer, False)
        assert printer.text_value == "TestClass()"

    def test_rich_repr_cycle(self):
        class TestClass(RichRepr):
            def __repr__(self):
                return "TestClass()"

        obj = TestClass()

        class MockPrinter:
            def __init__(self):
                self.text_value = None

            def text(self, value):
                self.text_value = value

        printer = MockPrinter()
        obj._repr_pretty_(printer, True)  # Cycle detection
        assert "..." in printer.text_value

    def test_rich_repr_html_default(self):
        class TestClass(RichRepr):
            pass

        obj = TestClass()
        assert obj._repr_html_() is None  # Default returns None


class TestIntegration:
    """Integration tests for display helpers"""

    def test_format_transcription_result(self):
        """Test formatting a typical transcription result"""
        text = "This is a test transcription"
        confidence = 0.94
        duration = 5.2

        result = f"Text: {truncate_text(text, 30)}\n"
        result += f"Confidence: {format_confidence(confidence)}\n"
        result += f"Duration: {format_duration(duration)}"

        assert "Text:" in result
        assert "94%" in result
        assert "5.2s" in result

    def test_format_audio_info(self):
        """Test formatting audio information"""
        sample_rate = 16000
        duration = 125.5
        file_size = 2000000

        result = f"Sample Rate: {format_sample_rate(sample_rate)}\n"
        result += f"Duration: {format_duration(duration)}\n"
        result += f"Size: {format_file_size(file_size)}"

        assert "16 kHz" in result
        assert "2m" in result
        assert "MB" in result
