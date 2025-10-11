"""
Tests for AudioClip class.
"""
import pytest
import numpy as np
from pathlib import Path
from parakeet_stream.audio_clip import AudioClip


class TestAudioClipCreation:
    """Tests for AudioClip creation and basic properties"""

    def test_create_basic(self):
        """Test creating basic AudioClip"""
        data = np.random.randn(16000).astype(np.float32)
        clip = AudioClip(data, sample_rate=16000)

        assert clip.sample_rate == 16000
        assert len(clip.data) == 16000

    def test_create_different_sample_rates(self):
        """Test creating clips with different sample rates"""
        data = np.random.randn(44100).astype(np.float32)
        clip = AudioClip(data, sample_rate=44100)

        assert clip.sample_rate == 44100

    def test_data_is_stored(self):
        """Test that data is stored correctly"""
        data = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        clip = AudioClip(data, sample_rate=16000)

        np.testing.assert_array_equal(clip.data, data)


class TestAudioClipProperties:
    """Tests for computed properties"""

    def test_duration_calculation(self):
        """Test duration property calculation"""
        # 1 second at 16kHz
        data = np.zeros(16000, dtype=np.float32)
        clip = AudioClip(data, sample_rate=16000)

        assert clip.duration == 1.0

    def test_duration_fractional(self):
        """Test duration with fractional seconds"""
        # 2.5 seconds at 16kHz
        data = np.zeros(40000, dtype=np.float32)
        clip = AudioClip(data, sample_rate=16000)

        assert clip.duration == 2.5

    def test_duration_different_rate(self):
        """Test duration with different sample rate"""
        # 1 second at 44.1kHz
        data = np.zeros(44100, dtype=np.float32)
        clip = AudioClip(data, sample_rate=44100)

        assert abs(clip.duration - 1.0) < 0.0001

    def test_num_samples(self):
        """Test num_samples property"""
        data = np.zeros(1234, dtype=np.float32)
        clip = AudioClip(data, sample_rate=16000)

        assert clip.num_samples == 1234


class TestAudioClipSave:
    """Tests for save functionality"""

    def test_save_creates_file(self, tmp_path):
        """Test that save creates a file"""
        data = np.random.randn(1600).astype(np.float32)
        clip = AudioClip(data, sample_rate=16000)

        output_path = tmp_path / "test.wav"
        clip.save(output_path)

        assert output_path.exists()

    def test_save_with_string_path(self, tmp_path):
        """Test save with string path"""
        data = np.random.randn(1600).astype(np.float32)
        clip = AudioClip(data, sample_rate=16000)

        output_path = str(tmp_path / "test.wav")
        clip.save(output_path)

        assert Path(output_path).exists()

    def test_save_preserves_data(self, tmp_path):
        """Test that saved data can be loaded back"""
        import soundfile as sf

        # Use data in valid range [-1, 1] to avoid clipping
        data = (np.random.randn(1600) * 0.5).astype(np.float32)
        clip = AudioClip(data, sample_rate=16000)

        output_path = tmp_path / "test.wav"
        clip.save(output_path)

        # Load back and verify
        loaded_data, loaded_sr = sf.read(output_path)
        assert loaded_sr == 16000
        # WAV format has some precision loss, so check shape and range
        assert loaded_data.shape == data.shape
        # Check most values are close (allow for some WAV encoding differences)
        close_matches = np.sum(np.abs(loaded_data - data) < 0.01)
        assert close_matches > len(data) * 0.9  # 90% should be very close


class TestAudioClipRepr:
    """Tests for __repr__ method"""

    def test_repr_basic(self):
        """Test basic repr"""
        data = np.zeros(16000, dtype=np.float32)
        clip = AudioClip(data, sample_rate=16000)

        repr_str = repr(clip)
        assert "AudioClip" in repr_str
        assert "1.0s" in repr_str
        assert "16000Hz" in repr_str

    def test_repr_different_duration(self):
        """Test repr with different duration"""
        data = np.zeros(48000, dtype=np.float32)
        clip = AudioClip(data, sample_rate=16000)

        repr_str = repr(clip)
        assert "3.0s" in repr_str


class TestAudioClipReprPretty:
    """Tests for _repr_pretty_ method (IPython)"""

    def test_repr_pretty_basic(self):
        """Test _repr_pretty_ basic output"""
        data = np.zeros(32000, dtype=np.float32)
        clip = AudioClip(data, sample_rate=16000)

        class MockPrinter:
            def __init__(self):
                self.text_content = ""

            def text(self, content):
                self.text_content = content

        printer = MockPrinter()
        clip._repr_pretty_(printer, cycle=False)

        assert "🔊" in printer.text_content
        assert "AudioClip" in printer.text_content
        assert "Duration:" in printer.text_content
        assert "Sample Rate:" in printer.text_content

    def test_repr_pretty_shows_duration(self):
        """Test _repr_pretty_ shows correct duration"""
        data = np.zeros(16000, dtype=np.float32)
        clip = AudioClip(data, sample_rate=16000)

        class MockPrinter:
            def __init__(self):
                self.text_content = ""

            def text(self, content):
                self.text_content = content

        printer = MockPrinter()
        clip._repr_pretty_(printer, cycle=False)

        assert "1.0s" in printer.text_content

    def test_repr_pretty_cycle(self):
        """Test _repr_pretty_ handles circular reference"""
        data = np.zeros(16000, dtype=np.float32)
        clip = AudioClip(data, sample_rate=16000)

        class MockPrinter:
            def __init__(self):
                self.text_content = ""

            def text(self, content):
                self.text_content = content

        printer = MockPrinter()
        clip._repr_pretty_(printer, cycle=True)

        assert printer.text_content == "AudioClip(...)"


class TestAudioClipReprHtml:
    """Tests for _repr_html_ method (Jupyter)"""

    def test_repr_html_basic(self):
        """Test _repr_html_ returns HTML"""
        data = np.zeros(16000, dtype=np.float32)
        clip = AudioClip(data, sample_rate=16000)

        html = clip._repr_html_()

        assert isinstance(html, str)
        assert "<div" in html
        assert "<table" in html
        assert "AudioClip" in html

    def test_repr_html_has_emoji(self):
        """Test _repr_html_ includes emoji"""
        data = np.zeros(16000, dtype=np.float32)
        clip = AudioClip(data, sample_rate=16000)

        html = clip._repr_html_()

        assert "🔊" in html

    def test_repr_html_shows_info(self):
        """Test _repr_html_ shows all info"""
        data = np.zeros(32000, dtype=np.float32)
        clip = AudioClip(data, sample_rate=16000)

        html = clip._repr_html_()

        assert "Duration:" in html
        assert "Sample Rate:" in html
        assert "Samples:" in html


class TestAudioClipToTensor:
    """Tests for to_tensor method"""

    def test_to_tensor_basic(self):
        """Test conversion to tensor"""
        import torch

        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        clip = AudioClip(data, sample_rate=16000)

        tensor = clip.to_tensor()

        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3,)
        torch.testing.assert_close(tensor, torch.tensor([1.0, 2.0, 3.0]))

    def test_to_tensor_preserves_values(self):
        """Test that tensor conversion preserves values"""
        import torch

        data = np.random.randn(100).astype(np.float32)
        clip = AudioClip(data, sample_rate=16000)

        tensor = clip.to_tensor()

        np.testing.assert_array_almost_equal(
            tensor.numpy(),
            data,
            decimal=6
        )


class TestAudioClipIntegration:
    """Integration tests for AudioClip"""

    def test_full_workflow(self, tmp_path):
        """Test complete workflow: create, save, display"""
        # Create
        data = np.random.randn(16000).astype(np.float32)
        clip = AudioClip(data, sample_rate=16000)

        # Check properties
        assert clip.duration == 1.0
        assert clip.num_samples == 16000

        # Save
        output_path = tmp_path / "test.wav"
        clip.save(output_path)
        assert output_path.exists()

        # Check repr
        repr_str = repr(clip)
        assert "AudioClip" in repr_str

    def test_different_durations(self):
        """Test clips with various durations"""
        durations = [0.5, 1.0, 2.5, 5.0]

        for duration in durations:
            samples = int(16000 * duration)
            data = np.zeros(samples, dtype=np.float32)
            clip = AudioClip(data, sample_rate=16000)

            assert abs(clip.duration - duration) < 0.0001
