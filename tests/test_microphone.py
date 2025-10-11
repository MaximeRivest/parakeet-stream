"""
Tests for Microphone class.
"""
import pytest
import numpy as np
from parakeet_stream.audio_clip import AudioClip


class TestMicrophoneImport:
    """Tests for Microphone import and basic structure"""

    def test_microphone_imports(self):
        """Test that Microphone can be imported"""
        from parakeet_stream.microphone import Microphone
        assert Microphone is not None

    def test_microphone_has_expected_methods(self):
        """Test that Microphone has expected methods"""
        from parakeet_stream.microphone import Microphone

        # Check class methods exist
        assert hasattr(Microphone, 'discover')
        assert hasattr(Microphone, '__init__')
        assert hasattr(Microphone, 'record')
        assert hasattr(Microphone, 'test')

    def test_microphone_has_rich_repr(self):
        """Test that Microphone has rich display methods"""
        from parakeet_stream.microphone import Microphone

        assert hasattr(Microphone, '__repr__')
        assert hasattr(Microphone, '_repr_pretty_')
        assert hasattr(Microphone, '_repr_html_')


class TestMicrophoneProperties:
    """Tests for Microphone properties"""

    def test_microphone_has_properties(self):
        """Test that Microphone has expected properties"""
        from parakeet_stream.microphone import Microphone

        # Check properties are defined
        assert hasattr(Microphone, 'name')
        assert hasattr(Microphone, 'channels')


# Note: Hardware-dependent tests are skipped by default
# To test with real hardware, run with: pytest -m hardware

@pytest.mark.hardware
class TestMicrophoneHardware:
    """Tests that require actual hardware (marked with @pytest.mark.hardware)"""

    def test_discover_with_hardware(self):
        """Test device discovery (requires sound hardware)"""
        pytest.importorskip('sounddevice')
        from parakeet_stream.microphone import Microphone

        mics = Microphone.discover()
        # Should find at least 0 devices (may be 0 in CI)
        assert isinstance(mics, list)

    def test_create_with_default(self):
        """Test creating microphone with default device (requires sound hardware)"""
        pytest.importorskip('sounddevice')
        from parakeet_stream.microphone import Microphone

        try:
            mic = Microphone()
            assert mic.sample_rate == 16000
            assert hasattr(mic, 'device')
            assert hasattr(mic, 'name')
        except RuntimeError:
            # No input devices available - OK in CI
            pytest.skip("No input devices available")

    def test_record_short(self):
        """Test short recording (requires sound hardware)"""
        pytest.importorskip('sounddevice')
        from parakeet_stream.microphone import Microphone

        try:
            mic = Microphone()
            clip = mic.record(duration=0.1)

            assert isinstance(clip, AudioClip)
            assert clip.duration >= 0.1
        except RuntimeError:
            pytest.skip("No input devices available")
