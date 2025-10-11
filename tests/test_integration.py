"""
Integration tests for complete workflows.
"""
import pytest
import numpy as np
from pathlib import Path


class TestImports:
    """Test that all exports work correctly"""

    def test_import_main_module(self):
        """Test importing the main module"""
        import parakeet_stream
        assert hasattr(parakeet_stream, '__version__')
        assert parakeet_stream.__version__ == "0.3.2"

    def test_import_core_classes(self):
        """Test importing core classes"""
        from parakeet_stream import (
            Parakeet,
            AudioConfig,
            ConfigPresets,
            TranscriptResult,
            TranscriptBuffer,
            Segment,
            Microphone,
            AudioClip,
            LiveTranscriber,
        )

        # Verify all classes are importable
        assert Parakeet is not None
        assert AudioConfig is not None
        assert ConfigPresets is not None
        assert TranscriptResult is not None
        assert TranscriptBuffer is not None
        assert Segment is not None
        assert Microphone is not None
        assert AudioClip is not None
        assert LiveTranscriber is not None

    def test_legacy_imports(self):
        """Test that legacy imports still work"""
        from parakeet_stream import (
            StreamingTranscriber,
            TranscriberConfig,
            TranscriptionResult,
        )

        assert StreamingTranscriber is not None
        assert TranscriberConfig is not None
        assert TranscriptionResult is not None

    def test_all_exports_defined(self):
        """Test that __all__ contains expected exports"""
        import parakeet_stream

        expected_exports = {
            'Parakeet',
            'AudioConfig',
            'ConfigPresets',
            'TranscriptResult',
            'TranscriptBuffer',
            'Segment',
            'Microphone',
            'AudioClip',
            'LiveTranscriber',
            'StreamChunk',
            # Legacy
            'TranscriptionResult',
            'StreamingTranscriber',
            'TranscriberConfig',
        }

        actual_exports = set(parakeet_stream.__all__)
        assert expected_exports == actual_exports


class TestAudioClipWorkflow:
    """Test AudioClip creation and usage"""

    def test_create_and_save_clip(self, tmp_path):
        """Test creating AudioClip and saving"""
        from parakeet_stream import AudioClip

        # Create clip
        data = np.random.randn(16000).astype(np.float32) * 0.5
        clip = AudioClip(data, sample_rate=16000)

        assert clip.duration == 1.0
        assert clip.num_samples == 16000

        # Save
        output_path = tmp_path / "test.wav"
        clip.save(output_path)
        assert output_path.exists()

    def test_clip_to_tensor(self):
        """Test converting AudioClip to tensor"""
        from parakeet_stream import AudioClip
        import torch

        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        clip = AudioClip(data, sample_rate=16000)

        tensor = clip.to_tensor()
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3,)


class TestTranscriptBufferWorkflow:
    """Test TranscriptBuffer accumulation"""

    def test_buffer_accumulation(self):
        """Test accumulating segments in buffer"""
        from parakeet_stream import TranscriptBuffer, Segment

        buffer = TranscriptBuffer()

        # Add segments
        buffer.append(Segment("Hello", 0.0, 1.0, confidence=0.95))
        buffer.append(Segment("world", 1.0, 2.0, confidence=0.93))
        buffer.append(Segment("test", 2.0, 3.0, confidence=0.96))

        # Check accumulation
        assert len(buffer) == 3
        assert buffer.text == "Hello world test"

        # Check stats
        stats = buffer.stats
        assert stats['segments'] == 3
        assert stats['duration'] == 3.0
        assert stats['words'] == 3
        assert 0.93 < stats['avg_confidence'] < 0.96

    def test_buffer_save_and_load(self, tmp_path):
        """Test saving and loading buffer"""
        import json
        from parakeet_stream import TranscriptBuffer, Segment

        buffer = TranscriptBuffer()
        buffer.append(Segment("First", 0.0, 1.0))
        buffer.append(Segment("Second", 1.0, 2.0))

        # Save
        output_path = tmp_path / "transcript.json"
        buffer.save(output_path)

        # Load and verify
        with open(output_path) as f:
            data = json.load(f)

        assert data['text'] == "First Second"
        assert len(data['segments']) == 2


class TestConfigurationWorkflow:
    """Test configuration changes"""

    def test_config_presets(self):
        """Test using different config presets"""
        from parakeet_stream import Parakeet, ConfigPresets

        pk = Parakeet(lazy=True)

        # Test preset access
        assert pk.configs == ConfigPresets
        assert pk.config.name == "balanced"

        # Test changing presets
        pk.with_config('high_quality')
        assert pk.config.name == "high_quality"

        pk.with_quality('max')
        assert pk.config.quality_score == 5

        pk.with_latency('low')
        assert pk.config.latency < 3.0

    def test_config_chaining(self):
        """Test fluent API chaining"""
        from parakeet_stream import Parakeet

        pk = Parakeet(lazy=True)

        # Chain multiple config changes
        result = pk.with_quality('high').with_params(chunk_secs=3.0)

        assert result is pk  # Returns self
        assert pk.config.chunk_secs == 3.0

    def test_config_changes_no_reload(self):
        """Test that config changes don't reload model"""
        from parakeet_stream import Parakeet
        import time

        pk = Parakeet(lazy=True)

        # Config changes should be instant (< 0.01s)
        start = time.time()
        pk.with_quality('high')
        elapsed = time.time() - start

        assert elapsed < 0.01  # Should be near-instant


class TestDisplayMethods:
    """Test that display methods work"""

    def test_parakeet_display(self):
        """Test Parakeet display methods"""
        from parakeet_stream import Parakeet

        pk = Parakeet(lazy=True)

        # Test __repr__
        repr_str = repr(pk)
        assert "Parakeet" in repr_str
        assert "not loaded" in repr_str

        # Test _repr_html_
        html = pk._repr_html_()
        assert "<div" in html
        assert "Parakeet" in html

    def test_transcript_result_display(self):
        """Test TranscriptResult display"""
        from parakeet_stream import TranscriptResult

        result = TranscriptResult(
            text="Test transcription",
            confidence=0.94,
            duration=2.5
        )

        repr_str = repr(result)
        assert "TranscriptResult" in repr_str
        assert "Test transcription" in repr_str

        html = result._repr_html_()
        assert "📝" in html
        assert "Test transcription" in html

    def test_audio_clip_display(self):
        """Test AudioClip display"""
        from parakeet_stream import AudioClip
        import numpy as np

        data = np.zeros(16000, dtype=np.float32)
        clip = AudioClip(data, sample_rate=16000)

        repr_str = repr(clip)
        assert "AudioClip" in repr_str
        assert "1.0s" in repr_str

        html = clip._repr_html_()
        assert "🔊" in html
        assert "AudioClip" in html

    def test_transcript_buffer_display(self):
        """Test TranscriptBuffer display"""
        from parakeet_stream import TranscriptBuffer, Segment

        buffer = TranscriptBuffer()
        buffer.append(Segment("Test", 0.0, 1.0))

        repr_str = repr(buffer)
        assert "TranscriptBuffer" in repr_str

        html = buffer._repr_html_()
        assert "📄" in html
        assert "TranscriptBuffer" in html


class TestParakeetMethods:
    """Test Parakeet has all expected methods"""

    def test_parakeet_has_transcribe(self):
        """Test Parakeet has transcribe method"""
        from parakeet_stream import Parakeet

        pk = Parakeet(lazy=True)
        assert hasattr(pk, 'transcribe')
        assert callable(pk.transcribe)

    def test_parakeet_has_listen(self):
        """Test Parakeet has listen method"""
        from parakeet_stream import Parakeet

        pk = Parakeet(lazy=True)
        assert hasattr(pk, 'listen')
        assert callable(pk.listen)

    def test_parakeet_has_config_methods(self):
        """Test Parakeet has config methods"""
        from parakeet_stream import Parakeet

        pk = Parakeet(lazy=True)
        assert hasattr(pk, 'with_config')
        assert hasattr(pk, 'with_quality')
        assert hasattr(pk, 'with_latency')
        assert hasattr(pk, 'with_params')

    def test_parakeet_has_properties(self):
        """Test Parakeet has expected properties"""
        from parakeet_stream import Parakeet

        pk = Parakeet(lazy=True)
        assert hasattr(pk, 'config')
        assert hasattr(pk, 'configs')
        assert hasattr(pk, 'model_name')
        assert hasattr(pk, 'device')


class TestEndToEndWorkflow:
    """Test complete end-to-end workflows"""

    def test_simple_transcribe_workflow(self):
        """Test simple transcription workflow (without loading model)"""
        from parakeet_stream import Parakeet
        import numpy as np

        pk = Parakeet(lazy=True)

        # Create fake audio
        audio = np.random.randn(16000).astype(np.float32)

        # Test that API is correct (will fail at model load, which is expected)
        # We just verify the method exists and signature is correct
        assert callable(pk.transcribe)

    def test_config_experiment_workflow(self):
        """Test quality experimentation workflow"""
        from parakeet_stream import Parakeet

        pk = Parakeet(lazy=True)

        # Try different configs
        configs_to_test = ['max', 'high', 'good', 'low', 'realtime']

        for config in configs_to_test:
            pk.with_quality(config)
            # Verify config changed
            assert pk.config is not None

    def test_buffer_workflow(self):
        """Test complete buffer workflow"""
        from parakeet_stream import TranscriptBuffer, Segment

        # Create buffer
        buffer = TranscriptBuffer()

        # Simulate live transcription
        segments = [
            Segment("Hello", 0.0, 1.0, 0.95),
            Segment("world", 1.0, 2.0, 0.93),
            Segment("this", 2.0, 3.0, 0.96),
            Segment("is", 3.0, 4.0, 0.92),
            Segment("a", 4.0, 5.0, 0.94),
            Segment("test", 5.0, 6.0, 0.95),
        ]

        for seg in segments:
            buffer.append(seg)

        # Check results
        assert len(buffer) == 6
        assert buffer.text == "Hello world this is a test"

        # Check head/tail
        assert len(buffer.head(3)) == 3
        assert len(buffer.tail(3)) == 3

        # Check stats
        stats = buffer.stats
        assert stats['segments'] == 6
        assert stats['duration'] == 6.0


class TestBackwardCompatibility:
    """Test that legacy API still works"""

    def test_legacy_streaming_transcriber_import(self):
        """Test legacy StreamingTranscriber import"""
        from parakeet_stream import StreamingTranscriber

        assert StreamingTranscriber is not None

    def test_legacy_config_import(self):
        """Test legacy TranscriberConfig import"""
        from parakeet_stream import TranscriberConfig

        assert TranscriberConfig is not None

    def test_legacy_transcription_result(self):
        """Test legacy TranscriptionResult import"""
        from parakeet_stream import TranscriptionResult

        # Can create instance
        result = TranscriptionResult(text="Test")
        assert result.text == "Test"
