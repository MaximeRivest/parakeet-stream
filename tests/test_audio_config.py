"""
Tests for audio configuration system.
"""
import pytest
from parakeet_stream.audio_config import AudioConfig, ConfigPresets


class TestAudioConfig:
    """Tests for AudioConfig class"""

    def test_audio_config_creation(self):
        """Test creating an AudioConfig"""
        cfg = AudioConfig(
            name="test",
            chunk_secs=2.0,
            left_context_secs=10.0,
            right_context_secs=2.0
        )
        assert cfg.name == "test"
        assert cfg.chunk_secs == 2.0
        assert cfg.left_context_secs == 10.0
        assert cfg.right_context_secs == 2.0

    def test_latency_calculation(self):
        """Test latency property calculation"""
        cfg = AudioConfig(
            name="test",
            chunk_secs=2.0,
            left_context_secs=10.0,
            right_context_secs=2.0
        )
        assert cfg.latency == 4.0

    def test_quality_score_maximum(self):
        """Test quality score for maximum quality config"""
        cfg = AudioConfig(
            name="max",
            chunk_secs=10.0,
            left_context_secs=10.0,
            right_context_secs=5.0
        )
        # Total context: 25s -> score 5
        assert cfg.quality_score == 5

    def test_quality_score_balanced(self):
        """Test quality score for balanced config"""
        cfg = AudioConfig(
            name="balanced",
            chunk_secs=2.0,
            left_context_secs=10.0,
            right_context_secs=2.0
        )
        # Total context: 14s -> score 4
        assert cfg.quality_score == 4

    def test_quality_score_realtime(self):
        """Test quality score for realtime config"""
        cfg = AudioConfig(
            name="realtime",
            chunk_secs=0.5,
            left_context_secs=10.0,
            right_context_secs=0.5
        )
        # Total context: 11s -> score 3
        assert cfg.quality_score == 3

    def test_quality_indicator(self):
        """Test quality visual indicator"""
        cfg = AudioConfig(
            name="test",
            chunk_secs=10.0,
            left_context_secs=10.0,
            right_context_secs=5.0
        )
        assert cfg.quality_indicator == "●●●●●"

        cfg2 = AudioConfig(
            name="test2",
            chunk_secs=2.0,
            left_context_secs=10.0,
            right_context_secs=2.0
        )
        assert cfg2.quality_indicator == "●●●●○"

    def test_repr(self):
        """Test string representation"""
        cfg = AudioConfig(
            name="test",
            chunk_secs=2.0,
            left_context_secs=10.0,
            right_context_secs=2.0
        )
        repr_str = repr(cfg)
        assert "test" in repr_str
        assert "4.0s" in repr_str
        assert "●" in repr_str

    def test_str(self):
        """Test string conversion"""
        cfg = AudioConfig(
            name="test",
            chunk_secs=2.0,
            left_context_secs=10.0,
            right_context_secs=2.0
        )
        str_repr = str(cfg)
        assert "test" in str_repr
        assert "Latency" in str_repr
        assert "Quality" in str_repr


class TestConfigPresets:
    """Tests for ConfigPresets class"""

    def test_presets_exist(self):
        """Test that all presets are defined"""
        assert ConfigPresets.MAXIMUM_QUALITY is not None
        assert ConfigPresets.HIGH_QUALITY is not None
        assert ConfigPresets.BALANCED is not None
        assert ConfigPresets.LOW_LATENCY is not None
        assert ConfigPresets.REALTIME is not None
        assert ConfigPresets.ULTRA_REALTIME is not None

    def test_preset_names(self):
        """Test preset names are correct"""
        assert ConfigPresets.MAXIMUM_QUALITY.name == "maximum_quality"
        assert ConfigPresets.BALANCED.name == "balanced"
        assert ConfigPresets.REALTIME.name == "realtime"

    def test_preset_latencies(self):
        """Test presets have expected latency ordering"""
        assert ConfigPresets.MAXIMUM_QUALITY.latency > ConfigPresets.BALANCED.latency
        assert ConfigPresets.BALANCED.latency > ConfigPresets.LOW_LATENCY.latency
        assert ConfigPresets.LOW_LATENCY.latency > ConfigPresets.REALTIME.latency

    def test_get_by_name(self):
        """Test getting preset by name"""
        cfg = ConfigPresets.get('balanced')
        assert cfg.name == 'balanced'
        assert cfg.latency > 0

    def test_get_by_alias(self):
        """Test getting preset by alias"""
        cfg = ConfigPresets.get('max')
        assert cfg.name == 'maximum_quality'

        cfg2 = ConfigPresets.get('high')
        assert cfg2.name == 'high_quality'

        cfg3 = ConfigPresets.get('good')
        assert cfg3.name == 'balanced'

    def test_get_case_insensitive(self):
        """Test case-insensitive preset names"""
        cfg1 = ConfigPresets.get('BALANCED')
        cfg2 = ConfigPresets.get('balanced')
        assert cfg1.name == cfg2.name

    def test_get_with_dashes(self):
        """Test preset names with dashes/spaces"""
        cfg = ConfigPresets.get('low-latency')
        assert cfg.name == 'low_latency'

    def test_get_invalid_name(self):
        """Test getting invalid preset raises error"""
        with pytest.raises(ValueError, match="Unknown preset"):
            ConfigPresets.get('invalid_preset')

    def test_list_presets(self):
        """Test listing all presets"""
        presets = ConfigPresets.list()
        assert isinstance(presets, list)
        assert len(presets) >= 6
        assert 'balanced' in presets
        assert 'realtime' in presets
        assert 'maximum_quality' in presets

    def test_list_with_details(self):
        """Test detailed preset listing"""
        details = ConfigPresets.list_with_details()
        assert isinstance(details, str)
        assert 'balanced' in details
        assert 'Latency' in details
        assert 'Quality' in details

    def test_by_quality(self):
        """Test getting preset by quality level"""
        cfg = ConfigPresets.by_quality('max')
        assert cfg.quality_score == 5

        cfg2 = ConfigPresets.by_quality('high')
        assert cfg2.quality_score >= 4

        cfg3 = ConfigPresets.by_quality('good')
        assert cfg3.quality_score >= 3

    def test_by_quality_invalid(self):
        """Test invalid quality level raises error"""
        with pytest.raises(ValueError, match="Unknown quality level"):
            ConfigPresets.by_quality('invalid')

    def test_by_latency(self):
        """Test getting preset by latency level"""
        cfg = ConfigPresets.by_latency('high')
        assert cfg.latency > 10

        cfg2 = ConfigPresets.by_latency('medium')
        assert 3 < cfg2.latency < 6

        cfg3 = ConfigPresets.by_latency('low')
        assert cfg3.latency < 3

        cfg4 = ConfigPresets.by_latency('realtime')
        assert cfg4.latency < 2

    def test_by_latency_invalid(self):
        """Test invalid latency level raises error"""
        with pytest.raises(ValueError, match="Unknown latency level"):
            ConfigPresets.by_latency('invalid')

    def test_balanced_is_default(self):
        """Test balanced is a reasonable default"""
        cfg = ConfigPresets.BALANCED
        assert 3 < cfg.latency < 5
        assert cfg.quality_score >= 3


class TestPresetsRelationships:
    """Tests for relationships between presets"""

    def test_quality_vs_latency_tradeoff(self):
        """Test that higher quality generally means higher latency"""
        # Maximum quality should have highest latency
        assert ConfigPresets.MAXIMUM_QUALITY.latency > ConfigPresets.LOW_LATENCY.latency

        # Realtime should have lowest latency
        assert ConfigPresets.REALTIME.latency < ConfigPresets.BALANCED.latency

    def test_all_presets_valid(self):
        """Test that all presets have valid parameters"""
        for preset_name in ConfigPresets.list():
            cfg = ConfigPresets.get(preset_name)
            assert cfg.chunk_secs > 0
            assert cfg.left_context_secs >= 0
            assert cfg.right_context_secs >= 0
            assert cfg.latency > 0
            assert 1 <= cfg.quality_score <= 5

    def test_presets_cover_range(self):
        """Test that presets cover a good range of latencies"""
        latencies = [
            ConfigPresets.MAXIMUM_QUALITY.latency,
            ConfigPresets.HIGH_QUALITY.latency,
            ConfigPresets.BALANCED.latency,
            ConfigPresets.LOW_LATENCY.latency,
            ConfigPresets.REALTIME.latency,
        ]

        # Should have good spread
        assert max(latencies) > 10  # High end
        assert min(latencies) < 2   # Low end
        assert len(set(latencies)) == 5  # All different
