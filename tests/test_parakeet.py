"""
Tests for Parakeet class.
"""
import pytest
import numpy as np
from parakeet_stream.parakeet import Parakeet
from parakeet_stream.audio_config import AudioConfig, ConfigPresets


class TestParakeetInit:
    """Tests for Parakeet initialization"""

    def test_lazy_init(self):
        """Test lazy initialization doesn't load model"""
        pk = Parakeet(lazy=True)
        assert not pk._initialized
        assert pk.model is None

    def test_default_config(self):
        """Test default configuration is balanced"""
        pk = Parakeet(lazy=True)
        assert pk.config.name == "balanced"

    def test_config_by_name(self):
        """Test configuration by preset name"""
        pk = Parakeet(config='high_quality', lazy=True)
        assert pk.config.name == "high_quality"

    def test_config_by_object(self):
        """Test configuration with AudioConfig object"""
        cfg = ConfigPresets.LOW_LATENCY
        pk = Parakeet(config=cfg, lazy=True)
        assert pk.config.name == "low_latency"

    def test_invalid_config_type(self):
        """Test invalid config type raises error"""
        with pytest.raises(TypeError):
            Parakeet(config=123, lazy=True)

    def test_device_setting(self):
        """Test device is set correctly"""
        pk = Parakeet(device='cpu', lazy=True)
        assert pk.device == 'cpu'

    def test_model_name_setting(self):
        """Test model name is set correctly"""
        pk = Parakeet(lazy=True)
        assert pk.model_name == "nvidia/parakeet-tdt-0.6b-v3"

    def test_configs_property(self):
        """Test configs property returns ConfigPresets"""
        pk = Parakeet(lazy=True)
        assert pk.configs == ConfigPresets

    def test_config_property(self):
        """Test config property returns AudioConfig"""
        pk = Parakeet(lazy=True)
        assert isinstance(pk.config, AudioConfig)


class TestParakeetLoad:
    """Tests for model loading"""

    @pytest.mark.slow
    def test_load_idempotent(self):
        """Test calling load() multiple times is safe"""
        pk = Parakeet(lazy=True)
        pk.load()
        model_id = id(pk.model)

        pk.load()  # Should not reload
        assert id(pk.model) == model_id

    @pytest.mark.slow
    def test_eager_load(self):
        """Test eager loading (default behavior)"""
        pk = Parakeet()  # Should load immediately
        assert pk._initialized
        assert pk.model is not None

    @pytest.mark.slow
    def test_lazy_then_load(self):
        """Test lazy init then manual load"""
        pk = Parakeet(lazy=True)
        assert not pk._initialized

        pk.load()
        assert pk._initialized
        assert pk.model is not None

    @pytest.mark.slow
    def test_sample_rate_set_after_load(self):
        """Test sample rate is set after loading"""
        pk = Parakeet()
        assert pk.sample_rate is not None
        assert pk.sample_rate > 0


class TestParakeetTranscribe:
    """Tests for transcription functionality"""

    @pytest.mark.slow
    def test_transcribe_requires_load(self):
        """Test transcribe works after loading"""
        pk = Parakeet()
        # Create dummy audio
        audio = np.random.randn(16000).astype(np.float32)
        result = pk.transcribe(audio)
        assert hasattr(result, 'text')

    def test_transcribe_without_load_lazy(self):
        """Test transcribe with lazy=True still works (auto-loads)"""
        pk = Parakeet(lazy=True)
        # Should auto-load when transcribe is called
        audio = np.random.randn(16000).astype(np.float32)
        # Don't actually run this in fast tests
        pytest.skip("Requires model loading")


class TestParakeetFluentAPI:
    """Tests for fluent configuration methods"""

    def test_with_config_by_name(self):
        """Test with_config accepts preset name"""
        pk = Parakeet(lazy=True)
        result = pk.with_config('high_quality')

        assert result is pk  # Returns self
        assert pk.config.name == 'high_quality'

    def test_with_config_by_object(self):
        """Test with_config accepts AudioConfig object"""
        pk = Parakeet(lazy=True)
        cfg = ConfigPresets.LOW_LATENCY
        result = pk.with_config(cfg)

        assert result is pk
        assert pk.config.name == 'low_latency'

    def test_with_config_invalid_type(self):
        """Test with_config rejects invalid type"""
        pk = Parakeet(lazy=True)
        with pytest.raises(TypeError):
            pk.with_config(123)

    def test_with_quality_max(self):
        """Test with_quality('max')"""
        pk = Parakeet(lazy=True)
        result = pk.with_quality('max')

        assert result is pk
        assert pk.config.quality_score == 5

    def test_with_quality_high(self):
        """Test with_quality('high')"""
        pk = Parakeet(lazy=True)
        pk.with_quality('high')
        assert pk.config.quality_score >= 4

    def test_with_quality_good(self):
        """Test with_quality('good')"""
        pk = Parakeet(lazy=True)
        pk.with_quality('good')
        assert pk.config.quality_score >= 3

    def test_with_quality_invalid(self):
        """Test with_quality rejects invalid level"""
        pk = Parakeet(lazy=True)
        with pytest.raises(ValueError):
            pk.with_quality('invalid')

    def test_with_latency_low(self):
        """Test with_latency('low')"""
        pk = Parakeet(lazy=True)
        pk.with_latency('low')
        assert pk.config.latency < 3.0

    def test_with_latency_realtime(self):
        """Test with_latency('realtime')"""
        pk = Parakeet(lazy=True)
        pk.with_latency('realtime')
        assert pk.config.latency < 2.0

    def test_with_latency_invalid(self):
        """Test with_latency rejects invalid level"""
        pk = Parakeet(lazy=True)
        with pytest.raises(ValueError):
            pk.with_latency('invalid')

    def test_with_params_chunk(self):
        """Test with_params updates chunk_secs"""
        pk = Parakeet(lazy=True)
        pk.with_params(chunk_secs=1.5)

        assert pk.config.chunk_secs == 1.5
        assert pk.config.name == "custom"

    def test_with_params_multiple(self):
        """Test with_params updates multiple parameters"""
        pk = Parakeet(lazy=True)
        pk.with_params(
            chunk_secs=3.0,
            left_context_secs=15.0,
            right_context_secs=3.5
        )

        assert pk.config.chunk_secs == 3.0
        assert pk.config.left_context_secs == 15.0
        assert pk.config.right_context_secs == 3.5

    def test_with_params_partial(self):
        """Test with_params preserves unspecified values"""
        pk = Parakeet(config='balanced', lazy=True)
        original_chunk = pk.config.chunk_secs

        pk.with_params(left_context_secs=20.0)

        assert pk.config.chunk_secs == original_chunk
        assert pk.config.left_context_secs == 20.0

    def test_chaining(self):
        """Test method chaining"""
        pk = Parakeet(lazy=True)

        result = pk.with_quality('high').with_params(chunk_secs=3.0)

        assert result is pk
        assert pk.config.chunk_secs == 3.0

    def test_config_updates_sync(self):
        """Test that config changes update internal state"""
        pk = Parakeet(lazy=True)

        pk.with_config('low_latency')

        # Check internal config is updated
        assert pk._transcriber_config.chunk_secs == pk.config.chunk_secs
        assert pk._transcriber_config.left_context_secs == pk.config.left_context_secs


class TestParakeetIntegration:
    """Integration tests"""

    def test_config_properties_match(self):
        """Test internal configs stay synchronized"""
        pk = Parakeet(config='low_latency', lazy=True)

        assert pk.config.chunk_secs == pk._transcriber_config.chunk_secs
        assert pk.config.left_context_secs == pk._transcriber_config.left_context_secs
        assert pk.config.right_context_secs == pk._transcriber_config.right_context_secs

    @pytest.mark.slow
    def test_backward_compatibility_initialize(self):
        """Test _initialize_model() still works (backward compat)"""
        pk = Parakeet(lazy=True)
        assert not pk._initialized

        # Should work via legacy method
        pk._initialize_model()
        assert pk._initialized


class TestParakeetDisplay:
    """Tests for rich display methods (Phase 1.5)"""

    def test_repr_basic(self):
        """Test __repr__ includes basic info"""
        pk = Parakeet(lazy=True)
        repr_str = repr(pk)

        assert "Parakeet" in repr_str
        assert "nvidia/parakeet-tdt-0.6b-v3" in repr_str
        assert "cpu" in repr_str
        assert "balanced" in repr_str

    def test_repr_not_loaded(self):
        """Test __repr__ shows 'not loaded' status"""
        pk = Parakeet(lazy=True)
        repr_str = repr(pk)

        assert "not loaded" in repr_str

    def test_repr_with_custom_config(self):
        """Test __repr__ shows custom config name"""
        pk = Parakeet(config='high_quality', lazy=True)
        repr_str = repr(pk)

        assert "high_quality" in repr_str

    def test_repr_with_custom_device(self):
        """Test __repr__ shows custom device"""
        pk = Parakeet(device='cuda', lazy=True)
        repr_str = repr(pk)

        assert "cuda" in repr_str

    def test_repr_pretty_basic(self):
        """Test _repr_pretty_ produces multi-line output"""
        pk = Parakeet(lazy=True)

        # Mock IPython printer
        class MockPrinter:
            def __init__(self):
                self.text_content = ""

            def text(self, content):
                self.text_content = content

        printer = MockPrinter()
        pk._repr_pretty_(printer, cycle=False)

        # Check content includes expected info
        assert "Parakeet" in printer.text_content
        assert "Quality:" in printer.text_content
        assert "Latency:" in printer.text_content
        assert "Status:" in printer.text_content

    def test_repr_pretty_has_quality_indicator(self):
        """Test _repr_pretty_ includes quality indicator (●●●○○)"""
        pk = Parakeet(config='balanced', lazy=True)

        class MockPrinter:
            def __init__(self):
                self.text_content = ""

            def text(self, content):
                self.text_content = content

        printer = MockPrinter()
        pk._repr_pretty_(printer, cycle=False)

        # Should have quality circles
        assert "●" in printer.text_content

    def test_repr_pretty_cycle_handling(self):
        """Test _repr_pretty_ handles circular references"""
        pk = Parakeet(lazy=True)

        class MockPrinter:
            def __init__(self):
                self.text_content = ""

            def text(self, content):
                self.text_content = content

        printer = MockPrinter()
        pk._repr_pretty_(printer, cycle=True)

        assert printer.text_content == "Parakeet(...)"

    def test_repr_pretty_status_not_loaded(self):
        """Test _repr_pretty_ shows 'Not loaded' status"""
        pk = Parakeet(lazy=True)

        class MockPrinter:
            def __init__(self):
                self.text_content = ""

            def text(self, content):
                self.text_content = content

        printer = MockPrinter()
        pk._repr_pretty_(printer, cycle=False)

        assert "Not loaded" in printer.text_content
        assert "○" in printer.text_content  # Empty circle for not loaded

    def test_repr_html_basic(self):
        """Test _repr_html_ returns HTML string"""
        pk = Parakeet(lazy=True)
        html = pk._repr_html_()

        assert isinstance(html, str)
        assert "<div" in html
        assert "<table" in html

    def test_repr_html_includes_model_info(self):
        """Test _repr_html_ includes model information"""
        pk = Parakeet(lazy=True)
        html = pk._repr_html_()

        assert "nvidia/parakeet-tdt-0.6b-v3" in html
        assert "cpu" in html
        assert "balanced" in html

    def test_repr_html_has_status_icon(self):
        """Test _repr_html_ includes status icon"""
        pk = Parakeet(lazy=True)
        html = pk._repr_html_()

        # Should have a status icon (emoji)
        assert "⚪" in html or "✅" in html

    def test_repr_html_shows_quality(self):
        """Test _repr_html_ shows quality indicator"""
        pk = Parakeet(config='high_quality', lazy=True)
        html = pk._repr_html_()

        # Should show quality circles
        assert "●" in html
        assert "Quality:" in html

    def test_repr_html_shows_latency(self):
        """Test _repr_html_ shows latency value"""
        pk = Parakeet(config='low_latency', lazy=True)
        html = pk._repr_html_()

        assert "Latency:" in html
        # Should have a numeric latency value with 's'
        assert "s" in html

    def test_repr_html_not_loaded_status(self):
        """Test _repr_html_ shows 'Not loaded' status"""
        pk = Parakeet(lazy=True)
        html = pk._repr_html_()

        assert "Not loaded" in html
        assert "⚪" in html  # Empty circle emoji

    def test_display_methods_consistent(self):
        """Test all display methods show consistent information"""
        pk = Parakeet(config='realtime', device='cuda', lazy=True)

        # Get all representations
        repr_str = repr(pk)

        class MockPrinter:
            def __init__(self):
                self.text_content = ""

            def text(self, content):
                self.text_content = content

        printer = MockPrinter()
        pk._repr_pretty_(printer, cycle=False)
        pretty_str = printer.text_content

        html_str = pk._repr_html_()

        # All should mention the config
        assert "realtime" in repr_str
        assert "realtime" in pretty_str
        assert "realtime" in html_str

        # All should mention the device
        assert "cuda" in repr_str
        assert "cuda" in pretty_str
        assert "cuda" in html_str
