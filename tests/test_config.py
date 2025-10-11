"""
Tests for configuration module
"""
import pytest

from parakeet_stream.config import TranscriberConfig


def test_default_config():
    """Test default configuration values."""
    config = TranscriberConfig()

    assert config.model_name == "nvidia/parakeet-tdt-0.6b-v3"
    assert config.device == "cpu"
    assert config.compute_dtype == "float32"
    assert config.chunk_secs == 2.0
    assert config.left_context_secs == 10.0
    assert config.right_context_secs == 2.0
    assert config.batch_size == 1
    assert config.timestamps is False
    assert config.sample_rate == 16000
    assert config.streaming is False


def test_custom_config():
    """Test custom configuration."""
    config = TranscriberConfig(
        model_name="custom/model",
        device="cuda",
        chunk_secs=5.0,
        streaming=True,
    )

    assert config.model_name == "custom/model"
    assert config.device == "cuda"
    assert config.chunk_secs == 5.0
    assert config.streaming is True


def test_invalid_chunk_secs():
    """Test validation of chunk_secs parameter."""
    with pytest.raises(ValueError, match="chunk_secs must be positive"):
        TranscriberConfig(chunk_secs=-1.0)

    with pytest.raises(ValueError, match="chunk_secs must be positive"):
        TranscriberConfig(chunk_secs=0.0)


def test_invalid_context_secs():
    """Test validation of context parameters."""
    with pytest.raises(ValueError, match="left_context_secs must be non-negative"):
        TranscriberConfig(left_context_secs=-1.0)

    with pytest.raises(ValueError, match="right_context_secs must be non-negative"):
        TranscriberConfig(right_context_secs=-1.0)


def test_invalid_sample_rate():
    """Test validation of sample_rate parameter."""
    with pytest.raises(ValueError, match="sample_rate must be positive"):
        TranscriberConfig(sample_rate=-1)

    with pytest.raises(ValueError, match="sample_rate must be positive"):
        TranscriberConfig(sample_rate=0)


def test_invalid_batch_size():
    """Test validation of batch_size parameter."""
    with pytest.raises(ValueError, match="batch_size must be positive"):
        TranscriberConfig(batch_size=-1)

    with pytest.raises(ValueError, match="batch_size must be positive"):
        TranscriberConfig(batch_size=0)


def test_invalid_device():
    """Test validation of device parameter."""
    with pytest.raises(ValueError, match="device must be one of"):
        TranscriberConfig(device="invalid")
