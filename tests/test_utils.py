"""
Tests for utility functions
"""
import pytest
import torch

from parakeet_stream.utils import make_divisible_by, format_timestamp


def test_make_divisible_by():
    """Test make_divisible_by function."""
    assert make_divisible_by(10, 3) == 9
    assert make_divisible_by(12, 4) == 12
    assert make_divisible_by(15, 5) == 15
    assert make_divisible_by(7, 3) == 6
    assert make_divisible_by(100, 7) == 98


def test_format_timestamp():
    """Test timestamp formatting."""
    assert format_timestamp(0.0) == "00:00:00.000"
    assert format_timestamp(1.5) == "00:00:01.500"
    assert format_timestamp(65.123) == "00:01:05.123"
    assert format_timestamp(3661.456) == "01:01:01.456"
    assert format_timestamp(7384.789) == "02:03:04.789"


def test_format_timestamp_edge_cases():
    """Test timestamp formatting edge cases."""
    # Very small values
    assert format_timestamp(0.001) == "00:00:00.001"

    # Exactly one hour
    assert format_timestamp(3600.0) == "01:00:00.000"

    # Large values
    assert format_timestamp(36000.5) == "10:00:00.500"
