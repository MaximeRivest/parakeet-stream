"""
Audio configuration presets for quality vs latency tradeoffs.
"""
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class AudioConfig:
    """
    Audio configuration preset for streaming transcription.

    Controls the tradeoff between transcription quality and latency through
    chunk sizes and context windows.

    Attributes:
        name: Configuration name
        chunk_secs: Size of processing chunks (affects latency)
        left_context_secs: Context from previous audio (improves quality)
        right_context_secs: Context from future audio (affects latency)
    """
    name: str
    chunk_secs: float
    left_context_secs: float
    right_context_secs: float

    @property
    def latency(self) -> float:
        """
        Theoretical latency in seconds.

        Latency = chunk_secs + right_context_secs
        (time to accumulate chunk + time to wait for right context)
        """
        return self.chunk_secs + self.right_context_secs

    @property
    def quality_score(self) -> int:
        """
        Quality rating from 1-5 based on context windows.

        Higher context windows generally produce better quality transcriptions.
        """
        # Calculate based on total context
        total_context = self.left_context_secs + self.chunk_secs + self.right_context_secs

        if total_context >= 20:
            return 5  # Maximum quality
        elif total_context >= 14:
            return 4  # High quality
        elif total_context >= 10:
            return 3  # Good quality
        elif total_context >= 6:
            return 2  # Fair quality
        else:
            return 1  # Low quality (realtime)

    @property
    def quality_indicator(self) -> str:
        """Visual quality indicator (●●●●○)"""
        score = self.quality_score
        return "●" * score + "○" * (5 - score)

    def __repr__(self) -> str:
        return (
            f"AudioConfig(name='{self.name}', "
            f"latency={self.latency:.1f}s, "
            f"quality={self.quality_indicator})"
        )

    def __str__(self) -> str:
        return (
            f"{self.name}:\n"
            f"  Chunk: {self.chunk_secs}s | "
            f"Left: {self.left_context_secs}s | "
            f"Right: {self.right_context_secs}s\n"
            f"  Latency: ~{self.latency:.1f}s | "
            f"Quality: {self.quality_indicator}"
        )


class ConfigPresets:
    """
    Named configuration presets for common use cases.

    Presets range from maximum quality (high latency) to realtime (low latency).
    All presets can be used without reloading the model.
    """

    # Maximum Quality: Best for offline file transcription
    # 25s total context, 15s latency
    MAXIMUM_QUALITY = AudioConfig(
        name="maximum_quality",
        chunk_secs=10.0,
        left_context_secs=10.0,
        right_context_secs=5.0
    )

    # High Quality: Similar to offline, good for long files
    # 20s total context, 10s latency
    HIGH_QUALITY = AudioConfig(
        name="high_quality",
        chunk_secs=5.0,
        left_context_secs=10.0,
        right_context_secs=5.0
    )

    # Balanced: Recommended default for streaming
    # 14s total context, 4s latency
    BALANCED = AudioConfig(
        name="balanced",
        chunk_secs=2.0,
        left_context_secs=10.0,
        right_context_secs=2.0
    )

    # Low Latency: Faster response, slight quality tradeoff
    # 12s total context, 2s latency
    LOW_LATENCY = AudioConfig(
        name="low_latency",
        chunk_secs=1.0,
        left_context_secs=10.0,
        right_context_secs=1.0
    )

    # Realtime: Minimum latency for interactive use
    # 11s total context, 1s latency
    REALTIME = AudioConfig(
        name="realtime",
        chunk_secs=0.5,
        left_context_secs=10.0,
        right_context_secs=0.5
    )

    # Ultra Realtime: Experimental very low latency
    # 10.3s total context, 0.3s latency
    ULTRA_REALTIME = AudioConfig(
        name="ultra_realtime",
        chunk_secs=0.16,
        left_context_secs=10.0,
        right_context_secs=0.14
    )

    @classmethod
    def get(cls, name: str) -> AudioConfig:
        """
        Get preset by name.

        Args:
            name: Preset name (e.g., 'balanced', 'low_latency')

        Returns:
            AudioConfig preset

        Raises:
            ValueError: If preset name not found
        """
        name = name.lower().replace('-', '_').replace(' ', '_')

        presets = {
            'maximum_quality': cls.MAXIMUM_QUALITY,
            'max': cls.MAXIMUM_QUALITY,
            'high_quality': cls.HIGH_QUALITY,
            'high': cls.HIGH_QUALITY,
            'balanced': cls.BALANCED,
            'good': cls.BALANCED,
            'default': cls.BALANCED,
            'low_latency': cls.LOW_LATENCY,
            'low': cls.LOW_LATENCY,
            'realtime': cls.REALTIME,
            'rt': cls.REALTIME,
            'ultra_realtime': cls.ULTRA_REALTIME,
            'ultra': cls.ULTRA_REALTIME,
        }

        if name not in presets:
            available = cls.list()
            raise ValueError(
                f"Unknown preset '{name}'. "
                f"Available: {', '.join(available)}"
            )

        return presets[name]

    @classmethod
    def list(cls) -> List[str]:
        """
        List all available preset names.

        Returns:
            List of preset names
        """
        return [
            'maximum_quality',
            'high_quality',
            'balanced',
            'low_latency',
            'realtime',
            'ultra_realtime',
        ]

    @classmethod
    def list_with_details(cls) -> str:
        """
        List all presets with details.

        Returns:
            Formatted string with all presets
        """
        lines = ["Available Configuration Presets:", ""]

        for name in cls.list():
            config = cls.get(name)
            lines.append(f"  {config}")
            lines.append("")

        return "\n".join(lines)

    @classmethod
    def by_quality(cls, level: str) -> AudioConfig:
        """
        Get preset by quality level.

        Args:
            level: 'max', 'high', 'good', 'low', or 'realtime'

        Returns:
            AudioConfig preset
        """
        quality_map = {
            'max': cls.MAXIMUM_QUALITY,
            'high': cls.HIGH_QUALITY,
            'good': cls.BALANCED,
            'low': cls.LOW_LATENCY,
            'realtime': cls.REALTIME,
        }

        level = level.lower()
        if level not in quality_map:
            raise ValueError(
                f"Unknown quality level '{level}'. "
                f"Available: {', '.join(quality_map.keys())}"
            )

        return quality_map[level]

    @classmethod
    def by_latency(cls, level: str) -> AudioConfig:
        """
        Get preset by latency level.

        Args:
            level: 'high', 'medium', 'low', or 'realtime'

        Returns:
            AudioConfig preset
        """
        latency_map = {
            'high': cls.MAXIMUM_QUALITY,
            'medium': cls.BALANCED,
            'low': cls.LOW_LATENCY,
            'realtime': cls.REALTIME,
        }

        level = level.lower()
        if level not in latency_map:
            raise ValueError(
                f"Unknown latency level '{level}'. "
                f"Available: {', '.join(latency_map.keys())}"
            )

        return latency_map[level]
