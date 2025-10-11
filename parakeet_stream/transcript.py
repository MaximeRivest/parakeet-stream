"""
Transcript result classes for transcription output.
"""
import json
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from parakeet_stream.display import (
    RichRepr,
    format_confidence,
    format_duration,
    truncate_text,
)


@dataclass
class TranscriptResult(RichRepr):
    """
    Result from transcription.

    Rich result object with text, confidence, duration, and optional timestamps.
    Provides beautiful display in REPL, IPython, and Jupyter.

    Attributes:
        text: Transcribed text
        confidence: Optional confidence score (0.0-1.0)
        duration: Optional audio duration in seconds
        timestamps: Optional word-level timestamps
    """

    text: str
    confidence: Optional[float] = None
    duration: Optional[float] = None
    timestamps: Optional[List[dict]] = None

    def __repr__(self) -> str:
        """
        String representation for Python REPL.

        Returns:
            Compact string with truncated text and key metrics
        """
        text_preview = truncate_text(self.text, max_length=50)

        parts = [f"TranscriptResult(text='{text_preview}'"]

        if self.confidence is not None:
            parts.append(f", confidence={self.confidence:.2f}")

        if self.duration is not None:
            parts.append(f", duration={self.duration:.1f}s")

        parts.append(")")
        return "".join(parts)

    def _repr_pretty_(self, p, cycle):
        """
        IPython pretty print representation.

        Displays multi-line format with text, confidence indicator,
        and duration.

        Args:
            p: IPython printer object
            cycle: Whether there's a circular reference
        """
        if cycle:
            p.text('TranscriptResult(...)')
            return

        lines = [f"📝 {self.text}"]

        if self.confidence is not None:
            lines.append(f"   Confidence: {format_confidence(self.confidence)}")

        if self.duration is not None:
            lines.append(f"   Duration: {format_duration(self.duration)}")

        if self.timestamps:
            lines.append(f"   Words: {len(self.timestamps)}")

        p.text('\n'.join(lines))

    def _repr_html_(self) -> str:
        """
        Jupyter HTML representation.

        Returns styled HTML display with text, confidence, and metadata.

        Returns:
            HTML string for Jupyter display
        """
        # Escape HTML in text
        escaped_text = (
            self.text.replace('&', '&amp;')
            .replace('<', '&lt;')
            .replace('>', '&gt;')
        )

        html_parts = [
            '<div style="border: 1px solid #ccc; padding: 12px; '
            'border-radius: 5px; background-color: #f9f9f9; margin: 8px 0;">',
            '<h4 style="margin-top: 0;">📝 Transcript</h4>',
            f'<p style="margin: 8px 0; font-family: monospace;">{escaped_text}</p>',
            '<table style="border-collapse: collapse; width: 100%; margin-top: 8px;">',
        ]

        if self.confidence is not None:
            conf_display = format_confidence(self.confidence)
            html_parts.append(
                f'<tr><td style="padding: 4px; font-weight: bold;">Confidence:</td>'
                f'<td style="padding: 4px;">{conf_display}</td></tr>'
            )

        if self.duration is not None:
            dur_display = format_duration(self.duration)
            html_parts.append(
                f'<tr><td style="padding: 4px; font-weight: bold;">Duration:</td>'
                f'<td style="padding: 4px;">{dur_display}</td></tr>'
            )

        if self.timestamps:
            html_parts.append(
                f'<tr><td style="padding: 4px; font-weight: bold;">Words:</td>'
                f'<td style="padding: 4px;">{len(self.timestamps)}</td></tr>'
            )

        html_parts.extend(['</table>', '</div>'])

        return ''.join(html_parts)

    @property
    def word_count(self) -> int:
        """
        Number of words in transcribed text.

        Returns:
            Word count (simple whitespace split)
        """
        return len(self.text.split())

    @property
    def has_timestamps(self) -> bool:
        """
        Whether word-level timestamps are available.

        Returns:
            True if timestamps exist
        """
        return self.timestamps is not None and len(self.timestamps) > 0


@dataclass
class Segment:
    """
    Single transcription segment for live transcription.

    Attributes:
        text: Transcribed text for this segment
        start_time: Start time in seconds
        end_time: End time in seconds
        confidence: Optional confidence score
    """

    text: str
    start_time: float
    end_time: float
    confidence: Optional[float] = None

    @property
    def duration(self) -> float:
        """
        Duration of segment in seconds.

        Returns:
            Duration
        """
        return self.end_time - self.start_time


class TranscriptBuffer(RichRepr):
    """
    Growing buffer of transcription segments for live transcription.

    Thread-safe buffer that accumulates segments during live transcription.

    Attributes:
        segments: List of transcription segments
    """

    def __init__(self):
        """Initialize empty transcript buffer."""
        self._segments: List[Segment] = []
        self._lock = threading.Lock()

    def append(self, segment: Segment):
        """
        Add segment to buffer (thread-safe).

        Args:
            segment: Segment to add
        """
        with self._lock:
            self._segments.append(segment)

    @property
    def text(self) -> str:
        """
        Full transcribed text (all segments joined).

        Returns:
            Complete text
        """
        with self._lock:
            return " ".join(s.text for s in self._segments)

    @property
    def segments(self) -> List[Segment]:
        """
        All segments (thread-safe copy).

        Returns:
            List of segments
        """
        with self._lock:
            return self._segments.copy()

    def __len__(self) -> int:
        """Number of segments."""
        return len(self._segments)

    def __getitem__(self, idx: int) -> Segment:
        """Get segment by index."""
        with self._lock:
            return self._segments[idx]

    def head(self, n: int = 5) -> List[Segment]:
        """
        Get first n segments.

        Args:
            n: Number of segments

        Returns:
            First n segments
        """
        return self.segments[:n]

    def tail(self, n: int = 5) -> List[Segment]:
        """
        Get last n segments.

        Args:
            n: Number of segments

        Returns:
            Last n segments
        """
        return self.segments[-n:]

    @property
    def stats(self) -> dict:
        """
        Buffer statistics.

        Returns:
            Dictionary with segments, duration, words, avg_confidence
        """
        segs = self.segments
        if not segs:
            return {
                'segments': 0,
                'duration': 0.0,
                'words': 0,
                'avg_confidence': 0.0
            }

        confidences = [s.confidence for s in segs if s.confidence is not None]
        avg_conf = np.mean(confidences) if confidences else 0.0

        return {
            'segments': len(segs),
            'duration': segs[-1].end_time if segs else 0.0,
            'words': sum(len(s.text.split()) for s in segs),
            'avg_confidence': float(avg_conf)
        }

    def to_dict(self) -> dict:
        """
        Export buffer to dictionary.

        Returns:
            Dictionary with text, segments, and stats
        """
        return {
            'text': self.text,
            'segments': [asdict(s) for s in self.segments],
            'stats': self.stats
        }

    def save(self, path: Union[str, Path]):
        """
        Save transcript buffer to JSON file.

        Args:
            path: Output file path
        """
        path = Path(path)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"✓ Saved transcript to {path}")

    def __repr__(self) -> str:
        """String representation for Python REPL."""
        stats = self.stats
        return (
            f"TranscriptBuffer(segments={stats['segments']}, "
            f"words={stats['words']})"
        )

    def _repr_pretty_(self, p, cycle):
        """IPython pretty print representation."""
        if cycle:
            p.text('TranscriptBuffer(...)')
            return

        stats = self.stats
        lines = [
            "📄 TranscriptBuffer",
            f"   Segments: {stats['segments']} | Words: {stats['words']}",
            f"   Duration: {format_duration(stats['duration'])}",
        ]

        if stats['avg_confidence'] > 0:
            lines.append(
                f"   Avg Confidence: {format_confidence(stats['avg_confidence'])}"
            )

        if self._segments:
            latest_text = truncate_text(self._segments[-1].text, max_length=50)
            lines.append(f"\n   Latest: \"{latest_text}\"")

        p.text('\n'.join(lines))

    def _repr_html_(self) -> str:
        """Jupyter HTML representation."""
        stats = self.stats

        html_parts = [
            '<div style="border: 1px solid #ccc; padding: 12px; '
            'border-radius: 5px; background-color: #f9f9f9;">',
            '<h4 style="margin-top: 0;">📄 TranscriptBuffer</h4>',
            '<table style="border-collapse: collapse; width: 100%;">',
            f'<tr><td style="padding: 4px; font-weight: bold;">Segments:</td>'
            f'<td style="padding: 4px;">{stats["segments"]}</td></tr>',
            f'<tr><td style="padding: 4px; font-weight: bold;">Words:</td>'
            f'<td style="padding: 4px;">{stats["words"]}</td></tr>',
            f'<tr><td style="padding: 4px; font-weight: bold;">Duration:</td>'
            f'<td style="padding: 4px;">{format_duration(stats["duration"])}</td></tr>',
        ]

        if stats['avg_confidence'] > 0:
            conf_display = format_confidence(stats['avg_confidence'])
            html_parts.append(
                f'<tr><td style="padding: 4px; font-weight: bold;">Avg Confidence:</td>'
                f'<td style="padding: 4px;">{conf_display}</td></tr>'
            )

        html_parts.append('</table>')

        if self._segments:
            latest_text = self._segments[-1].text
            escaped_text = (
                latest_text.replace('&', '&amp;')
                .replace('<', '&lt;')
                .replace('>', '&gt;')
            )
            html_parts.append(
                f'<p style="margin-top: 12px; font-family: monospace; '
                f'font-size: 0.9em; color: #666;">'
                f'Latest: "{escaped_text}"</p>'
            )

        html_parts.append('</div>')

        return ''.join(html_parts)
