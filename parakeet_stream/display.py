"""
Display helpers for rich REPL/Jupyter output.
"""
from typing import Optional


def format_duration(seconds: float) -> str:
    """
    Format seconds as human-readable duration.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted string (e.g., "1m 23.5s", "45.2s", "1h 2m 15s")

    Examples:
        >>> format_duration(5.2)
        '5.2s'
        >>> format_duration(65.5)
        '1m 5.5s'
        >>> format_duration(3665)
        '1h 1m 5.0s'
    """
    if seconds < 0:
        return "0s"

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60

    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if secs > 0 or not parts:  # Always show seconds if nothing else
        parts.append(f"{secs:.1f}s")

    return " ".join(parts)


def format_confidence(score: Optional[float]) -> str:
    """
    Format confidence score with visual indicator.

    Args:
        score: Confidence score (0.0-1.0) or None

    Returns:
        Formatted string with visual indicator

    Examples:
        >>> format_confidence(0.95)
        '95% ●●●●●'
        >>> format_confidence(0.73)
        '73% ●●●○○'
        >>> format_confidence(None)
        'N/A'
    """
    if score is None:
        return "N/A"

    percentage = score * 100

    # Calculate filled circles (5-point scale)
    filled = round(score * 5)
    filled = max(0, min(5, filled))  # Clamp to 0-5

    circles = "●" * filled + "○" * (5 - filled)

    return f"{percentage:.0f}% {circles}"


def create_progress_bar(
    current: int,
    total: int,
    width: int = 40,
    fill_char: str = "━",
    empty_char: str = "━",
    show_percentage: bool = True
) -> str:
    """
    Create ASCII progress bar.

    Args:
        current: Current progress value
        total: Total value
        width: Width of progress bar in characters
        fill_char: Character for filled portion
        empty_char: Character for empty portion
        show_percentage: Show percentage text

    Returns:
        Formatted progress bar string

    Examples:
        >>> create_progress_bar(50, 100, width=20)
        '[━━━━━━━━━━          ] 50%'
        >>> create_progress_bar(75, 100, width=10)
        '[━━━━━━━━  ] 75%'
    """
    if total <= 0:
        return f"[{empty_char * width}] 0%"

    ratio = min(current / total, 1.0)
    filled_width = int(width * ratio)
    empty_width = width - filled_width

    bar = fill_char * filled_width + empty_char * empty_width

    if show_percentage:
        percentage = int(ratio * 100)
        return f"[{bar}] {percentage}%"
    else:
        return f"[{bar}]"


def format_file_size(bytes: int) -> str:
    """
    Format file size in human-readable format.

    Args:
        bytes: Size in bytes

    Returns:
        Formatted string (e.g., "1.5 MB", "500 KB")

    Examples:
        >>> format_file_size(1024)
        '1.0 KB'
        >>> format_file_size(1536000)
        '1.5 MB'
    """
    if bytes < 1024:
        return f"{bytes} B"
    elif bytes < 1024 * 1024:
        return f"{bytes / 1024:.1f} KB"
    elif bytes < 1024 * 1024 * 1024:
        return f"{bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{bytes / (1024 * 1024 * 1024):.2f} GB"


def format_sample_rate(hz: int) -> str:
    """
    Format sample rate in human-readable format.

    Args:
        hz: Sample rate in Hz

    Returns:
        Formatted string (e.g., "16 kHz", "48 kHz")

    Examples:
        >>> format_sample_rate(16000)
        '16 kHz'
        >>> format_sample_rate(44100)
        '44.1 kHz'
    """
    if hz >= 1000:
        khz = hz / 1000
        if khz == int(khz):
            return f"{int(khz)} kHz"
        else:
            return f"{khz:.1f} kHz"
    else:
        return f"{hz} Hz"


class RichRepr:
    """
    Mixin class for rich display in IPython/Jupyter.

    Classes that inherit from this can implement:
    - _repr_pretty_: For IPython pretty printing
    - _repr_html_: For Jupyter HTML display

    The default __repr__ is used for standard Python REPL.
    """

    def _repr_pretty_(self, p, cycle):
        """
        IPython pretty print representation.

        Args:
            p: IPython printer object
            cycle: Whether there's a circular reference

        Override this method in subclasses for custom IPython display.
        """
        if cycle:
            p.text(f'{self.__class__.__name__}(...)')
        else:
            p.text(repr(self))

    def _repr_html_(self):
        """
        Jupyter HTML representation.

        Returns:
            HTML string for Jupyter display

        Override this method in subclasses for custom Jupyter display.
        """
        return None  # Fall back to _repr_pretty_


def table_row(items: list, widths: Optional[list] = None) -> str:
    """
    Create a formatted table row.

    Args:
        items: List of cell contents
        widths: Optional list of column widths

    Returns:
        Formatted row string

    Examples:
        >>> table_row(['Name', 'Value', 'Status'])
        'Name    Value   Status'
        >>> table_row(['A', 'B', 'C'], widths=[10, 10, 10])
        'A         B         C         '
    """
    if widths:
        cells = [str(item).ljust(width) for item, width in zip(items, widths)]
    else:
        cells = [str(item).ljust(8) for item in items]
    return " ".join(cells)


def format_timestamp(seconds: float) -> str:
    """
    Format timestamp for transcription segments.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted timestamp (e.g., "00:12.5", "01:23.0")

    Examples:
        >>> format_timestamp(12.5)
        '00:12.5'
        >>> format_timestamp(83.2)
        '01:23.2'
    """
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes:02d}:{secs:04.1f}"


def truncate_text(text: str, max_length: int = 50, suffix: str = "...") -> str:
    """
    Truncate text to maximum length.

    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated

    Returns:
        Truncated text

    Examples:
        >>> truncate_text("This is a long sentence", 10)
        'This is...'
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix
