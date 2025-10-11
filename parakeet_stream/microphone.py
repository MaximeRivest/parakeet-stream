"""
Microphone class for audio input and recording.
"""
import random
from dataclasses import dataclass
from typing import List, Optional, TYPE_CHECKING

import numpy as np

from parakeet_stream.audio_clip import AudioClip
from parakeet_stream.display import RichRepr, format_confidence, format_duration, format_sample_rate

if TYPE_CHECKING:
    from parakeet_stream.parakeet import Parakeet
    from parakeet_stream.transcript import TranscriptResult


# Test phrases for microphone quality testing
TEST_PHRASES = [
    "The quick brown fox jumps over the lazy dog",
    "Speech recognition technology continues to improve every year",
    "Python is a powerful programming language for data science",
    "Machine learning models require high quality training data",
    "The weather forecast predicts sunny skies for tomorrow",
    "Artificial intelligence is transforming modern technology",
    "Open source software powers much of the internet today",
    "Deep learning networks excel at pattern recognition tasks",
]


@dataclass
class MicrophoneTestResult:
    """
    Result from testing a microphone.

    Attributes:
        microphone: The microphone that was tested
        clip: Recorded audio clip
        expected_text: Text the user was supposed to say
        transcribed_text: What was actually transcribed
        confidence: Transcription confidence score
        has_audio: Whether audio was detected (not silent)
        rms_level: RMS audio level (higher = louder)
        match_score: How well transcription matches expected (0-1)
    """
    microphone: 'Microphone'
    clip: AudioClip
    expected_text: str
    transcribed_text: str
    confidence: Optional[float]
    has_audio: bool
    rms_level: float
    match_score: float

    @property
    def quality_score(self) -> float:
        """
        Overall quality score (0-1) combining confidence and match.

        Returns:
            Quality score
        """
        if not self.has_audio:
            return 0.0

        conf = self.confidence if self.confidence else 0.5
        return (conf * 0.6 + self.match_score * 0.4)

    def __repr__(self) -> str:
        status = "✓" if self.has_audio else "✗"
        return (
            f"MicrophoneTestResult({status} {self.microphone.name}, "
            f"quality={self.quality_score:.2f})"
        )


class Microphone(RichRepr):
    """
    Microphone input manager for audio recording.

    Provides device discovery, recording, and quality testing capabilities.

    Attributes:
        device: Device index or None for default
        sample_rate: Sample rate in Hz (default: 16000)
    """

    def __init__(
        self,
        device: Optional[int] = None,
        sample_rate: int = 16000
    ):
        """
        Initialize microphone.

        Args:
            device: Device index (None for auto-select default)
            sample_rate: Sample rate in Hz (default: 16000)

        Raises:
            ImportError: If sounddevice is not installed
        """
        try:
            import sounddevice as sd
        except ImportError:
            raise ImportError(
                "sounddevice is required for microphone input. "
                "Install with: pip install sounddevice"
            )

        self.sample_rate = sample_rate

        # Auto-select device if not provided
        if device is None:
            self.device = self._auto_select_device()
        else:
            self.device = device

        # Get device info
        import sounddevice as sd
        self._device_info = sd.query_devices(self.device, 'input')

    @classmethod
    def discover(cls) -> List['Microphone']:
        """
        Discover all available microphones.

        Returns:
            List of Microphone objects for each input device

        Raises:
            ImportError: If sounddevice is not installed
        """
        try:
            import sounddevice as sd
        except ImportError:
            raise ImportError(
                "sounddevice is required for microphone discovery. "
                "Install with: pip install sounddevice"
            )

        devices = sd.query_devices()
        microphones = []

        for idx, dev in enumerate(devices):
            # Check if device has input channels
            if dev['max_input_channels'] > 0:
                try:
                    mic = cls(device=idx)
                    microphones.append(mic)
                except Exception:
                    # Skip devices that can't be initialized
                    continue

        return microphones

    @staticmethod
    def _auto_select_device() -> int:
        """
        Auto-select best microphone device.

        Returns:
            Device index

        Raises:
            ImportError: If sounddevice is not installed
        """
        try:
            import sounddevice as sd
        except ImportError:
            raise ImportError(
                "sounddevice is required. "
                "Install with: pip install sounddevice"
            )

        # Use default input device
        default_device = sd.default.device[0]

        if default_device is None or default_device < 0:
            # Fall back to first available input device
            devices = sd.query_devices()
            for idx, dev in enumerate(devices):
                if dev['max_input_channels'] > 0:
                    return idx
            raise RuntimeError("No input devices found")

        return default_device

    def record(self, duration: float = 3.0) -> AudioClip:
        """
        Record audio from microphone.

        Args:
            duration: Recording duration in seconds (default: 3.0)

        Returns:
            AudioClip with recorded audio

        Raises:
            ImportError: If sounddevice is not installed
        """
        try:
            import sounddevice as sd
        except ImportError:
            raise ImportError(
                "sounddevice is required for recording. "
                "Install with: pip install sounddevice"
            )

        print(f"🎤 Recording {format_duration(duration)}...")

        # Record audio
        samples = int(self.sample_rate * duration)
        data = sd.rec(
            samples,
            samplerate=self.sample_rate,
            channels=1,
            dtype='float32',
            device=self.device
        )
        sd.wait()

        print("✓ Recording complete")

        # Flatten to 1D array and return as AudioClip
        return AudioClip(data.flatten(), self.sample_rate)

    @staticmethod
    def _detect_audio(clip: AudioClip, silence_threshold: float = 0.01) -> tuple[bool, float]:
        """
        Detect if audio clip contains actual audio or is silent.

        Args:
            clip: Audio clip to analyze
            silence_threshold: RMS threshold below which audio is considered silent

        Returns:
            Tuple of (has_audio, rms_level)
        """
        rms = np.sqrt(np.mean(clip.data ** 2))
        has_audio = rms > silence_threshold
        return has_audio, float(rms)

    @staticmethod
    def _compute_match_score(expected: str, actual: str) -> float:
        """
        Compute how well actual transcription matches expected text.

        Uses simple word overlap metric (Jaccard similarity).

        Args:
            expected: Expected text
            actual: Actual transcribed text

        Returns:
            Match score (0-1)
        """
        if not actual.strip():
            return 0.0

        # Normalize and split into words
        expected_words = set(expected.lower().split())
        actual_words = set(actual.lower().split())

        if not expected_words:
            return 0.0

        # Jaccard similarity: intersection / union
        intersection = len(expected_words & actual_words)
        union = len(expected_words | actual_words)

        return intersection / union if union > 0 else 0.0

    def test(
        self,
        transcriber: 'Parakeet',
        duration: float = 5.0,
        phrase: Optional[str] = None,
        playback: bool = True
    ) -> MicrophoneTestResult:
        """
        Test microphone quality with transcription.

        Displays a test phrase for the user to read, records, transcribes,
        and evaluates quality.

        Args:
            transcriber: Parakeet transcriber instance
            duration: Test recording duration in seconds (default: 5.0)
            phrase: Test phrase to say (random if None)
            playback: Whether to play back the recording

        Returns:
            MicrophoneTestResult with detailed quality metrics
        """
        # Choose test phrase
        if phrase is None:
            phrase = random.choice(TEST_PHRASES)

        print(f"\n🎤 Microphone Test: {self.name}")
        print(f"   Duration: {format_duration(duration)}")
        print(f"\n📝 Please read this sentence clearly:")
        print(f"\n   \"{phrase}\"")
        print(f"\n   Press Enter when ready...")
        input()

        # Record
        print(f"\n🎤 Recording...")
        clip = self.record(duration)

        # Detect audio level
        has_audio, rms_level = self._detect_audio(clip)

        if not has_audio:
            print(f"\n⚠️  No audio detected (RMS: {rms_level:.4f})")
            print(f"   Check that microphone is not muted and positioned correctly")
            return MicrophoneTestResult(
                microphone=self,
                clip=clip,
                expected_text=phrase,
                transcribed_text="",
                confidence=None,
                has_audio=False,
                rms_level=rms_level,
                match_score=0.0
            )

        print(f"✓ Audio detected (RMS: {rms_level:.4f})")

        # Playback
        if playback:
            print(f"\n🔊 Playing back recording...")
            clip.play()

        # Transcribe
        print(f"\n📝 Transcribing...")
        result = transcriber.transcribe(clip.data)

        # Compute match score
        match_score = self._compute_match_score(phrase, result.text)

        # Create test result
        test_result = MicrophoneTestResult(
            microphone=self,
            clip=clip,
            expected_text=phrase,
            transcribed_text=result.text,
            confidence=result.confidence,
            has_audio=has_audio,
            rms_level=rms_level,
            match_score=match_score
        )

        # Display results
        print(f"\n✓ Test Complete")
        print(f"\n   Expected:     \"{phrase}\"")
        print(f"   Transcribed:  \"{result.text}\"")
        print(f"   Match Score:  {match_score:.1%}")
        if result.confidence:
            print(f"   Confidence:   {format_confidence(result.confidence)}")
        print(f"   Quality:      {test_result.quality_score:.1%}")

        return test_result

    @classmethod
    def test_all(
        cls,
        transcriber: 'Parakeet',
        duration: float = 5.0,
        playback: bool = False
    ) -> List[MicrophoneTestResult]:
        """
        Test all available microphones and compare quality.

        This automatically discovers all microphones, tests each one with the
        same phrase, and ranks them by quality.

        Args:
            transcriber: Parakeet transcriber instance
            duration: Test recording duration per microphone
            playback: Whether to play back recordings (can be tedious)

        Returns:
            List of MicrophoneTestResult objects, sorted by quality (best first)

        Example:
            >>> pk = Parakeet()
            >>> results = Microphone.test_all(pk)
            >>> best_mic = results[0].microphone
            >>> print(f"Best microphone: {best_mic.name}")
        """
        print("\n" + "="*60)
        print("🎤 MICROPHONE QUALITY TEST")
        print("="*60)

        # Discover all microphones
        print("\n🔍 Discovering microphones...")
        microphones = cls.discover()

        if not microphones:
            print("❌ No microphones found!")
            return []

        print(f"✓ Found {len(microphones)} microphone(s):")
        for i, mic in enumerate(microphones, 1):
            print(f"   {i}. {mic.name} (device {mic.device})")

        # Use same phrase for all mics for fair comparison
        phrase = random.choice(TEST_PHRASES)

        print(f"\n📝 Test phrase (same for all microphones):")
        print(f"\n   \"{phrase}\"")
        print(f"\nWe'll now test each microphone. Press Enter to start...")
        input()

        # Test each microphone
        results = []
        for i, mic in enumerate(microphones, 1):
            print(f"\n{'─'*60}")
            print(f"Testing Microphone {i}/{len(microphones)}")
            print(f"{'─'*60}")

            try:
                result = mic.test(
                    transcriber=transcriber,
                    duration=duration,
                    phrase=phrase,
                    playback=playback
                )
                results.append(result)
            except Exception as e:
                print(f"\n❌ Error testing {mic.name}: {e}")
                # Create a failed result
                results.append(MicrophoneTestResult(
                    microphone=mic,
                    clip=AudioClip(np.zeros(int(mic.sample_rate * duration)), mic.sample_rate),
                    expected_text=phrase,
                    transcribed_text="",
                    confidence=None,
                    has_audio=False,
                    rms_level=0.0,
                    match_score=0.0
                ))

        # Sort by quality (best first)
        results.sort(key=lambda r: r.quality_score, reverse=True)

        # Display summary
        print(f"\n{'='*60}")
        print("📊 RESULTS SUMMARY")
        print(f"{'='*60}")

        print(f"\nRanking (Best to Worst):")
        for i, result in enumerate(results, 1):
            status = "✓" if result.has_audio else "✗"
            quality_bar = "█" * int(result.quality_score * 20)
            print(f"\n{i}. {status} {result.microphone.name}")
            print(f"   Device: {result.microphone.device}")
            print(f"   Quality: [{quality_bar:<20}] {result.quality_score:.1%}")
            print(f"   Match:   {result.match_score:.1%}")
            if result.confidence:
                print(f"   Confidence: {format_confidence(result.confidence)}")
            print(f"   Audio Level: {result.rms_level:.4f}")
            if result.has_audio and result.transcribed_text:
                print(f"   Transcribed: \"{result.transcribed_text[:50]}{'...' if len(result.transcribed_text) > 50 else ''}\"")

        # Recommend best microphone
        if results and results[0].has_audio:
            best = results[0]
            print(f"\n{'─'*60}")
            print(f"🏆 RECOMMENDATION")
            print(f"{'─'*60}")
            print(f"\nBest microphone: {best.microphone.name}")
            print(f"Device index: {best.microphone.device}")
            print(f"Quality score: {best.quality_score:.1%}")
            print(f"\nTo use this microphone:")
            print(f">>> mic = Microphone(device={best.microphone.device})")
            print(f">>> live = pk.listen(microphone=mic)")
        else:
            print(f"\n⚠️  No working microphones detected!")
            print(f"   Check that microphones are connected and not muted")

        print(f"\n{'='*60}")
        print(f"Tip: You can replay any recording:")
        print(f">>> results[0].clip.play()  # Play best mic's recording")
        print(f"{'='*60}\n")

        return results

    @property
    def name(self) -> str:
        """
        Device name.

        Returns:
            Human-readable device name
        """
        return self._device_info['name']

    @property
    def channels(self) -> int:
        """
        Number of input channels.

        Returns:
            Channel count
        """
        return self._device_info['max_input_channels']

    def __repr__(self) -> str:
        """
        String representation for Python REPL.

        Returns:
            Compact string with device info
        """
        return (
            f"Microphone(device={self.device}, "
            f"name='{self.name}')"
        )

    def _repr_pretty_(self, p, cycle):
        """
        IPython pretty print representation.

        Args:
            p: IPython printer object
            cycle: Whether there's a circular reference
        """
        if cycle:
            p.text('Microphone(...)')
            return

        lines = [
            f"🎤 Microphone {self.device}: {self.name}",
            f"   Channels: {self.channels}",
            f"   Sample Rate: {format_sample_rate(self.sample_rate)}",
        ]
        p.text('\n'.join(lines))

    def _repr_html_(self) -> str:
        """
        Jupyter HTML representation.

        Returns:
            HTML string for Jupyter display
        """
        return f"""
        <div style="border: 1px solid #ccc; padding: 12px; border-radius: 5px; background-color: #f9f9f9;">
            <h4 style="margin-top: 0;">🎤 Microphone</h4>
            <table style="border-collapse: collapse; width: 100%;">
                <tr>
                    <td style="padding: 4px; font-weight: bold;">Device:</td>
                    <td style="padding: 4px;">{self.device}</td>
                </tr>
                <tr>
                    <td style="padding: 4px; font-weight: bold;">Name:</td>
                    <td style="padding: 4px;">{self.name}</td>
                </tr>
                <tr>
                    <td style="padding: 4px; font-weight: bold;">Channels:</td>
                    <td style="padding: 4px;">{self.channels}</td>
                </tr>
                <tr>
                    <td style="padding: 4px; font-weight: bold;">Sample Rate:</td>
                    <td style="padding: 4px;">{format_sample_rate(self.sample_rate)}</td>
                </tr>
            </table>
        </div>
        """
