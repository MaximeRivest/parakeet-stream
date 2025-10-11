# Parakeet Stream API Redesign - Implementation Plan

## Overview

Incremental refactoring with testing at each step to ensure robust foundation.

---

## Phase 1: Foundation - Core Classes and Model Loading

### Step 1.1: Create AudioConfig System
**Goal**: Configuration presets without model reloading

**Files to Create**:
- `parakeet_stream/audio_config.py`

**Implementation**:
```python
@dataclass
class AudioConfig:
    """Audio configuration preset"""
    name: str
    chunk_secs: float
    left_context_secs: float
    right_context_secs: float

    @property
    def latency(self) -> float:
        return self.chunk_secs + self.right_context_secs

    @property
    def quality_score(self) -> int:
        """1-5 quality rating"""
        # Based on context windows

    def __repr__(self):
        # Rich display

class ConfigPresets:
    """Named configuration presets"""

    # Presets
    MAXIMUM_QUALITY = AudioConfig(...)
    BALANCED = AudioConfig(...)
    LOW_LATENCY = AudioConfig(...)
    REALTIME = AudioConfig(...)

    @classmethod
    def get(cls, name: str) -> AudioConfig:
        ...

    @classmethod
    def list(cls) -> List[str]:
        ...
```

**Tests**:
```python
def test_audio_config_creation():
    cfg = AudioConfig(name="test", chunk_secs=2.0, ...)
    assert cfg.latency == 4.0

def test_config_presets():
    cfg = ConfigPresets.get('balanced')
    assert cfg.name == 'balanced'
    assert cfg.latency > 0

def test_preset_listing():
    presets = ConfigPresets.list()
    assert 'balanced' in presets
    assert 'realtime' in presets
```

**Verification**:
```python
# In IPython:
from parakeet_stream.audio_config import ConfigPresets
ConfigPresets.list()
cfg = ConfigPresets.BALANCED
print(cfg)
```

**Success Criteria**:
- ✅ 4+ presets defined
- ✅ All tests pass
- ✅ Clean repr in REPL

---

### Step 1.2: Create Display Helpers
**Goal**: Rich IPython/Jupyter display

**Files to Create**:
- `parakeet_stream/display.py`

**Implementation**:
```python
def format_duration(seconds: float) -> str:
    """Format seconds as human readable"""

def format_confidence(score: float) -> str:
    """Format confidence with visual indicator"""

def create_progress_bar(current, total, width=40) -> str:
    """ASCII progress bar"""

class RichRepr:
    """Mixin for rich display"""

    def _repr_pretty_(self, p, cycle):
        """IPython pretty print"""

    def _repr_html_(self):
        """Jupyter HTML display"""
```

**Tests**:
```python
def test_format_duration():
    assert format_duration(65.5) == "1m 5.5s"

def test_format_confidence():
    assert "●●●●●" in format_confidence(0.95)

def test_progress_bar():
    bar = create_progress_bar(50, 100)
    assert "━" in bar
    assert "50%" in bar
```

**Verification**:
```python
from parakeet_stream.display import *
print(format_duration(123.4))
print(format_confidence(0.94))
print(create_progress_bar(75, 100))
```

**Success Criteria**:
- ✅ Utility functions work
- ✅ Tests pass
- ✅ Visual output looks good

---

### Step 1.3: Implement Eager Model Loading with Progress
**Goal**: Load model immediately with progress bar (no lazy init)

**Files to Modify**:
- `parakeet_stream/parakeet.py` (rename from transcriber.py)

**Implementation**:
```python
class Parakeet:
    """Main transcription interface"""

    def __init__(
        self,
        model_name: str = "nvidia/parakeet-tdt-0.6b-v3",
        device: str = "cpu",
        config: Optional[Union[str, AudioConfig]] = None,
        lazy: bool = False  # Default to eager
    ):
        self.model_name = model_name
        self.device = device
        self._config = self._resolve_config(config)
        self._model = None
        self._initialized = False

        if not lazy:
            self.load()  # Eager loading

    def load(self):
        """Load model with progress bar"""
        if self._initialized:
            return

        print(f"Loading {self.model_name} on {self.device}...")

        # Use tqdm or rich for progress
        with tqdm(total=5, desc="Loading model") as pbar:
            pbar.update(1)  # Downloading
            # ... actual loading with callbacks
            pbar.update(4)

        print(f"✓ Ready! ({self.model_name} on {self.device})")
        self._initialized = True

    def _resolve_config(self, config) -> AudioConfig:
        """Resolve config name to AudioConfig"""
        if config is None:
            return ConfigPresets.BALANCED
        if isinstance(config, str):
            return ConfigPresets.get(config)
        return config
```

**Tests**:
```python
def test_parakeet_lazy_loading():
    pk = Parakeet(lazy=True)
    assert not pk._initialized

def test_parakeet_eager_loading():
    pk = Parakeet(lazy=False)
    assert pk._initialized
    assert pk._model is not None

def test_parakeet_default_eager():
    pk = Parakeet()  # Should load by default
    assert pk._initialized
```

**Verification**:
```python
# Time the loading
import time
start = time.time()
pk = Parakeet()
elapsed = time.time() - start
print(f"Loaded in {elapsed:.1f}s")

# Should show progress bar and "✓ Ready!" message
```

**Success Criteria**:
- ✅ Progress bar shows during load
- ✅ Default is eager loading
- ✅ Model ready after __init__
- ✅ Tests pass

---

### Step 1.4: Add Fluent Configuration Methods
**Goal**: Chain config changes without reloading

**Files to Modify**:
- `parakeet_stream/parakeet.py`

**Implementation**:
```python
class Parakeet:
    # ... existing code ...

    def with_config(self, config: Union[str, AudioConfig]) -> 'Parakeet':
        """Set configuration (chainable, no reload)"""
        new_config = self._resolve_config(config)
        self._config = new_config
        self._apply_config_to_model()  # Update decoder params only
        return self

    def with_quality(self, level: str) -> 'Parakeet':
        """Set quality level: 'max', 'high', 'good', 'low', 'realtime'"""
        quality_map = {
            'max': 'maximum_quality',
            'high': 'high_quality',
            'good': 'balanced',
            'low': 'low_latency',
            'realtime': 'realtime'
        }
        return self.with_config(quality_map.get(level, 'balanced'))

    def with_latency(self, level: str) -> 'Parakeet':
        """Set latency level: 'high', 'medium', 'low', 'realtime'"""
        latency_map = {
            'high': 'maximum_quality',
            'medium': 'balanced',
            'low': 'low_latency',
            'realtime': 'realtime'
        }
        return self.with_config(latency_map.get(level, 'balanced'))

    def with_params(
        self,
        chunk_secs: Optional[float] = None,
        left_context_secs: Optional[float] = None,
        right_context_secs: Optional[float] = None
    ) -> 'Parakeet':
        """Set custom parameters (chainable, no reload)"""
        # Create custom config
        custom = AudioConfig(
            name="custom",
            chunk_secs=chunk_secs or self._config.chunk_secs,
            left_context_secs=left_context_secs or self._config.left_context_secs,
            right_context_secs=right_context_secs or self._config.right_context_secs
        )
        return self.with_config(custom)

    def _apply_config_to_model(self):
        """Apply config to loaded model (no reload)"""
        if not self._initialized:
            return
        # Update decoder parameters only
```

**Tests**:
```python
def test_with_config_chainable():
    pk = Parakeet(lazy=True)
    result = pk.with_config('balanced')
    assert result is pk  # Returns self

def test_with_quality():
    pk = Parakeet(lazy=True)
    pk.with_quality('high')
    assert pk._config.quality_score >= 4

def test_with_latency():
    pk = Parakeet(lazy=True)
    pk.with_latency('low')
    assert pk._config.latency < 3.0

def test_with_params():
    pk = Parakeet(lazy=True)
    pk.with_params(chunk_secs=1.5)
    assert pk._config.chunk_secs == 1.5

def test_chaining():
    pk = Parakeet(lazy=True)
    result = pk.with_quality('high').with_params(chunk_secs=5.0)
    assert result is pk
```

**Verification**:
```python
pk = Parakeet()

# Try different configs - should be instant (no reload)
import time
start = time.time()
pk.with_quality('high')
elapsed = time.time() - start
print(f"Config change: {elapsed:.3f}s (should be < 0.1s)")

# Chaining
pk.with_quality('high').with_params(chunk_secs=3.0)
```

**Success Criteria**:
- ✅ Config changes instant (< 0.1s)
- ✅ Chainable (returns self)
- ✅ No model reload
- ✅ Tests pass

---

### Step 1.5: Add Rich Display to Parakeet
**Goal**: Beautiful REPL/Jupyter display

**Files to Modify**:
- `parakeet_stream/parakeet.py`

**Implementation**:
```python
class Parakeet(RichRepr):
    # ... existing code ...

    @property
    def configs(self):
        """Access to config presets"""
        return ConfigPresets

    def __repr__(self):
        """Rich string representation"""
        status = "ready" if self._initialized else "not loaded"
        return (
            f"Parakeet(model='{self.model_name}', device='{self.device}', "
            f"config='{self._config.name}', status='{status}')"
        )

    def _repr_pretty_(self, p, cycle):
        """IPython pretty print"""
        if cycle:
            p.text('Parakeet(...)')
            return

        lines = [
            f"Parakeet(model='{self.model_name}', device='{self.device}')",
            f"  Quality: {self._config.quality_score * '●' + (5-self._config.quality_score) * '○'} ({self._config.name})",
            f"  Latency: ~{self._config.latency:.1f}s",
            f"  Status: {'✓ Ready' if self._initialized else '○ Not loaded'}",
        ]
        p.text('\n'.join(lines))

    def _repr_html_(self):
        """Jupyter HTML display"""
        status_icon = "✅" if self._initialized else "⚪"
        return f"""
        <div style="border: 1px solid #ccc; padding: 10px; border-radius: 5px;">
            <h4>{status_icon} Parakeet</h4>
            <table>
                <tr><td><b>Model:</b></td><td>{self.model_name}</td></tr>
                <tr><td><b>Device:</b></td><td>{self.device}</td></tr>
                <tr><td><b>Config:</b></td><td>{self._config.name}</td></tr>
                <tr><td><b>Latency:</b></td><td>~{self._config.latency:.1f}s</td></tr>
            </table>
        </div>
        """
```

**Tests**:
```python
def test_repr():
    pk = Parakeet(lazy=True)
    repr_str = repr(pk)
    assert "Parakeet" in repr_str
    assert "not loaded" in repr_str

def test_repr_after_load():
    pk = Parakeet()
    repr_str = repr(pk)
    assert "ready" in repr_str
```

**Verification**:
```python
# In IPython
pk = Parakeet()
pk  # Should show rich display

pk.configs  # Should show available configs
pk.configs.BALANCED  # Should show config details
```

**Success Criteria**:
- ✅ Clean repr in Python REPL
- ✅ Rich display in IPython
- ✅ HTML display in Jupyter
- ✅ Tests pass

---

## Phase 2: Transcription Interface

### Step 2.1: Update Transcribe Method
**Goal**: Simple, clean transcribe interface

**Files to Modify**:
- `parakeet_stream/parakeet.py`

**Implementation**:
```python
class Parakeet:
    # ... existing code ...

    def transcribe(
        self,
        audio: Union[str, Path, np.ndarray, torch.Tensor],
        timestamps: bool = False
    ) -> 'TranscriptResult':
        """
        Transcribe audio file or array.

        Args:
            audio: File path, numpy array, or torch tensor
            timestamps: Include word-level timestamps

        Returns:
            TranscriptResult with text, confidence, etc.
        """
        if not self._initialized:
            raise RuntimeError("Model not loaded. Call .load() first.")

        # Load audio if needed
        audio_tensor = self._prepare_audio(audio)

        # Transcribe
        result = self._model.transcribe([audio_tensor])

        return TranscriptResult(
            text=result[0].text,
            confidence=getattr(result[0], 'confidence', None),
            duration=len(audio_tensor) / self.sample_rate,
            timestamps=result[0].timestamps if timestamps else None
        )
```

**Tests**:
```python
def test_transcribe_file(sample_audio_file):
    pk = Parakeet()
    result = pk.transcribe(sample_audio_file)
    assert isinstance(result.text, str)
    assert len(result.text) > 0

def test_transcribe_array():
    pk = Parakeet()
    audio = np.random.randn(16000 * 3)  # 3 seconds
    result = pk.transcribe(audio)
    assert isinstance(result, TranscriptResult)

def test_transcribe_without_load():
    pk = Parakeet(lazy=True)
    with pytest.raises(RuntimeError):
        pk.transcribe("audio.wav")
```

**Verification**:
```python
pk = Parakeet()
result = pk.transcribe("2086-149220-0033.wav")
print(result.text)
print(f"Confidence: {result.confidence}")
print(f"Duration: {result.duration}s")
```

**Success Criteria**:
- ✅ Transcribes files correctly
- ✅ Handles different input types
- ✅ Returns rich result object
- ✅ Tests pass

---

### Step 2.2: Create TranscriptResult Class
**Goal**: Rich result object with metadata

**Files to Create**:
- `parakeet_stream/transcript.py`

**Implementation**:
```python
@dataclass
class TranscriptResult(RichRepr):
    """Result from transcription"""
    text: str
    confidence: Optional[float] = None
    duration: Optional[float] = None
    timestamps: Optional[List[dict]] = None

    def __repr__(self):
        conf_str = f", confidence={self.confidence:.2f}" if self.confidence else ""
        dur_str = f", duration={self.duration:.1f}s" if self.duration else ""
        return f"TranscriptResult(text='{self.text[:50]}...'{conf_str}{dur_str})"

    def _repr_pretty_(self, p, cycle):
        if cycle:
            p.text('TranscriptResult(...)')
            return

        lines = [f"📝 {self.text}"]
        if self.confidence:
            lines.append(f"   Confidence: {format_confidence(self.confidence)}")
        if self.duration:
            lines.append(f"   Duration: {format_duration(self.duration)}")
        p.text('\n'.join(lines))
```

**Tests**:
```python
def test_transcript_result_creation():
    result = TranscriptResult(text="Hello", confidence=0.95)
    assert result.text == "Hello"
    assert result.confidence == 0.95

def test_transcript_result_repr():
    result = TranscriptResult(text="Hello world")
    assert "Hello world" in repr(result)
```

**Verification**:
```python
result = TranscriptResult(text="Test transcription", confidence=0.94, duration=5.2)
result  # Should show rich display
```

**Success Criteria**:
- ✅ Clean dataclass
- ✅ Rich display
- ✅ Tests pass

---

## Phase 3: Microphone Support

### Step 3.1: Create AudioClip Class
**Goal**: Wrapper for recorded audio

**Files to Create**:
- `parakeet_stream/audio_clip.py`

**Implementation**:
```python
class AudioClip(RichRepr):
    """Recorded audio clip"""

    def __init__(self, data: np.ndarray, sample_rate: int):
        self.data = data
        self.sample_rate = sample_rate

    @property
    def duration(self) -> float:
        return len(self.data) / self.sample_rate

    def play(self):
        """Play audio"""
        import sounddevice as sd
        sd.play(self.data, self.sample_rate)
        sd.wait()

    def save(self, path: Union[str, Path]):
        """Save to file"""
        import soundfile as sf
        sf.write(str(path), self.data, self.sample_rate)

    def __repr__(self):
        return f"AudioClip(duration={self.duration:.1f}s, sample_rate={self.sample_rate}Hz)"

    def _repr_pretty_(self, p, cycle):
        p.text(f"🔊 AudioClip ({format_duration(self.duration)}, {self.sample_rate}Hz)")
```

**Tests**:
```python
def test_audio_clip_creation():
    data = np.random.randn(16000 * 2)
    clip = AudioClip(data, 16000)
    assert clip.duration == 2.0

def test_audio_clip_save(tmp_path):
    data = np.random.randn(16000)
    clip = AudioClip(data, 16000)
    path = tmp_path / "test.wav"
    clip.save(path)
    assert path.exists()
```

**Verification**:
```python
data = np.random.randn(16000 * 3)
clip = AudioClip(data, 16000)
clip  # Rich display
clip.play()
```

**Success Criteria**:
- ✅ Clean API
- ✅ Play/save work
- ✅ Tests pass

---

### Step 3.2: Create Microphone Class
**Goal**: Device discovery and recording

**Files to Create**:
- `parakeet_stream/microphone.py`

**Implementation**:
```python
class Microphone(RichRepr):
    """Microphone input manager"""

    def __init__(self, device: Optional[int] = None, sample_rate: int = 16000):
        self.device = device or self._auto_select_device()
        self.sample_rate = sample_rate
        self._device_info = sd.query_devices(self.device)

    @classmethod
    def discover(cls) -> List['Microphone']:
        """Discover all available microphones"""
        devices = sd.query_devices()
        mics = []
        for idx, dev in enumerate(devices):
            if dev['max_input_channels'] > 0:
                mics.append(cls(device=idx))
        return mics

    @staticmethod
    def _auto_select_device() -> int:
        """Auto-select best microphone"""
        # Prefer USB/external over built-in
        devices = sd.query_devices()
        default = sd.default.device[0]
        return default

    def record(self, duration: float = 3.0) -> AudioClip:
        """Record audio"""
        print(f"🎤 Recording {duration}s...")
        samples = int(self.sample_rate * duration)
        data = sd.rec(samples, samplerate=self.sample_rate,
                      channels=1, dtype='float32', device=self.device)
        sd.wait()
        print("✓ Recording complete")
        return AudioClip(data.flatten(), self.sample_rate)

    def test(self, transcriber: 'Parakeet') -> TranscriptResult:
        """Test microphone quality"""
        print("🎤 Microphone test (3 seconds)")
        clip = self.record(3.0)

        print("🔊 Playing back...")
        clip.play()

        print("📝 Transcribing...")
        result = transcriber.transcribe(clip.data)

        print(f"\n✓ Result: {result.text}")
        print(f"  Confidence: {format_confidence(result.confidence)}")

        return result

    def __repr__(self):
        return f"Microphone(device={self.device}, name='{self._device_info['name']}')"

    def _repr_pretty_(self, p, cycle):
        info = self._device_info
        p.text(f"🎤 Microphone {self.device}: {info['name']}\n")
        p.text(f"   Channels: {info['max_input_channels']}, ")
        p.text(f"Sample Rate: {self.sample_rate}Hz")
```

**Tests**:
```python
def test_microphone_discovery():
    mics = Microphone.discover()
    assert len(mics) > 0

def test_microphone_creation():
    mic = Microphone()
    assert mic.device >= 0
    assert mic.sample_rate == 16000

@pytest.mark.slow
def test_microphone_record():
    mic = Microphone()
    clip = mic.record(0.5)  # Short recording
    assert isinstance(clip, AudioClip)
    assert clip.duration > 0
```

**Verification**:
```python
# Discover
mics = Microphone.discover()
for mic in mics:
    print(mic)

# Use default
mic = Microphone()
mic

# Record
clip = mic.record(2.0)
clip.play()

# Test with transcriber
pk = Parakeet()
mic.test(pk)
```

**Success Criteria**:
- ✅ Discovery works
- ✅ Recording works
- ✅ Test function works
- ✅ Tests pass

---

## Phase 4: Live Transcription

### Step 4.1: Create TranscriptBuffer
**Goal**: Growing buffer for live transcription

**Files to Modify**:
- `parakeet_stream/transcript.py`

**Implementation**:
```python
@dataclass
class Segment:
    """Single transcription segment"""
    text: str
    start_time: float
    end_time: float
    confidence: Optional[float] = None

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

class TranscriptBuffer(RichRepr):
    """Growing buffer of transcription segments"""

    def __init__(self):
        self._segments: List[Segment] = []
        self._lock = threading.Lock()

    def append(self, segment: Segment):
        """Add segment (thread-safe)"""
        with self._lock:
            self._segments.append(segment)

    @property
    def text(self) -> str:
        """Full text (all segments)"""
        with self._lock:
            return " ".join(s.text for s in self._segments)

    @property
    def segments(self) -> List[Segment]:
        """All segments (copy)"""
        with self._lock:
            return self._segments.copy()

    def __len__(self):
        return len(self._segments)

    def __getitem__(self, idx):
        with self._lock:
            return self._segments[idx]

    def head(self, n: int = 5) -> List[Segment]:
        """First n segments"""
        return self.segments[:n]

    def tail(self, n: int = 5) -> List[Segment]:
        """Last n segments"""
        return self.segments[-n:]

    @property
    def stats(self) -> dict:
        """Statistics"""
        segs = self.segments
        return {
            'segments': len(segs),
            'duration': segs[-1].end_time if segs else 0,
            'words': sum(len(s.text.split()) for s in segs),
            'avg_confidence': np.mean([s.confidence for s in segs if s.confidence]) if segs else 0
        }

    def to_dict(self) -> dict:
        """Export to dictionary"""
        return {
            'text': self.text,
            'segments': [asdict(s) for s in self.segments],
            'stats': self.stats
        }

    def save(self, path: Union[str, Path]):
        """Save to JSON"""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    def __repr__(self):
        stats = self.stats
        return f"TranscriptBuffer(segments={stats['segments']}, words={stats['words']})"

    def _repr_pretty_(self, p, cycle):
        stats = self.stats
        lines = [
            f"📄 TranscriptBuffer",
            f"   Segments: {stats['segments']} | Words: {stats['words']}",
            f"   Duration: {format_duration(stats['duration'])}",
            f"   Avg Confidence: {format_confidence(stats['avg_confidence'])}"
        ]
        if self._segments:
            lines.append(f"\n   Latest: \"{self._segments[-1].text}\"")
        p.text('\n'.join(lines))
```

**Tests**:
```python
def test_transcript_buffer_append():
    buffer = TranscriptBuffer()
    buffer.append(Segment("Hello", 0.0, 1.0, 0.95))
    assert len(buffer) == 1
    assert buffer.text == "Hello"

def test_transcript_buffer_text():
    buffer = TranscriptBuffer()
    buffer.append(Segment("Hello", 0.0, 1.0))
    buffer.append(Segment("world", 1.0, 2.0))
    assert buffer.text == "Hello world"

def test_transcript_buffer_stats():
    buffer = TranscriptBuffer()
    buffer.append(Segment("Hello world", 0.0, 2.0, 0.95))
    stats = buffer.stats
    assert stats['segments'] == 1
    assert stats['words'] == 2
```

**Verification**:
```python
buffer = TranscriptBuffer()
buffer.append(Segment("Hello", 0.0, 1.0, 0.95))
buffer.append(Segment("world", 1.0, 2.0, 0.93))
buffer  # Rich display
buffer.text
buffer.stats
```

**Success Criteria**:
- ✅ Thread-safe
- ✅ Rich display
- ✅ Stats work
- ✅ Tests pass

---

### Step 4.2: Create LiveTranscriber
**Goal**: Background transcription thread

**Files to Create**:
- `parakeet_stream/live.py`

**Implementation**:
```python
class LiveTranscriber(RichRepr):
    """Background live transcription"""

    def __init__(
        self,
        transcriber: 'Parakeet',
        microphone: Optional[Microphone] = None,
        output: Optional[Union[str, Path]] = None,
        chunk_duration: float = 2.0
    ):
        self.transcriber = transcriber
        self.microphone = microphone or Microphone()
        self.output_file = output
        self.chunk_duration = chunk_duration

        self.transcript = TranscriptBuffer()
        self._running = False
        self._paused = False
        self._thread = None
        self._start_time = None

    def start(self):
        """Start listening"""
        if self._running:
            raise RuntimeError("Already running")

        self._running = True
        self._start_time = time.time()
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()

        print("🎤 Listening... (Ctrl+C to stop)")

    def _listen_loop(self):
        """Background listening loop"""
        audio_queue = queue.Queue()

        def audio_callback(indata, frames, time_info, status):
            if status:
                print(f"Audio error: {status}")
            audio_queue.put(indata.copy())

        chunk_samples = int(self.microphone.sample_rate * self.chunk_duration)

        with sd.InputStream(
            samplerate=self.microphone.sample_rate,
            channels=1,
            dtype='float32',
            callback=audio_callback,
            blocksize=chunk_samples,
            device=self.microphone.device
        ):
            buffer = []
            while self._running:
                if self._paused:
                    time.sleep(0.1)
                    continue

                try:
                    chunk = audio_queue.get(timeout=0.1)
                    buffer.append(chunk)

                    # When we have enough, transcribe
                    if len(buffer) * len(chunk) >= chunk_samples:
                        audio_data = np.concatenate(buffer).flatten()
                        buffer = []

                        # Transcribe
                        result = self.transcriber.transcribe(audio_data)

                        if result.text.strip():
                            # Add to buffer
                            elapsed = time.time() - self._start_time
                            segment = Segment(
                                text=result.text,
                                start_time=elapsed - self.chunk_duration,
                                end_time=elapsed,
                                confidence=result.confidence
                            )
                            self.transcript.append(segment)

                            # Write to file
                            if self.output_file:
                                with open(self.output_file, 'a') as f:
                                    f.write(f"{result.text}\n")

                except queue.Empty:
                    continue

    def pause(self):
        """Pause transcription"""
        self._paused = True
        print("⏸ Paused")

    def resume(self):
        """Resume transcription"""
        self._paused = False
        print("▶ Resumed")

    def stop(self):
        """Stop transcription"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        print(f"\n✓ Stopped. Transcribed {len(self.transcript)} segments")

    @property
    def text(self) -> str:
        """Current full text"""
        return self.transcript.text

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def elapsed(self) -> float:
        if self._start_time:
            return time.time() - self._start_time
        return 0

    def __repr__(self):
        status = "running" if self._running else "stopped"
        return f"LiveTranscriber(status='{status}', segments={len(self.transcript)})"

    def _repr_pretty_(self, p, cycle):
        status = "🟢 Running" if self._running else "⚪ Stopped"
        lines = [
            f"🎤 LiveTranscriber ({status})",
            f"   Duration: {format_duration(self.elapsed)}",
            f"   Segments: {len(self.transcript)}",
        ]
        if self.transcript.segments:
            lines.append(f"   Last: \"{self.transcript.segments[-1].text}\"")
        p.text('\n'.join(lines))
```

**Tests**:
```python
@pytest.mark.slow
def test_live_transcriber_start_stop():
    pk = Parakeet()
    live = LiveTranscriber(pk)

    live.start()
    assert live.is_running

    time.sleep(1.0)

    live.stop()
    assert not live.is_running

def test_live_transcriber_pause_resume():
    pk = Parakeet()
    live = LiveTranscriber(pk)
    live.start()

    live.pause()
    assert live._paused

    live.resume()
    assert not live._paused

    live.stop()
```

**Verification**:
```python
pk = Parakeet()
live = pk.listen()  # This method needs to be added

# Check in real-time
live.text
live.transcript
live  # Rich display

# Control
live.pause()
live.resume()
live.stop()
```

**Success Criteria**:
- ✅ Background thread works
- ✅ Real-time transcription
- ✅ Pause/resume/stop work
- ✅ Thread-safe
- ✅ Tests pass

---

### Step 4.3: Add listen() Method to Parakeet
**Goal**: One-liner to start live transcription

**Files to Modify**:
- `parakeet_stream/parakeet.py`

**Implementation**:
```python
class Parakeet:
    # ... existing code ...

    def listen(
        self,
        microphone: Optional[Microphone] = None,
        output: Optional[Union[str, Path]] = None,
        chunk_duration: float = None
    ) -> LiveTranscriber:
        """
        Start live transcription from microphone.

        Args:
            microphone: Microphone to use (auto-detected if None)
            output: File path to save transcript
            chunk_duration: Duration of chunks to process

        Returns:
            LiveTranscriber object (already started)
        """
        if not self._initialized:
            raise RuntimeError("Model not loaded")

        # Use config's chunk_secs if not specified
        chunk_duration = chunk_duration or self._config.chunk_secs

        live = LiveTranscriber(
            transcriber=self,
            microphone=microphone,
            output=output,
            chunk_duration=chunk_duration
        )
        live.start()
        return live
```

**Tests**:
```python
@pytest.mark.slow
def test_parakeet_listen():
    pk = Parakeet()
    live = pk.listen()

    assert isinstance(live, LiveTranscriber)
    assert live.is_running

    time.sleep(0.5)
    live.stop()
```

**Verification**:
```python
pk = Parakeet()

# One-liner
live = pk.listen()

# With options
live = pk.listen(output="transcript.txt")

# Check progress
live.text
live.transcript.stats

# Stop
live.stop()
```

**Success Criteria**:
- ✅ One-liner works
- ✅ Returns LiveTranscriber
- ✅ Auto-starts
- ✅ Tests pass

---

## Phase 5: API Export and Documentation

### Step 5.1: Update __init__.py
**Goal**: Clean, minimal exports

**Files to Modify**:
- `parakeet_stream/__init__.py`

**Implementation**:
```python
"""
Parakeet Stream - Simple, powerful streaming transcription
"""

__version__ = "0.2.0"

from parakeet_stream.parakeet import Parakeet
from parakeet_stream.microphone import Microphone
from parakeet_stream.audio_config import AudioConfig, ConfigPresets
from parakeet_stream.transcript import TranscriptResult, TranscriptBuffer, Segment
from parakeet_stream.audio_clip import AudioClip
from parakeet_stream.live import LiveTranscriber

__all__ = [
    "Parakeet",
    "Microphone",
    "AudioConfig",
    "ConfigPresets",
    "TranscriptResult",
    "TranscriptBuffer",
    "Segment",
    "AudioClip",
    "LiveTranscriber",
]
```

**Tests**:
```python
def test_imports():
    from parakeet_stream import Parakeet, Microphone
    assert Parakeet is not None
    assert Microphone is not None

def test_version():
    import parakeet_stream
    assert hasattr(parakeet_stream, '__version__')
```

**Verification**:
```python
# Should work
from parakeet_stream import Parakeet, Microphone

# Check exports
import parakeet_stream
dir(parakeet_stream)
```

**Success Criteria**:
- ✅ Clean imports
- ✅ Version set
- ✅ Tests pass

---

### Step 5.2: Create Usage Examples
**Goal**: Document new API with examples

**Files to Create**:
- `examples/01_quick_start.py`
- `examples/02_quality_experiments.py`
- `examples/03_live_transcription.py`
- `examples/04_microphone_testing.py`

**Implementation**: See example files

**Verification**: Run each example

**Success Criteria**:
- ✅ All examples run
- ✅ Clear documentation
- ✅ Cover main use cases

---

### Step 5.3: Update README
**Goal**: Document new API

**Files to Modify**:
- `README.md`

**Updates**:
- New API examples
- Updated installation
- Fluent API docs
- Live transcription docs

**Success Criteria**:
- ✅ Examples match new API
- ✅ Clear, concise
- ✅ Progressive disclosure

---

## Phase 6: Testing and Polish

### Step 6.1: Integration Tests
**Goal**: End-to-end tests

**Files to Create**:
- `tests/test_integration.py`

**Tests**:
```python
def test_full_workflow():
    # Load model
    pk = Parakeet()

    # Transcribe
    result = pk.transcribe("test.wav")
    assert len(result.text) > 0

    # Change config
    pk.with_quality('high')
    result2 = pk.transcribe("test.wav")

    # Should be different
    assert result.text != result2.text or result.confidence != result2.confidence

@pytest.mark.slow
def test_live_workflow():
    pk = Parakeet()

    # Start live
    live = pk.listen()
    time.sleep(2.0)

    # Check buffer grows
    assert len(live.transcript) > 0

    live.stop()
```

**Success Criteria**:
- ✅ All integration tests pass
- ✅ No regressions

---

### Step 6.2: Performance Tests
**Goal**: Verify no model reloading

**Files to Create**:
- `tests/test_performance.py`

**Tests**:
```python
def test_config_change_performance():
    pk = Parakeet()

    # Config changes should be instant
    start = time.time()
    pk.with_quality('high')
    elapsed = time.time() - start

    assert elapsed < 0.1  # Should be near-instant

def test_no_model_reload():
    pk = Parakeet()
    model_id_before = id(pk._model)

    pk.with_quality('high')
    model_id_after = id(pk._model)

    assert model_id_before == model_id_after  # Same object
```

**Success Criteria**:
- ✅ Config changes < 0.1s
- ✅ No reloads

---

### Step 6.3: REPL Experience Test
**Goal**: Verify interactive experience

**Manual Testing Checklist**:
```
□ Import is clean: `from parakeet_stream import Parakeet`
□ Loading shows progress bar
□ Rich display works in IPython
□ Tab completion works
□ Help text is clear
□ Chaining works: `pk.with_quality('high').transcribe(...)`
□ Live transcription updates in real-time
□ Error messages are helpful
```

**Success Criteria**:
- ✅ All checklist items pass
- ✅ Feels smooth and intuitive

---

## Verification Summary

### Phase 1 Tests
```bash
pytest tests/test_audio_config.py -v
pytest tests/test_display.py -v
pytest tests/test_parakeet.py::test_eager_loading -v
pytest tests/test_parakeet.py::test_fluent_api -v
```

### Phase 2 Tests
```bash
pytest tests/test_parakeet.py::test_transcribe -v
pytest tests/test_transcript.py -v
```

### Phase 3 Tests
```bash
pytest tests/test_audio_clip.py -v
pytest tests/test_microphone.py -v
```

### Phase 4 Tests
```bash
pytest tests/test_transcript.py::test_buffer -v
pytest tests/test_live.py -v
```

### Phase 5 Tests
```bash
pytest tests/test_imports.py -v
python examples/01_quick_start.py
```

### Phase 6 Tests
```bash
pytest tests/test_integration.py -v
pytest tests/test_performance.py -v
```

### Full Test Suite
```bash
pytest --cov=parakeet_stream --cov-report=html
```

---

## Success Criteria Summary

**API Design**:
- ✅ Clean, minimal exports
- ✅ Fluent/chainable interface
- ✅ Rich REPL display
- ✅ Sensible defaults

**Performance**:
- ✅ Eager loading with progress
- ✅ No reloads on config change
- ✅ Config changes < 0.1s

**Functionality**:
- ✅ Basic transcription works
- ✅ Live transcription works
- ✅ Microphone support works
- ✅ Quality presets work

**Code Quality**:
- ✅ 90%+ test coverage
- ✅ All tests pass
- ✅ Type hints throughout
- ✅ Clean docstrings

**Documentation**:
- ✅ README updated
- ✅ Examples work
- ✅ API docs clear

---

## Rollout Strategy

1. **Create feature branch**: `feature/api-redesign`
2. **Implement phases incrementally**
3. **Test each phase before moving on**
4. **Keep main branch stable**
5. **Merge when all tests pass**

## Backward Compatibility

**Breaking Changes**:
- `StreamingTranscriber` → `Parakeet`
- `TranscriberConfig` → `AudioConfig`
- Lazy loading → Eager loading (default)

**Migration Guide**:
```python
# Old API
from parakeet_stream import StreamingTranscriber
transcriber = StreamingTranscriber()
result = transcriber.transcribe("audio.wav")

# New API
from parakeet_stream import Parakeet
pk = Parakeet()
result = pk.transcribe("audio.wav")
```

---

**Ready to implement Phase 1?**
