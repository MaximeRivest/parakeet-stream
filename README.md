# Parakeet Stream

**Simple, powerful streaming transcription for Python using NVIDIA's Parakeet TDT 0.6b**

A modern Python library with a beautiful REPL-friendly API for audio transcription, featuring instant quality tuning, live microphone support, and rich interactive displays.

## ✨ Features

- 🎯 **Simple & Intuitive** - Beautiful API designed for interactive use
- 🎨 **Rich Displays** - Gorgeous output in Python REPL, IPython, and Jupyter notebooks
- ⚡ **Instant Quality Tuning** - Switch between 6 quality presets without reloading model
- 🎤 **Live Transcription** - Real-time microphone transcription with one line of code
- 🌊 **Streaming Support** - Process audio in chunks with configurable latency
- 💻 **CPU Optimized** - Efficient inference on CPU (GPU optional)
- 🌍 **25 Languages** - Automatic language detection
- 📦 **Batch Processing** - Transcribe multiple files efficiently
- ⏱️ **Timestamps** - Optional word-level timestamps

## 🚀 Installation

### Quick Install

```bash
# Install with pip
pip install git+https://github.com/maximerivest/parakeet-stream.git

# Or with uv (recommended)
uv pip install git+https://github.com/maximerivest/parakeet-stream.git

# With microphone support
pip install "parakeet-stream[microphone] @ git+https://github.com/maximerivest/parakeet-stream.git"
```

### Install from Source

```bash
git clone https://github.com/maximerivest/parakeet-stream.git
cd parakeet-stream

# Install with uv
uv pip install -e .

# Or with pip
pip install -e .

# With microphone support
uv pip install -e ".[microphone]"
```

### Requirements

- Python 3.9-3.13
- 2GB+ RAM (4GB+ recommended)
- Any modern CPU (GPU optional)

**Note**: Python 3.13 support requires `ml-dtypes>=0.5.0` which is automatically installed as a dependency.

## 📖 Quick Start

### Basic Transcription

```python
from parakeet_stream import Parakeet

# Initialize (loads model with clean progress bar)
pk = Parakeet()

# Transcribe an audio file
result = pk.transcribe("audio.wav")
print(result.text)
```

The model loads immediately on initialization with a clean progress bar (no verbose logging). First run takes 3-5 minutes (downloads ~600MB from HuggingFace), subsequent runs load from cache in ~5 seconds.

### Live Microphone Transcription

```python
from parakeet_stream import Parakeet

# Initialize transcriber
pk = Parakeet()

# Start live transcription (silent mode - no console output)
live = pk.listen()

# Speak into microphone...
# Transcription happens silently in background

# Access transcript
print(live.text)  # Get current text
print(live.transcript.stats)  # Get statistics

# Stop and get results
live.stop()
print(live.transcript.text)

# Verbose mode - prints transcriptions to console
live = pk.listen(verbose=True)
# [2.5s] Hello world
# [4.6s] This is a test
```

### Quality/Latency Tuning

Switch between quality presets instantly - **no model reload needed**!

```python
from parakeet_stream import Parakeet

pk = Parakeet()

# Try different quality levels (no reload!)
pk.with_quality('max').transcribe("audio.wav")      # ●●●●● (15s latency)
pk.with_quality('high').transcribe("audio.wav")     # ●●●●○ (10s latency)
pk.with_quality('good').transcribe("audio.wav")     # ●●●○○ (4s latency)
pk.with_quality('low').transcribe("audio.wav")      # ●●○○○ (2s latency)
pk.with_quality('realtime').transcribe("audio.wav") # ●○○○○ (1s latency)

# Or use preset names
pk.with_config('balanced').transcribe("audio.wav")
pk.with_config('low_latency').transcribe("audio.wav")
```

### Streaming Transcription

Process long audio files in chunks:

```python
from parakeet_stream import Parakeet

pk = Parakeet()

# Stream transcription results as they become available
for chunk in pk.stream("long_audio.wav"):
    print(f"[{chunk.timestamp_start:.1f}s]: {chunk.text}")
    if chunk.is_final:
        print(f"✓ Final: {chunk.text}")
```

### Microphone Features

```python
from parakeet_stream import Parakeet, Microphone

pk = Parakeet()

# Test ALL microphones automatically (recommended!)
results = Microphone.test_all(pk)
# Shows test phrase for you to read
# Tests each microphone with the same phrase
# Ranks by quality and recommends best one
# You can play back any recording: results[0].clip.play()

# Use the best microphone
best_mic = results[0].microphone
live = pk.listen(microphone=best_mic)

# Or manually discover and test
mics = Microphone.discover()
for mic in mics:
    print(mic)
# 🎤 Microphone 0: Built-in Microphone
# 🎤 Microphone 1: USB Microphone

# Test a specific microphone
mic = Microphone(device=1)
test_result = mic.test(pk)
# Shows random test phrase
# Records, transcribes, and evaluates quality
# Returns detailed metrics: match score, confidence, audio level

# Record audio
clip = mic.record(duration=5.0)
clip.play()  # Playback
clip.save("recording.wav")  # Save to file
```

### Batch Processing

```python
from parakeet_stream import Parakeet

pk = Parakeet()

# Transcribe multiple files with progress bar
audio_files = ["file1.wav", "file2.wav", "file3.wav"]
results = pk.transcribe_batch(audio_files, show_progress=True)

for file, result in zip(audio_files, results):
    print(f"{file}: {result.text}")
```

## 🎛️ Configuration Guide

### Quality Presets

Parakeet Stream includes 6 carefully tuned presets for different use cases:

| Preset | Quality | Latency | Use Case |
|--------|---------|---------|----------|
| `maximum_quality` | ●●●●● | ~15s | Offline transcription, highest accuracy |
| `high_quality` | ●●●●○ | ~10s | Long audio files, near-perfect quality |
| `balanced` | ●●●○○ | ~4s | **Default** - Great quality, acceptable latency |
| `low_latency` | ●●○○○ | ~2s | Interactive applications |
| `realtime` | ●○○○○ | ~1s | Live conversations, minimal delay |
| `ultra_realtime` | ●○○○○ | ~0.3s | Experimental ultra-low latency |

```python
from parakeet_stream import Parakeet

# Use preset at initialization
pk = Parakeet(config='balanced')

# Or change on the fly (no reload!)
pk.with_config('high_quality')

# Access preset information
from parakeet_stream import ConfigPresets

print(ConfigPresets.list())
# ['maximum_quality', 'high_quality', 'balanced', 'low_latency', 'realtime', 'ultra_realtime']

print(ConfigPresets.BALANCED)
# balanced:
#   Chunk: 2.0s | Left: 10.0s | Right: 2.0s
#   Latency: ~4.0s | Quality: ●●●○○
```

### Custom Parameters

Fine-tune parameters for specific needs:

```python
from parakeet_stream import Parakeet

pk = Parakeet()

# Adjust individual parameters
pk.with_params(
    chunk_secs=3.0,           # Process in 3-second chunks
    left_context_secs=15.0,   # More context for better quality
    right_context_secs=1.5    # Less lookahead for lower latency
)

result = pk.transcribe("audio.wav")
```

**Understanding Parameters:**

- **chunk_secs**: Size of each processing chunk (affects latency)
- **left_context_secs**: Context from previous audio (improves quality)
- **right_context_secs**: Context from future audio (affects latency)

**Latency Formula**: `latency = chunk_secs + right_context_secs`

### Device Selection

```python
from parakeet_stream import Parakeet

# CPU (default) - works everywhere
pk = Parakeet(device="cpu")

# NVIDIA GPU - 5-10x faster
pk = Parakeet(device="cuda")

# Apple Silicon (M1/M2/M3/M4)
pk = Parakeet(device="mps")
```

### Lazy Loading

By default, models load immediately (eager loading). For advanced use cases:

```python
from parakeet_stream import Parakeet

# Delay model loading
pk = Parakeet(lazy=True)

# Model loads on first use
result = pk.transcribe("audio.wav")

# Or load manually
pk.load()
```

## 🎨 Rich REPL Experience

Parakeet Stream provides beautiful displays in interactive environments:

### Python REPL

```python
>>> from parakeet_stream import Parakeet
>>> pk = Parakeet()

Loading nvidia/parakeet-tdt-0.6b-v3 on cpu...
Loading model:  20%|████████          | 1/5
Moving to device:  40%|████████████████          | 2/5
Configuring streaming:  60%|████████████████████████          | 3/5
Setting up decoder:  80%|████████████████████████████████          | 4/5
Computing context: 100%|████████████████████████████████████████| 5/5
✓ Ready! (nvidia/parakeet-tdt-0.6b-v3 on cpu)

>>> pk
Parakeet(model='nvidia/parakeet-tdt-0.6b-v3', device='cpu', config='balanced', status='ready')
```

### IPython

```python
In [1]: from parakeet_stream import Parakeet
In [2]: pk = Parakeet()
In [3]: pk
Out[3]:
Parakeet(model='nvidia/parakeet-tdt-0.6b-v3', device='cpu')
  Quality: ●●●○○ (balanced)
  Latency: ~4.0s
  Status: ✓ Ready

In [4]: result = pk.transcribe("audio.wav")
In [5]: result
Out[5]:
📝 This is a sample transcription
   Confidence: 95% ●●●●●
   Duration: 5.2s
```

### Jupyter Notebooks

Results display as styled HTML tables with rich formatting.

### Explore Configuration

```python
>>> from parakeet_stream import ConfigPresets
>>> ConfigPresets.list()
['maximum_quality', 'high_quality', 'balanced', 'low_latency', 'realtime', 'ultra_realtime']

>>> ConfigPresets.BALANCED
AudioConfig(name='balanced', latency=4.0s, quality=●●●○○)

>>> print(ConfigPresets.list_with_details())
Available Configuration Presets:

  balanced:
    Chunk: 2.0s | Left: 10.0s | Right: 2.0s
    Latency: ~4.0s | Quality: ●●●○○

  high_quality:
    Chunk: 5.0s | Left: 10.0s | Right: 5.0s
    Latency: ~10.0s | Quality: ●●●●○
  ...
```

## 🎤 Microphone Quality Testing

Not sure which microphone to use? Test them all automatically!

### Test All Microphones

```python
from parakeet_stream import Parakeet, Microphone

pk = Parakeet()

# Automatically test all microphones
results = Microphone.test_all(pk)
```

**What it does:**
1. Discovers all available microphones
2. Shows you a test phrase to read
3. Records from each microphone (same phrase for fair comparison)
4. Transcribes and evaluates quality
5. Detects silent/broken microphones
6. Ranks by quality score (transcription accuracy + confidence)
7. Recommends the best one

**Output:**
```
============================================================
🎤 MICROPHONE QUALITY TEST
============================================================

🔍 Discovering microphones...
✓ Found 3 microphone(s):
   1. Built-in Microphone (device 0)
   2. USB Microphone (device 1)
   3. Bluetooth Headset (device 2)

📝 Test phrase (same for all microphones):

   "Speech recognition technology continues to improve every year"

We'll now test each microphone. Press Enter to start...

... tests each mic ...

============================================================
📊 RESULTS SUMMARY
============================================================

Ranking (Best to Worst):

1. ✓ USB Microphone
   Device: 1
   Quality: [████████████████    ] 82.3%
   Match:   85.0%
   Confidence: 92% ●●●●●
   Audio Level: 0.0523
   Transcribed: "speech recognition technology continues to improve..."

2. ✓ Built-in Microphone
   Device: 0
   Quality: [███████████         ] 65.4%
   Match:   70.0%
   Confidence: 85% ●●●●○
   Audio Level: 0.0312

3. ✗ Bluetooth Headset
   Device: 2
   Quality: [                    ] 0.0%
   Match:   0.0%
   Audio Level: 0.0001
   ⚠️  No audio detected

────────────────────────────────────────────────────────
🏆 RECOMMENDATION
────────────────────────────────────────────────────────

Best microphone: USB Microphone
Device index: 1
Quality score: 82.3%

To use this microphone:
>>> mic = Microphone(device=1)
>>> live = pk.listen(microphone=mic)

============================================================
Tip: You can replay any recording:
>>> results[0].clip.play()  # Play best mic's recording
============================================================
```

### Access Test Results

```python
# Get results
results = Microphone.test_all(pk)

# Use best microphone
best = results[0]
print(f"Best: {best.microphone.name}")
print(f"Quality: {best.quality_score:.1%}")

# Play back recordings
best.clip.play()

# See what was transcribed
print(f"Expected: {best.expected_text}")
print(f"Got: {best.transcribed_text}")

# Check metrics
print(f"Match: {best.match_score:.1%}")
print(f"Confidence: {best.confidence:.1%}")
print(f"Audio level (RMS): {best.rms_level:.4f}")

# Start live transcription with best mic
live = pk.listen(microphone=best.microphone)
```

### Test Single Microphone

```python
pk = Parakeet()
mic = Microphone(device=1)

# Test with random phrase
result = mic.test(pk, duration=5.0)
# Shows phrase, records, transcribes, evaluates

# Test with specific phrase
result = mic.test(pk, phrase="Hello world", duration=3.0)

# Skip playback (faster)
result = mic.test(pk, playback=False)
```

## 🎯 Live Transcription Deep Dive

### Basic Usage

```python
from parakeet_stream import Parakeet

pk = Parakeet()

# Silent mode (default) - no console output
live = pk.listen()

# Transcription runs in background
# Check current transcript
print(live.text)

# Get statistics
print(live.transcript.stats)
# {'segments': 15, 'duration': 45.2, 'words': 234, 'avg_confidence': 0.94}

# Control playback
live.pause()   # Pause transcription
live.resume()  # Resume transcription
live.stop()    # Stop completely

# Verbose mode - prints to console
live = pk.listen(verbose=True)
# 🎤 Listening on: Built-in Microphone
#    (Press Ctrl+C or call .stop() to end)
# [2.5s] Hello world
# [4.6s] This is a test
```

### Save to File

```python
pk = Parakeet()

# Transcription automatically saved to file
live = pk.listen(output="transcript.txt")

# Stop and save complete transcript
live.stop()
live.transcript.save("transcript.json")  # Save with metadata
```

### Custom Microphone

```python
from parakeet_stream import Parakeet, Microphone

# Use specific microphone
mic = Microphone(device=1)  # USB microphone

pk = Parakeet()
live = pk.listen(microphone=mic)
```

### Access Segments

```python
live = pk.listen()

# Wait for some transcription...

# Get all segments
for segment in live.transcript.segments:
    print(f"[{segment.start_time:.1f}s - {segment.end_time:.1f}s] {segment.text}")

# Get last 5 segments
recent = live.transcript.tail(5)

# Get first 5 segments
beginning = live.transcript.head(5)
```

## 📚 API Reference

### Parakeet

Main interface for transcription.

```python
Parakeet(
    model_name: str = "nvidia/parakeet-tdt-0.6b-v3",
    device: str = "cpu",
    config: Union[str, AudioConfig] = "balanced",
    lazy: bool = False
)
```

**Methods:**

- `transcribe(audio, timestamps=False)` → `TranscriptResult`
  - Transcribe audio file or array

- `stream(audio)` → `Generator[StreamChunk]`
  - Stream transcription results as chunks

- `transcribe_batch(audio_files, timestamps=False, show_progress=True)` → `List[TranscriptResult]`
  - Batch transcribe multiple files

- `listen(microphone=None, output=None, chunk_duration=None, verbose=False)` → `LiveTranscriber`
  - Start live microphone transcription (silent by default)

**Configuration Methods (Chainable):**

- `with_config(config)` → `Parakeet`
  - Set configuration preset or custom AudioConfig

- `with_quality(level)` → `Parakeet`
  - Set quality level: 'max', 'high', 'good', 'low', 'realtime'

- `with_latency(level)` → `Parakeet`
  - Set latency level: 'high', 'medium', 'low', 'realtime'

- `with_params(chunk_secs=None, left_context_secs=None, right_context_secs=None)` → `Parakeet`
  - Set custom parameters

**Properties:**

- `config` - Current AudioConfig
- `configs` - Access to ConfigPresets

### TranscriptResult

Rich result object from transcription.

**Attributes:**
- `text` (str) - Transcribed text
- `confidence` (float) - Confidence score (0.0-1.0)
- `duration` (float) - Audio duration in seconds
- `timestamps` (List[dict]) - Word-level timestamps (if enabled)
- `word_count` (int) - Number of words
- `has_timestamps` (bool) - Whether timestamps are available

### LiveTranscriber

Background live transcription manager.

Runs silently by default - transcription happens in background without console output.
Use `verbose=True` to print transcriptions to console.

**Methods:**

- `start()` - Start transcription (called automatically by `pk.listen()`)
- `pause()` - Pause transcription
- `resume()` - Resume transcription
- `stop()` - Stop transcription

**Properties:**

- `text` (str) - Current full transcript
- `transcript` (TranscriptBuffer) - Buffer with all segments
- `is_running` (bool) - Whether currently running
- `is_paused` (bool) - Whether currently paused
- `elapsed` (float) - Elapsed time in seconds
- `verbose` (bool) - Whether console output is enabled

### TranscriptBuffer

Thread-safe buffer for live transcription segments.

**Methods:**

- `append(segment)` - Add segment
- `save(path)` - Save to JSON file
- `head(n=5)` - Get first n segments
- `tail(n=5)` - Get last n segments

**Properties:**

- `text` (str) - Full text (all segments joined)
- `segments` (List[Segment]) - All segments
- `stats` (dict) - Statistics (segments, duration, words, avg_confidence)

### Microphone

Microphone input manager with quality testing.

```python
Microphone(device=None, sample_rate=16000)
```

**Class Methods:**

- `discover()` → `List[Microphone]`
  - Discover all available microphones

- `test_all(transcriber, duration=5.0, playback=False)` → `List[MicrophoneTestResult]`
  - Test all microphones and rank by quality (recommended!)

**Methods:**

- `record(duration=3.0)` → `AudioClip`
  - Record audio for specified duration

- `test(transcriber, duration=5.0, phrase=None, playback=True)` → `MicrophoneTestResult`
  - Test microphone quality with transcription
  - Shows test phrase for user to read
  - Returns detailed quality metrics

**Properties:**

- `name` (str) - Device name
- `channels` (int) - Number of input channels

### MicrophoneTestResult

Result from microphone quality test.

**Attributes:**

- `microphone` (Microphone) - The tested microphone
- `clip` (AudioClip) - Recorded audio (can replay with `.clip.play()`)
- `expected_text` (str) - Text user was supposed to say
- `transcribed_text` (str) - What was actually transcribed
- `confidence` (float) - Transcription confidence score
- `has_audio` (bool) - Whether audio was detected (not silent)
- `rms_level` (float) - Audio level (higher = louder)
- `match_score` (float) - How well transcription matches (0-1)
- `quality_score` (float) - Overall quality (0-1)

### AudioClip

Recorded audio wrapper.

**Methods:**

- `play()` - Play audio through default device
- `save(path)` - Save to WAV file
- `to_tensor()` - Convert to PyTorch tensor

**Properties:**

- `duration` (float) - Duration in seconds
- `num_samples` (int) - Number of samples
- `data` (np.ndarray) - Audio data array
- `sample_rate` (int) - Sample rate in Hz

### ConfigPresets

Pre-configured quality/latency presets.

**Presets:**

- `MAXIMUM_QUALITY` - Best quality (15s latency)
- `HIGH_QUALITY` - High quality (10s latency)
- `BALANCED` - Balanced (4s latency) - **Default**
- `LOW_LATENCY` - Low latency (2s latency)
- `REALTIME` - Real-time (1s latency)
- `ULTRA_REALTIME` - Ultra real-time (0.3s latency)

**Methods:**

- `get(name)` → `AudioConfig` - Get preset by name
- `list()` → `List[str]` - List all preset names
- `list_with_details()` → `str` - Formatted list with details
- `by_quality(level)` → `AudioConfig` - Get by quality level
- `by_latency(level)` → `AudioConfig` - Get by latency level

### AudioConfig

Custom audio configuration.

```python
AudioConfig(
    name: str,
    chunk_secs: float,
    left_context_secs: float,
    right_context_secs: float
)
```

**Properties:**

- `latency` (float) - Theoretical latency in seconds
- `quality_score` (int) - Quality rating (1-5)
- `quality_indicator` (str) - Visual indicator (●●●○○)

## 📂 Examples

The `examples/` directory contains complete working examples:

### Available Examples

- **simple_transcribe.py** - Basic file transcription
- **streaming_transcribe.py** - Streaming with custom configuration
- **batch_transcribe.py** - Batch processing multiple files
- **test_microphones.py** - 🎤 **Test all microphones and find the best one**
- **microphone_simple.py** - Simple microphone recording
- **stream_microphone.py** - Full-featured live transcription
- **benchmark.py** - Compare configurations and benchmark performance

### Running Examples

```bash
# Test all microphones (recommended first step!)
python examples/test_microphones.py

# Simple transcription
python examples/simple_transcribe.py

# Live microphone (Ctrl+C to stop)
python examples/stream_microphone.py

# Save transcript to file
python examples/stream_microphone.py --output transcript.txt

# Use different quality preset
python examples/stream_microphone.py --config low_latency

# Benchmark different configurations
python examples/benchmark.py --audio audio.wav --benchmark
```

## 🌍 Supported Languages

The model automatically detects and transcribes in **25 European languages**:

Bulgarian, Croatian, Czech, Danish, Dutch, English, Estonian, Finnish, French, German, Greek, Hungarian, Italian, Latvian, Lithuanian, Maltese, Polish, Portuguese, Romanian, Russian, Slovak, Slovenian, Spanish, Swedish, Ukrainian

## 🚀 Performance

### Speed

- **CPU**: ~2-3x real-time on modern CPUs (transcribe 1 hour in 20-30 minutes)
- **GPU**: ~10x real-time on NVIDIA GPUs (transcribe 1 hour in 6 minutes)
- **Apple Silicon**: ~3-5x real-time on M1/M2/M3/M4

### Memory

- **CPU**: 2-4GB RAM
- **GPU**: 2-4GB RAM + 2GB VRAM
- **Model Size**: ~600MB download

### First Run

Model downloads from HuggingFace on first run (~600MB). Subsequent runs load from cache (~3-5 seconds).

## 🛠️ Development

### Setup Development Environment

```bash
# Clone repository
git clone https://github.com/maximerivest/parakeet-stream.git
cd parakeet-stream

# Install with dev dependencies
uv pip install -e ".[dev]"

# Install with microphone support
uv pip install -e ".[dev,microphone]"
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=parakeet_stream --cov-report=html

# Run specific test file
pytest tests/test_parakeet.py

# Run specific test
pytest tests/test_parakeet.py::test_transcribe

# Run verbose
pytest -v
```

### Code Quality

```bash
# Format code
black parakeet_stream/

# Lint code
ruff check parakeet_stream/

# Type checking (if using mypy)
mypy parakeet_stream/
```

## 🐛 Troubleshooting

### Installation Issues

**Build errors during installation:**

```bash
# Install build dependencies first
pip install "Cython>=0.29.0" "numpy>=1.20.0"

# Then install the package
pip install -e .
```

**Python 3.13 compatibility:**

The package automatically installs `ml-dtypes>=0.5.0` for Python 3.13 support.

### Microphone Issues

**Linux (Ubuntu/Debian):**

```bash
sudo apt-get install portaudio19-dev
pip install sounddevice --force-reinstall
```

**Linux (Fedora/RHEL):**

```bash
sudo dnf install portaudio-devel
pip install sounddevice --force-reinstall
```

**macOS:**

```bash
brew install portaudio
pip install sounddevice --force-reinstall
```

**Test microphone:**

```python
from parakeet_stream import Microphone

# List available microphones
mics = Microphone.discover()
for mic in mics:
    print(mic)

# Test specific microphone
mic = Microphone(device=0)
clip = mic.record(2.0)
clip.play()
```

### Performance Issues

**Slow transcription:**

- Use GPU if available: `Parakeet(device="cuda")`
- Use lower quality preset: `pk.with_config('low_latency')`
- Close other applications to free RAM
- Check CPU usage - transcription is CPU-intensive

**High memory usage:**

- Use `lazy=True` for delayed loading
- Process files in smaller batches
- Reduce context window sizes with `pk.with_params()`

**Model download fails:**

```bash
# Set HuggingFace cache directory
export HF_HOME=/path/to/cache

# Or use offline mode (requires cached model)
export HF_HUB_OFFLINE=1
```

### Common Errors

**`RuntimeError: Model not loaded`:**

If using `lazy=True`, call `pk.load()` before transcribing.

**`ImportError: sounddevice is required`:**

Install microphone dependencies:
```bash
pip install "parakeet-stream[microphone]"
```

**Audio format errors:**

Ensure audio is 16kHz mono WAV. Convert with:
```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
```

## 📄 License

MIT License - See LICENSE file for details.

This library uses NVIDIA's Parakeet TDT model, which is licensed under [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/).

## 🙏 Acknowledgments

- Built on [NVIDIA NeMo](https://github.com/NVIDIA/NeMo)
- Uses [Parakeet TDT 0.6b v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) model
- Inspired by NVIDIA's streaming inference examples

## 📖 Citation

If you use this library in your research, please cite the Parakeet model:

```bibtex
@misc{parakeet-tdt-0.6b-v3,
  title={Parakeet TDT 0.6B V3},
  author={NVIDIA},
  year={2025},
  url={https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## 🛠️ CLI Tools

Parakeet Stream includes production-ready CLI tools for server and client deployment.

### Server CLI

Install and run the transcription server:

```bash
# Run server directly with uvx (no installation needed)
uvx --from parakeet-stream parakeet-server run --host 0.0.0.0 --port 8765 --device cuda

# Or install as systemd service for production (requires sudo)
uvx --from parakeet-stream parakeet-server install

# Check service status
sudo systemctl status parakeet-server
sudo journalctl -u parakeet-server -f  # View logs
```

**Server options:**
- `--host`: Host to bind to (default: 0.0.0.0)
- `--port`: Port to listen on (default: 8765)
- `--device`: Device to use (cpu, cuda, mps)
- `--config`: Quality preset (low_latency, balanced, high_quality)
- `--chunk-secs`: Audio chunk size in seconds
- `--left-context-secs`: Left context window
- `--right-context-secs`: Right context window

### Client CLI (Hotkey Transcription)

System-wide hotkey transcription that works anywhere:

```bash
# Run client with uvx (installs dependencies automatically)
uvx --from 'parakeet-stream[hotkey]' parakeet-client run \
  --server ws://192.168.1.100:8765 \
  --auto-paste

# Or install as user systemd service (autostart on login)
uvx --from 'parakeet-stream[hotkey]' parakeet-client install

# Check service status
systemctl --user status parakeet-hotkey
```

**Client features:**
- Press **Alt+W** to start/stop recording
- Transcription copied to clipboard automatically
- Optional auto-paste with smart terminal detection (Ctrl+Shift+V for terminals, Ctrl+V for apps)
- Transcription shown in system status bar (requires `panelstatus`)
- Works system-wide in any application

**Client requirements:**
- Linux with X11 (requires `xdotool` for auto-paste)
- `pynput`, `panelstatus`, `pyperclip` (installed automatically with `[hotkey]` extras)

### Installation as Tools

For persistent installation:

```bash
# Install server tool
uv tool install 'parakeet-stream[server]'

# Install client tool with hotkey dependencies
uv tool install 'parakeet-stream[hotkey]'

# Now use commands directly
parakeet-server run --device cuda
parakeet-client run --server ws://localhost:8765
```

## 💬 Support

- **Documentation**: This README and inline code documentation
- **Issues**: [GitHub Issues](https://github.com/maximerivest/parakeet-stream/issues)
- **Discussions**: [GitHub Discussions](https://github.com/maximerivest/parakeet-stream/discussions)

---

**Made with ❤️ for the speech recognition community**
