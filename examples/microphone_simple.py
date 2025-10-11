"""
Simple example of real-time microphone transcription

Requirements:
    pip install parakeet-stream[microphone]
    # or manually: pip install sounddevice
"""
from parakeet_stream import StreamingTranscriber, TranscriberConfig

# Configure for real-time streaming
config = TranscriberConfig(
    model_name="nvidia/parakeet-tdt-0.6b-v3",
    device="cpu",  # Use "cuda" for GPU
    chunk_secs=2.0,  # Process every 2 seconds
    left_context_secs=10.0,
    right_context_secs=2.0,
)

print("Initializing transcriber...")
transcriber = StreamingTranscriber(config=config)

print("\nReady! Use the stream_microphone.py script for full microphone streaming:")
print("  python stream_microphone.py --output transcription.txt")
print("\nOr integrate into your own code using sounddevice.")
