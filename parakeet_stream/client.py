"""
WebSocket client for streaming transcription with Parakeet
"""

import asyncio
import json
import logging
from typing import Optional, Generator, Dict, Any
import numpy as np
import time

try:
    import websockets
except ImportError:
    raise ImportError(
        "websockets is required for client functionality. "
        "Install with: pip install websockets"
    )

try:
    from parakeet_stream import Microphone, AudioClip
    MICROPHONE_AVAILABLE = True
except ImportError:
    MICROPHONE_AVAILABLE = False


logger = logging.getLogger(__name__)


class ParakeetClient:
    """
    WebSocket client for streaming audio to a Parakeet transcription server.

    The client can:
    - Stream audio from microphone in real-time
    - Stream audio from files
    - Send raw audio data
    - Receive transcription segments as they arrive

    Example:
        >>> from parakeet_stream import ParakeetClient
        >>>
        >>> # Connect to server
        >>> client = ParakeetClient('ws://localhost:8765')
        >>>
        >>> # Stream from microphone
        >>> for segment in client.stream_microphone(duration=10):
        ...     print(f"[{segment['timestamp']:.1f}s]: {segment['text']}")
        >>>
        >>> # Stream from file
        >>> for segment in client.stream_file('audio.wav'):
        ...     print(segment['text'])
    """

    def __init__(
        self,
        server_url: str,
        timeout: float = 30.0,
        reconnect: bool = True,
        max_reconnect_attempts: int = 3,
    ):
        """
        Initialize the Parakeet client.

        Args:
            server_url: WebSocket server URL (e.g., 'ws://localhost:8765')
            timeout: Connection timeout in seconds
            reconnect: Whether to automatically reconnect on failure
            max_reconnect_attempts: Maximum number of reconnection attempts
        """
        self.server_url = server_url
        self.timeout = timeout
        self.reconnect = reconnect
        self.max_reconnect_attempts = max_reconnect_attempts

        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self.connected = False
        self.server_config: Optional[Dict[str, Any]] = None

    async def connect(self):
        """Connect to the server and perform handshake."""
        logger.info(f"Connecting to {self.server_url}")

        self.websocket = await websockets.connect(
            self.server_url,
            ping_interval=20,
            ping_timeout=10,
            max_size=20 * 1024 * 1024  # 20 MB to match server
        )
        self.connected = True

        # Wait for handshake
        handshake_msg = await asyncio.wait_for(
            self.websocket.recv(),
            timeout=self.timeout
        )
        handshake = json.loads(handshake_msg)

        if handshake.get("type") != "ready":
            raise RuntimeError(f"Unexpected handshake: {handshake}")

        self.server_config = handshake.get("config", {})
        logger.info(f"Connected to server: {self.server_config}")

    async def disconnect(self):
        """Disconnect from the server."""
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
            self.connected = False
            logger.info("Disconnected from server")

    async def send_audio(self, audio_data: np.ndarray):
        """
        Send audio data to the server.

        Args:
            audio_data: Audio as numpy array (float32, range -1.0 to 1.0)
        """
        if not self.connected or not self.websocket:
            raise RuntimeError("Not connected to server. Call connect() first.")

        # Convert to int16 PCM
        audio_int16 = (audio_data * 32768.0).astype(np.int16)
        audio_bytes = audio_int16.tobytes()

        await self.websocket.send(audio_bytes)

    async def flush(self):
        """
        Flush the audio buffer and get final transcription.

        Returns:
            Dict: Flushed acknowledgment message
        """
        if not self.connected or not self.websocket:
            raise RuntimeError("Not connected to server")

        flush_msg = {"type": "flush"}
        await self.websocket.send(json.dumps(flush_msg))

        # Wait for acknowledgment
        response = await self.websocket.recv()
        return json.loads(response)

    async def receive_segments(self) -> Generator[Dict[str, Any], None, None]:
        """
        Receive transcription segments from the server.

        Yields:
            Dict: Transcription segments with keys:
                - type: 'segment' or 'error'
                - text: Transcribed text
                - confidence: Confidence score
                - duration: Audio duration
        """
        if not self.connected or not self.websocket:
            raise RuntimeError("Not connected to server")

        try:
            async for message in self.websocket:
                data = json.loads(message)
                yield data

        except websockets.exceptions.ConnectionClosed:
            logger.info("Connection closed by server")
            self.connected = False

    def stream_microphone(
        self,
        duration: Optional[float] = None,
        chunk_duration: float = 1.0,
        microphone: Optional['Microphone'] = None,
        verbose: bool = True,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Stream audio from microphone and receive transcriptions.

        Args:
            duration: Total duration to record (None = infinite)
            chunk_duration: Duration of each audio chunk to send
            microphone: Microphone instance (None = use default)
            verbose: Print transcriptions to console

        Yields:
            Dict: Transcription segments from server

        Example:
            >>> client = ParakeetClient('ws://localhost:8765')
            >>> for segment in client.stream_microphone(duration=30):
            ...     print(f"Got: {segment['text']}")
        """
        if not MICROPHONE_AVAILABLE:
            raise ImportError(
                "Microphone support requires parakeet-stream[microphone]. "
                "Install with: pip install 'parakeet-stream[microphone]'"
            )

        # Run the async generator and yield results
        async def _run():
            async for segment in self._stream_microphone_async(
                duration=duration,
                chunk_duration=chunk_duration,
                microphone=microphone,
                verbose=verbose,
            ):
                yield segment

        # Collect all results from async generator
        results = []
        async def _collect():
            async for segment in _run():
                results.append(segment)

        asyncio.run(_collect())

        # Yield collected results
        for result in results:
            yield result

    async def _stream_microphone_async(
        self,
        duration: Optional[float],
        chunk_duration: float,
        microphone: Optional['Microphone'],
        verbose: bool,
    ):
        """Async implementation of stream_microphone."""
        # Connect if not already connected
        if not self.connected:
            await self.connect()

        # Setup microphone
        if microphone is None:
            microphone = Microphone()

        if verbose:
            print(f"🎤 Streaming from: {microphone.name}")
            print(f"   Server: {self.server_url}")
            print(f"   Press Ctrl+C to stop")
            print()

        # Start recording and sending in parallel
        start_time = time.time()
        segment_count = 0

        try:
            # Create tasks for recording and receiving
            async def record_and_send():
                """Record audio and send to server."""
                elapsed = 0.0
                while duration is None or elapsed < duration:
                    # Record chunk
                    clip = microphone.record(duration=chunk_duration)

                    # Send to server
                    await self.send_audio(clip.data)

                    elapsed = time.time() - start_time

                # Send flush message (but don't wait for response - receiver will handle it)
                flush_msg = {"type": "flush"}
                await self.websocket.send(json.dumps(flush_msg))

            async def receive_and_yield():
                """Receive segments and yield them."""
                nonlocal segment_count
                async for segment in self.receive_segments():
                    if segment.get("type") == "segment":
                        segment_count += 1
                        segment["timestamp"] = time.time() - start_time

                        if verbose:
                            print(f"[{segment['timestamp']:.1f}s] {segment['text']}")

                        yield segment

                    elif segment.get("type") == "error":
                        logger.error(f"Server error: {segment.get('message')}")
                        if verbose:
                            print(f"❌ Error: {segment.get('message')}")

                    elif segment.get("type") == "flushed":
                        # Done
                        break

            # Run both tasks concurrently
            send_task = asyncio.create_task(record_and_send())

            async for segment in receive_and_yield():
                yield segment

            # Wait for send task to complete
            await send_task

        except KeyboardInterrupt:
            if verbose:
                print("\n✓ Stopped")

        finally:
            if verbose:
                elapsed = time.time() - start_time
                print(f"\n📊 Summary:")
                print(f"   Duration: {elapsed:.1f}s")
                print(f"   Segments: {segment_count}")

    def stream_file(
        self,
        audio_path: str,
        chunk_duration: float = 2.0,
        realtime: bool = False,
        verbose: bool = True,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Stream audio from a file and receive transcriptions.

        Args:
            audio_path: Path to audio file
            chunk_duration: Duration of each chunk to send
            realtime: Simulate real-time by sleeping between chunks
            verbose: Print progress

        Yields:
            Dict: Transcription segments from server

        Example:
            >>> client = ParakeetClient('ws://localhost:8765')
            >>> for segment in client.stream_file('audio.wav'):
            ...     print(segment['text'])
        """
        # Run the async generator and yield results
        async def _run():
            async for segment in self._stream_file_async(
                audio_path=audio_path,
                chunk_duration=chunk_duration,
                realtime=realtime,
                verbose=verbose,
            ):
                yield segment

        # Collect all results from async generator
        results = []
        async def _collect():
            async for segment in _run():
                results.append(segment)

        asyncio.run(_collect())

        # Yield collected results
        for result in results:
            yield result

    async def _stream_file_async(
        self,
        audio_path: str,
        chunk_duration: float,
        realtime: bool,
        verbose: bool,
    ):
        """Async implementation of stream_file."""
        import soundfile as sf

        # Connect if not already connected
        if not self.connected:
            await self.connect()

        if verbose:
            print(f"📁 Streaming file: {audio_path}")
            print(f"   Server: {self.server_url}")
            print()

        # Read audio file
        audio, sample_rate = sf.read(audio_path, dtype='float32')

        # Ensure mono
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        # Resample if needed (simple decimation/interpolation)
        if sample_rate != 16000:
            from scipy.signal import resample
            target_length = int(len(audio) * 16000 / sample_rate)
            audio = resample(audio, target_length)
            sample_rate = 16000

        total_duration = len(audio) / sample_rate
        chunk_size = int(chunk_duration * sample_rate)

        if verbose:
            print(f"   Duration: {total_duration:.1f}s")
            print(f"   Chunk size: {chunk_duration}s")
            print()

        start_time = time.time()
        segment_count = 0

        try:
            async def send_chunks():
                """Send audio chunks to server."""
                for i in range(0, len(audio), chunk_size):
                    chunk = audio[i:i + chunk_size]
                    await self.send_audio(chunk)

                    if realtime:
                        # Simulate real-time by sleeping
                        await asyncio.sleep(chunk_duration)

                # Send flush message (but don't wait for response - receiver will handle it)
                flush_msg = {"type": "flush"}
                await self.websocket.send(json.dumps(flush_msg))

            async def receive_and_yield():
                """Receive segments and yield them."""
                nonlocal segment_count
                async for segment in self.receive_segments():
                    if segment.get("type") == "segment":
                        segment_count += 1

                        if verbose:
                            print(f"📝 {segment['text']}")
                            print(f"   Confidence: {segment['confidence']:.1%}")
                            print()

                        yield segment

                    elif segment.get("type") == "error":
                        logger.error(f"Server error: {segment.get('message')}")
                        if verbose:
                            print(f"❌ Error: {segment.get('message')}")

                    elif segment.get("type") == "flushed":
                        break

            # Run both tasks concurrently
            send_task = asyncio.create_task(send_chunks())

            async for segment in receive_and_yield():
                yield segment

            await send_task

        finally:
            if verbose:
                elapsed = time.time() - start_time
                print(f"✓ Complete")
                print(f"   Processed: {total_duration:.1f}s audio in {elapsed:.1f}s")
                print(f"   Segments: {segment_count}")

    def transcribe(
        self,
        audio: np.ndarray,
        verbose: bool = False,
    ) -> str:
        """
        Transcribe audio array and return full text.

        Args:
            audio: Audio as numpy array (float32, -1.0 to 1.0, 16kHz mono)
            verbose: Print progress

        Returns:
            str: Full transcribed text

        Example:
            >>> import numpy as np
            >>> audio = np.random.randn(16000 * 5)  # 5 seconds
            >>> text = client.transcribe(audio)
            >>> print(text)
        """
        return asyncio.run(self._transcribe_async(audio, verbose))

    async def _transcribe_async(self, audio: np.ndarray, verbose: bool) -> str:
        """Async implementation of transcribe."""
        if not self.connected:
            await self.connect()

        try:
            # Send audio
            await self.send_audio(audio)

            # Flush and get result
            await self.flush()

            # Collect all segments
            segments = []
            async for segment in self.receive_segments():
                if segment.get("type") == "segment":
                    segments.append(segment['text'])
                    if verbose:
                        print(f"Got segment: {segment['text']}")
                elif segment.get("type") == "flushed":
                    break

            return " ".join(segments)

        finally:
            pass


# Convenience context manager
class ParakeetConnection:
    """
    Context manager for automatic connection/disconnection.

    Example:
        >>> async with ParakeetConnection('ws://localhost:8765') as client:
        ...     result = await client.transcribe(audio)
        ...     print(result)
    """

    def __init__(self, server_url: str, **kwargs):
        self.client = ParakeetClient(server_url, **kwargs)

    async def __aenter__(self):
        await self.client.connect()
        return self.client

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.disconnect()


if __name__ == "__main__":
    # Simple CLI for testing
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    if len(sys.argv) < 2:
        print("Usage: python -m parakeet_stream.client <server_url> [microphone|file <path>]")
        print()
        print("Examples:")
        print("  python -m parakeet_stream.client ws://localhost:8765 microphone")
        print("  python -m parakeet_stream.client ws://localhost:8765 file audio.wav")
        sys.exit(1)

    server_url = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else "microphone"

    client = ParakeetClient(server_url)

    if mode == "microphone":
        print("Starting microphone stream...")
        for segment in client.stream_microphone(duration=30):
            pass  # Already printed in verbose mode

    elif mode == "file":
        if len(sys.argv) < 4:
            print("Error: File path required")
            sys.exit(1)

        file_path = sys.argv[3]
        print(f"Streaming file: {file_path}")

        for segment in client.stream_file(file_path):
            pass  # Already printed in verbose mode
