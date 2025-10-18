"""
WebSocket server for streaming transcription with Parakeet
"""

import asyncio
import json
import logging
from typing import Optional, Dict, Any
import numpy as np

try:
    import websockets
    from websockets.server import WebSocketServerProtocol
except ImportError:
    raise ImportError(
        "websockets is required for server functionality. "
        "Install with: pip install websockets"
    )

from parakeet_stream import Parakeet


logger = logging.getLogger(__name__)


class ParakeetServer:
    """
    WebSocket server for streaming audio transcription.

    The server:
    - Accepts WebSocket connections from clients
    - Receives audio chunks as binary data (16-bit PCM, 16kHz mono)
    - Transcribes audio using Parakeet
    - Streams back transcription segments as JSON

    Example:
        >>> server = ParakeetServer(host='0.0.0.0', port=8765)
        >>> server.start()  # Blocking

        Or with custom Parakeet config:
        >>> server = ParakeetServer(
        ...     host='0.0.0.0',
        ...     port=8765,
        ...     parakeet_config='low_latency',
        ...     device='cuda'
        ... )
        >>> server.start()
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8765,
        parakeet_config: str = "balanced",
        device: str = "cpu",
        chunk_secs: Optional[float] = None,
        left_context_secs: Optional[float] = None,
        right_context_secs: Optional[float] = None,
    ):
        """
        Initialize the Parakeet transcription server.

        Args:
            host: Host to bind to (default: '0.0.0.0')
            port: Port to listen on (default: 8765)
            parakeet_config: Parakeet quality preset (default: 'balanced')
            device: Device to run on ('cpu', 'cuda', 'mps')
            chunk_secs: Optional chunk size override
            left_context_secs: Optional left context override
            right_context_secs: Optional right context override
        """
        self.host = host
        self.port = port
        self.parakeet_config = parakeet_config
        self.device = device

        # Custom parameters
        self.chunk_secs = chunk_secs
        self.left_context_secs = left_context_secs
        self.right_context_secs = right_context_secs

        # Initialize Parakeet
        logger.info(f"Initializing Parakeet with config={parakeet_config}, device={device}")
        self.parakeet = Parakeet(config=parakeet_config, device=device)

        # Apply custom parameters if provided
        if any([chunk_secs, left_context_secs, right_context_secs]):
            params = {}
            if chunk_secs:
                params['chunk_secs'] = chunk_secs
            if left_context_secs:
                params['left_context_secs'] = left_context_secs
            if right_context_secs:
                params['right_context_secs'] = right_context_secs
            self.parakeet.with_params(**params)

        logger.info("Parakeet initialized successfully")

        # Active connections
        self.connections: set = set()

    async def handle_client(self, websocket: WebSocketServerProtocol):
        """
        Handle a single client connection.

        Protocol:
        1. Client connects
        2. Server sends handshake: {"type": "ready", "config": {...}}
        3. Client sends audio chunks as binary data (int16 PCM, 16kHz mono)
        4. Server sends transcription segments: {"type": "segment", "text": "...", "timestamp": 1.5, ...}
        5. Client can send control messages: {"type": "flush"} to flush audio buffer
        """
        client_id = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        logger.info(f"Client connected: {client_id}")
        self.connections.add(websocket)

        # Audio buffer for this connection
        audio_buffer = []
        buffer_duration = 0.0
        sample_rate = 16000

        try:
            # Send handshake
            handshake = {
                "type": "ready",
                "config": {
                    "sample_rate": sample_rate,
                    "config": self.parakeet_config,
                    "device": self.device,
                }
            }
            await websocket.send(json.dumps(handshake))
            logger.info(f"Sent handshake to {client_id}")

            async for message in websocket:
                # Handle binary audio data
                if isinstance(message, bytes):
                    # Convert bytes to numpy array (int16 PCM)
                    audio_chunk = np.frombuffer(message, dtype=np.int16).astype(np.float32) / 32768.0
                    audio_buffer.append(audio_chunk)
                    buffer_duration += len(audio_chunk) / sample_rate

                    # Process when we have enough audio
                    min_chunk_duration = self.parakeet.config.chunk_secs
                    if buffer_duration >= min_chunk_duration:
                        await self._process_audio_buffer(websocket, audio_buffer, sample_rate)
                        audio_buffer = []
                        buffer_duration = 0.0

                # Handle JSON control messages
                elif isinstance(message, str):
                    try:
                        data = json.loads(message)
                        msg_type = data.get("type")

                        if msg_type == "flush":
                            # Process any remaining audio
                            if audio_buffer:
                                await self._process_audio_buffer(websocket, audio_buffer, sample_rate)
                                audio_buffer = []
                                buffer_duration = 0.0

                            # Send acknowledgment
                            await websocket.send(json.dumps({"type": "flushed"}))

                        elif msg_type == "ping":
                            await websocket.send(json.dumps({"type": "pong"}))

                        else:
                            logger.warning(f"Unknown message type from {client_id}: {msg_type}")

                    except json.JSONDecodeError:
                        logger.error(f"Invalid JSON from {client_id}: {message}")

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client disconnected: {client_id}")

        except Exception as e:
            logger.error(f"Error handling client {client_id}: {e}", exc_info=True)
            try:
                error_msg = {
                    "type": "error",
                    "message": str(e)
                }
                await websocket.send(json.dumps(error_msg))
            except:
                pass

        finally:
            self.connections.discard(websocket)
            logger.info(f"Connection closed: {client_id}")

    async def _process_audio_buffer(
        self,
        websocket: WebSocketServerProtocol,
        audio_buffer: list,
        sample_rate: int
    ):
        """Process accumulated audio buffer and send transcription."""
        # Concatenate audio chunks
        audio = np.concatenate(audio_buffer)

        # Transcribe
        try:
            result = self.parakeet.transcribe(audio)

            # Send transcription segment
            segment = {
                "type": "segment",
                "text": result.text,
                "confidence": result.confidence,
                "duration": result.duration,
            }
            await websocket.send(json.dumps(segment))
            logger.debug(f"Sent segment: {result.text[:50]}...")

        except Exception as e:
            logger.error(f"Transcription error: {e}", exc_info=True)
            error_msg = {
                "type": "error",
                "message": f"Transcription failed: {str(e)}"
            }
            await websocket.send(json.dumps(error_msg))

    def start(self):
        """
        Start the server (blocking).

        This will run forever until interrupted (Ctrl+C).
        """
        logger.info(f"Starting Parakeet server on {self.host}:{self.port}")
        print(f"🎤 Parakeet Server")
        print(f"   Listening on ws://{self.host}:{self.port}")
        print(f"   Config: {self.parakeet_config}")
        print(f"   Device: {self.device}")
        print(f"   Press Ctrl+C to stop")
        print()

        async def serve():
            async with websockets.serve(
                self.handle_client,
                self.host,
                self.port,
                max_size=20 * 1024 * 1024  # 20 MB (allows ~10 minutes of audio per message)
            ):
                await asyncio.Future()  # Run forever

        try:
            asyncio.run(serve())
        except KeyboardInterrupt:
            logger.info("Server stopped by user")
            print("\n✓ Server stopped")

    async def start_async(self):
        """
        Start the server asynchronously (non-blocking).

        Returns a server instance that can be closed later.

        Example:
            >>> server = ParakeetServer()
            >>> server_instance = await server.start_async()
            >>> # Do other async work...
            >>> server_instance.close()
            >>> await server_instance.wait_closed()
        """
        logger.info(f"Starting Parakeet server on {self.host}:{self.port}")
        return await websockets.serve(
            self.handle_client,
            self.host,
            self.port,
            max_size=20 * 1024 * 1024  # 20 MB (allows ~10 minutes of audio per message)
        )


# Convenience function
def run_server(
    host: str = "0.0.0.0",
    port: int = 8765,
    config: str = "balanced",
    device: str = "cpu",
):
    """
    Convenience function to quickly start a server.

    Example:
        >>> from parakeet_stream.server import run_server
        >>> run_server(host='0.0.0.0', port=8765, config='low_latency')
    """
    server = ParakeetServer(host=host, port=port, parakeet_config=config, device=device)
    server.start()


if __name__ == "__main__":
    # Simple CLI
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    config = sys.argv[2] if len(sys.argv) > 2 else "balanced"
    device = sys.argv[3] if len(sys.argv) > 3 else "cpu"

    run_server(port=port, config=config, device=device)
