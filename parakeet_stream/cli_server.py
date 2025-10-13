#!/usr/bin/env python3
"""
CLI for parakeet-stream server installation and management.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def install_systemd_service():
    """Install parakeet-stream server as a systemd service."""
    # Check if running as root
    if os.geteuid() != 0:
        print("Error: This command needs to be run with sudo")
        print("Usage: sudo parakeet-server install")
        sys.exit(1)

    # Get the actual user (not root)
    actual_user = os.environ.get('SUDO_USER', os.environ.get('USER'))
    actual_home = Path(f"/home/{actual_user}") if actual_user != 'root' else Path.home()

    print(f"Installing Parakeet Server as systemd service...")
    print(f"User: {actual_user}")
    print(f"Home: {actual_home}")
    print()

    # Create service file content
    service_content = f"""[Unit]
Description=Parakeet Transcription Server
After=network.target

[Service]
Type=simple
User={actual_user}
WorkingDirectory={actual_home}
ExecStart={sys.executable} -m parakeet_stream.server 8765 balanced cuda
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

# Resource limits (adjust as needed)
MemoryMax=8G

[Install]
WantedBy=multi-user.target
"""

    # Write service file
    service_path = Path("/etc/systemd/system/parakeet-server.service")
    service_path.write_text(service_content)

    # Reload systemd
    subprocess.run(["systemctl", "daemon-reload"], check=True)

    # Enable service
    subprocess.run(["systemctl", "enable", "parakeet-server"], check=True)

    # Start service
    subprocess.run(["systemctl", "start", "parakeet-server"], check=True)

    print("✓ Service installed and started!")
    print()
    print("Useful commands:")
    print("  sudo systemctl status parakeet-server   # Check status")
    print("  sudo systemctl stop parakeet-server     # Stop service")
    print("  sudo systemctl restart parakeet-server  # Restart service")
    print("  sudo journalctl -u parakeet-server -f   # View logs")
    print()
    print("The server will start automatically on boot.")


def run_server():
    """Run the parakeet-stream server directly (no systemd)."""
    parser = argparse.ArgumentParser(description="Parakeet Transcription Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8765, help="Port to listen on")
    parser.add_argument("--config", type=str, default="balanced",
                       choices=["low_latency", "balanced", "high_quality"],
                       help="Quality preset")
    parser.add_argument("--device", type=str, default="cpu",
                       choices=["cpu", "cuda", "mps"],
                       help="Device to run on")
    parser.add_argument("--chunk-secs", type=float, default=None,
                       help="Audio chunk size in seconds")
    parser.add_argument("--left-context-secs", type=float, default=None,
                       help="Left context in seconds")
    parser.add_argument("--right-context-secs", type=float, default=None,
                       help="Right context in seconds")

    args = parser.parse_args()

    from parakeet_stream.server import ParakeetServer

    server = ParakeetServer(
        host=args.host,
        port=args.port,
        parakeet_config=args.config,
        device=args.device,
        chunk_secs=args.chunk_secs,
        left_context_secs=args.left_context_secs,
        right_context_secs=args.right_context_secs,
    )
    server.start()


def main():
    """Main CLI entrypoint for parakeet-server."""
    parser = argparse.ArgumentParser(
        description="Parakeet Stream Server - Streaming transcription server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run server directly
  parakeet-server run --host 0.0.0.0 --port 8765 --device cuda

  # Install as systemd service (requires sudo)
  sudo parakeet-server install

  # Check service status
  sudo systemctl status parakeet-server
"""
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Install command
    subparsers.add_parser("install", help="Install server as systemd service (requires sudo)")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run server directly")
    run_parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    run_parser.add_argument("--port", type=int, default=8765, help="Port to listen on")
    run_parser.add_argument("--config", type=str, default="balanced",
                           choices=["low_latency", "balanced", "high_quality"],
                           help="Quality preset")
    run_parser.add_argument("--device", type=str, default="cpu",
                           choices=["cpu", "cuda", "mps"],
                           help="Device to run on")
    run_parser.add_argument("--chunk-secs", type=float, default=None,
                           help="Audio chunk size in seconds")
    run_parser.add_argument("--left-context-secs", type=float, default=None,
                           help="Left context in seconds")
    run_parser.add_argument("--right-context-secs", type=float, default=None,
                           help="Right context in seconds")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "install":
        install_systemd_service()
    elif args.command == "run":
        from parakeet_stream.server import ParakeetServer

        server = ParakeetServer(
            host=args.host,
            port=args.port,
            parakeet_config=args.config,
            device=args.device,
            chunk_secs=args.chunk_secs,
            left_context_secs=args.left_context_secs,
            right_context_secs=args.right_context_secs,
        )
        server.start()


if __name__ == "__main__":
    main()
