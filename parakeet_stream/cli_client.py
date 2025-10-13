#!/usr/bin/env python3
"""
CLI for parakeet-stream client (hotkey transcription).
"""

import argparse
import os
import sys
from pathlib import Path


def install_hotkey_service():
    """Install parakeet-stream hotkey client as a user systemd service."""
    # Get configuration from user
    server_url = input("Server URL [ws://localhost:8765]: ").strip() or "ws://localhost:8765"
    auto_paste = input("Enable auto-paste? [y/N]: ").strip().lower() == 'y'

    user = os.environ.get('USER')
    home = Path.home()

    print(f"\nInstalling Parakeet Hotkey Client as user systemd service...")
    print(f"User: {user}")
    print(f"Server: {server_url}")
    print(f"Auto-paste: {auto_paste}")
    print()

    # Build ExecStart command
    exec_start = f"{sys.executable} -m parakeet_stream.cli_client run --server {server_url}"
    if auto_paste:
        exec_start += " --auto-paste"

    # Create service file content
    service_content = f"""[Unit]
Description=Parakeet Hotkey Transcription Client
After=network.target sound.target

[Service]
Type=simple
Environment="DISPLAY=:0"
Environment="XAUTHORITY={home}/.Xauthority"
Environment="DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/%U/bus"
WorkingDirectory={home}
ExecStart={exec_start}
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
"""

    # Create systemd user directory
    service_dir = home / ".config" / "systemd" / "user"
    service_dir.mkdir(parents=True, exist_ok=True)

    # Write service file
    service_path = service_dir / "parakeet-hotkey.service"
    service_path.write_text(service_content)

    # Reload and enable service
    import subprocess
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    subprocess.run(["systemctl", "--user", "enable", "parakeet-hotkey"], check=True)
    subprocess.run(["systemctl", "--user", "start", "parakeet-hotkey"], check=True)

    print("✓ Service installed and started!")
    print()
    print("Useful commands:")
    print("  systemctl --user status parakeet-hotkey   # Check status")
    print("  systemctl --user stop parakeet-hotkey     # Stop service")
    print("  systemctl --user restart parakeet-hotkey  # Restart service")
    print("  journalctl --user -u parakeet-hotkey -f   # View logs")
    print()
    print("Press Alt+W to start/stop recording!")


def run_hotkey_client():
    """Run the hotkey transcription client."""
    parser = argparse.ArgumentParser(description="Parakeet Hotkey Transcription Client")
    parser.add_argument("--server", type=str, default="ws://localhost:8765",
                       help="WebSocket server URL")
    parser.add_argument("--auto-paste", action="store_true",
                       help="Automatically paste transcription with Ctrl+Shift+V")

    args = parser.parse_args()

    # Check dependencies
    try:
        import pynput
        import panelstatus
        import pyperclip
    except ImportError as e:
        print(f"Error: Missing required dependency: {e}")
        print("\nInstall required packages:")
        print("  uv pip install pynput panelstatus pyperclip")
        sys.exit(1)

    # Import and run
    from examples.hotkey_transcribe import HotkeyTranscriber

    transcriber = HotkeyTranscriber(args.server, auto_paste=args.auto_paste)

    try:
        transcriber.start()
    except KeyboardInterrupt:
        import panelstatus as ps
        ps.status.set("", color=None)
        print("\n👋 Goodbye!")


def main():
    """Main CLI entrypoint for parakeet-client."""
    parser = argparse.ArgumentParser(
        description="Parakeet Stream Client - System-wide hotkey transcription",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run client directly
  parakeet-client run --server ws://192.168.1.100:8765 --auto-paste

  # Install as user systemd service
  parakeet-client install

  # Check service status
  systemctl --user status parakeet-hotkey
"""
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Install command
    subparsers.add_parser("install", help="Install client as user systemd service")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run client directly")
    run_parser.add_argument("--server", type=str, default="ws://localhost:8765",
                           help="WebSocket server URL")
    run_parser.add_argument("--auto-paste", action="store_true",
                           help="Automatically paste transcription")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "install":
        install_hotkey_service()
    elif args.command == "run":
        # Check dependencies
        try:
            import pynput
            import panelstatus
            import pyperclip
        except ImportError as e:
            print(f"Error: Missing required dependency: {e}")
            print("\nInstall required packages:")
            print("  uv pip install pynput panelstatus pyperclip")
            sys.exit(1)

        from examples.hotkey_transcribe import HotkeyTranscriber

        transcriber = HotkeyTranscriber(args.server, auto_paste=args.auto_paste)

        try:
            transcriber.start()
        except KeyboardInterrupt:
            import panelstatus as ps
            ps.status.set("", color=None)
            print("\n👋 Goodbye!")


if __name__ == "__main__":
    main()
