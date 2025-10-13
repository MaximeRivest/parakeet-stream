#!/bin/bash
# Install parakeet-hotkey as a user systemd service

set -e

SERVICE_FILE="parakeet-hotkey.service"
SERVICE_DIR="$HOME/.config/systemd/user"

# Create systemd user directory if it doesn't exist
mkdir -p "$SERVICE_DIR"

# Copy service file
cp "$SERVICE_FILE" "$SERVICE_DIR/"

# Reload systemd
systemctl --user daemon-reload

# Enable service to start on login
systemctl --user enable parakeet-hotkey.service

# Start service now
systemctl --user start parakeet-hotkey.service

echo "✓ Service installed and started!"
echo ""
echo "Useful commands:"
echo "  systemctl --user status parakeet-hotkey   # Check status"
echo "  systemctl --user stop parakeet-hotkey     # Stop service"
echo "  systemctl --user restart parakeet-hotkey  # Restart service"
echo "  journalctl --user -u parakeet-hotkey -f   # View logs"
