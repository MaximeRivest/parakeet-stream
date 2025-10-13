#!/bin/bash
# Install parakeet-server as a systemd service on the GPU server

set -e

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SERVICE_FILE="$SCRIPT_DIR/parakeet-server.service"
SERVICE_NAME="parakeet-server"

# Check if running as root (needed for system service)
if [ "$EUID" -ne 0 ]; then
    echo "This script needs to be run with sudo to install a system service"
    echo "Usage: sudo ./install_server_service.sh"
    exit 1
fi

# Get the actual user (not root)
ACTUAL_USER="${SUDO_USER:-$USER}"
ACTUAL_HOME=$(getent passwd "$ACTUAL_USER" | cut -d: -f6)

echo "Installing Parakeet Server as systemd service..."
echo "User: $ACTUAL_USER"
echo "Home: $ACTUAL_HOME"
echo ""

# Create a temporary service file with correct paths
TMP_SERVICE=$(mktemp)
sed "s|/home/maxime|$ACTUAL_HOME|g; s|User=%u|User=$ACTUAL_USER|g" "$SERVICE_FILE" > "$TMP_SERVICE"

# Copy service file to systemd
cp "$TMP_SERVICE" /etc/systemd/system/"$SERVICE_NAME.service"
rm "$TMP_SERVICE"

# Reload systemd
systemctl daemon-reload

# Enable service to start on boot
systemctl enable "$SERVICE_NAME"

# Start service now
systemctl start "$SERVICE_NAME"

echo "✓ Service installed and started!"
echo ""
echo "Useful commands:"
echo "  sudo systemctl status $SERVICE_NAME   # Check status"
echo "  sudo systemctl stop $SERVICE_NAME     # Stop service"
echo "  sudo systemctl restart $SERVICE_NAME  # Restart service"
echo "  sudo journalctl -u $SERVICE_NAME -f   # View logs"
echo ""
echo "The server will start automatically on boot."
