#!/bin/bash

echo "Updating vocal analysis application..."

# Navigate to the application directory
cd /home/developer1/vocal-analysis

# Pull the latest changes
git pull origin main

# Activate virtual environment
source venv/bin/activate

# Install/update dependencies
pip install -r requirements.txt

# Restart the application using systemctl if available, otherwise use supervisor
if command -v systemctl >/dev/null 2>&1; then
    # Try systemctl restart first
    if sudo -n systemctl restart vocal-analysis 2>/dev/null; then
        echo "Service restarted via systemctl"
    else
        # If systemctl fails, try supervisor
        if command -v supervisorctl >/dev/null 2>&1; then
            supervisorctl restart vocal-analysis
            echo "Service restarted via supervisor"
        else
            echo "Warning: Could not restart service automatically"
            echo "Please restart the service manually"
        fi
    fi
else
    # Try supervisor if systemctl is not available
    if command -v supervisorctl >/dev/null 2>&1; then
        supervisorctl restart vocal-analysis
        echo "Service restarted via supervisor"
    else
        echo "Warning: Could not restart service automatically"
        echo "Please restart the service manually"
    fi
fi

echo "Update complete!"