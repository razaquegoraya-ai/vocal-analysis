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

# Restart the application (assuming you're using systemd)
sudo systemctl restart vocal-analysis

echo "Update complete!"