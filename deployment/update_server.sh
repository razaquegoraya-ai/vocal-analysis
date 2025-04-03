#!/bin/bash

# Configuration
SERVER_USER="developer1"
SERVER_IP="178.156.162.123"
APP_DIR="/home/developer1/vocal-analysis"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting server update process...${NC}"

# Copy application files
echo "Copying application files..."
scp app.py $SERVER_USER@$SERVER_IP:$APP_DIR/
scp vocal_analyzer.py $SERVER_USER@$SERVER_IP:$APP_DIR/
scp vocal_separator.py $SERVER_USER@$SERVER_IP:$APP_DIR/
scp generate_report.py $SERVER_USER@$SERVER_IP:$APP_DIR/
scp -r templates $SERVER_USER@$SERVER_IP:$APP_DIR/
scp -r static $SERVER_USER@$SERVER_IP:$APP_DIR/

# Copy configuration files if they exist
if [ -f "vocal-analysis.conf" ]; then
    echo "Copying Nginx configuration..."
    scp vocal-analysis.conf $SERVER_USER@$SERVER_IP:$APP_DIR/
    ssh $SERVER_USER@$SERVER_IP "sudo cp $APP_DIR/vocal-analysis.conf /etc/nginx/sites-available/ && sudo nginx -t && sudo systemctl restart nginx"
fi

if [ -f "vocal-analysis.service" ]; then
    echo "Copying systemd service configuration..."
    scp vocal-analysis.service $SERVER_USER@$SERVER_IP:$APP_DIR/
    ssh $SERVER_USER@$SERVER_IP "sudo cp $APP_DIR/vocal-analysis.service /etc/systemd/system/ && sudo systemctl daemon-reload && sudo systemctl restart vocal-analysis.service"
fi

# Restart the application service
echo "Restarting application service..."
ssh $SERVER_USER@$SERVER_IP "sudo systemctl restart vocal-analysis.service"

# Check service status
echo "Checking service status..."
ssh $SERVER_USER@$SERVER_IP "systemctl status vocal-analysis.service"

echo -e "${GREEN}Update complete!${NC}"
echo -e "You can check the application at http://$SERVER_IP/"