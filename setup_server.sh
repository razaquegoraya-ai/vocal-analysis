#!/bin/bash

# Update system and install required packages
sudo apt update
sudo apt install -y python3-venv python3-pip nginx supervisor

# Create project directory if it doesn't exist
mkdir -p ~/vocal-analysis

# Set up Python virtual environment
cd ~/vocal-analysis
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Create required directories
mkdir -p uploads reports

# Create Nginx configuration
sudo tee /etc/nginx/sites-available/vocal-analysis << 'EOF'
server {
    listen 80;
    server_name 178.156.162.123;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /uploads {
        alias /home/developer1/vocal-analysis/uploads;
    }
}
EOF

# Enable the Nginx site
sudo ln -sf /etc/nginx/sites-available/vocal-analysis /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl restart nginx

# Create Supervisor configuration
sudo tee /etc/supervisor/conf.d/vocal-analysis.conf << 'EOF'
[program:vocal-analysis]
directory=/home/developer1/vocal-analysis
command=/home/developer1/vocal-analysis/venv/bin/python app.py
user=developer1
autostart=true
autorestart=true
stderr_logfile=/var/log/vocal-analysis/err.log
stdout_logfile=/var/log/vocal-analysis/out.log
environment=PYTHONUNBUFFERED=1
EOF

# Create log directory
sudo mkdir -p /var/log/vocal-analysis
sudo chown -R developer1:developer1 /var/log/vocal-analysis

# Reload Supervisor and start the application
sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl start vocal-analysis

echo "Setup completed! The application should now be running."
echo "You can check the status with: sudo supervisorctl status vocal-analysis"