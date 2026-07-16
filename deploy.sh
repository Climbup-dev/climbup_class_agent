#!/bin/bash
set -e

echo "🚀 Starting Deployment on Oracle Cloud..."

# 1. Update and install dependencies
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg

# 2. Install Docker
if ! command -v docker &> /dev/null
then
    echo "🐳 Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker ubuntu
    echo "✅ Docker installed!"
else
    echo "✅ Docker already installed."
fi

# 3. Ask for Github Repo Link if directory doesn't exist
if [ ! -d "climbup_class_agent" ]; then
    echo "📥 Cloning GitHub Repository..."
    git clone https://github.com/Climbup-dev/climbup_class_agent.git
fi

cd climbup_class_agent/backend

echo "------------------------------------------------"
echo "⚠️ IMPORTANT: Please configure your environment."
echo "1. Run: cp .env.example .env (and fill in your Supabase, Groq, OpenRouter keys)"
echo "2. Edit Caddyfile: nano Caddyfile (replace YOUR_DOMAIN.COM with api.yourdomain.com)"
echo "3. Run: docker compose up -d --build"
echo "------------------------------------------------"
