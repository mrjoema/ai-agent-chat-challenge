#!/bin/bash

# Thoughtful AI Customer Support Agent Startup Script

echo "🤖 Starting Thoughtful AI Customer Support Agent..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Install dependencies if needed
if [ ! -f "venv/pyvenv.cfg" ] || ! pip list | grep -q streamlit; then
    echo "📥 Installing dependencies..."
    python3 -m pip install -r requirements.txt
fi

# Check if .env file exists, if not copy from example
if [ ! -f ".env" ]; then
    echo "⚙️  Creating .env file from template..."
    cp .env.example .env
    echo "📝 Please edit .env file to add your OpenAI API key (optional)"
fi

# Run the application
echo "🚀 Starting Streamlit application..."
echo "📱 Open your browser to: http://localhost:8501"
echo "⏹️  Press Ctrl+C to stop the application"
echo ""

streamlit run app.py