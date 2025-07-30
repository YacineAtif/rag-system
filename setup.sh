#!/bin/bash

# RAG System Setup Script
echo "🚀 Setting up RAG System Environment..."

# Check if virtual environment exists
if [ ! -d "rag-env" ]; then
    echo "📦 Creating virtual environment..."
    python -m venv rag-env
fi

# Activate virtual environment
echo "⚡ Activating virtual environment..."
source rag-env/bin/activate

# Upgrade pip and install dependencies
echo "🔧 Installing dependencies..."
pip install --upgrade pip wheel
pip install -r requirements.txt

# Check for .env file
if [ ! -f ".env" ]; then
    echo "🔑 Setting up environment variables..."
    echo "Please enter your OpenAI API key (it will be saved to .env):"
    read -s OPENAI_API_KEY
    echo "OPENAI_API_KEY=$OPENAI_API_KEY" > .env
    echo "✅ API key saved to .env file"
else
    echo "✅ .env file already exists"
fi

# Check Docker services
echo "🐳 Checking Docker services..."
if ! docker ps | grep -q weaviate; then
    echo "🔄 Starting Weaviate services..."
    docker-compose up -d
else
    echo "✅ Weaviate already running"
fi

# Wait for Weaviate to be ready
echo "⏳ Waiting for Weaviate to be ready..."
while ! curl -s http://localhost:8080/v1/.well-known/ready > /dev/null; do
    sleep 2
    echo "   Waiting..."
done
echo "✅ Weaviate is ready!"

# Run tests
echo "🧪 Running tests..."
python -m pytest -q

echo ""
echo "🎉 Setup complete! You can now run:"
echo "   python weaviate_rag_pipeline_transformers.py"
echo ""
echo "🔒 Your API key is safely stored in .env file"
echo "📁 Make sure to never commit .env to git!"