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

# Set environment variables to suppress extraneous output
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_PROGRESS_BARS=true
export TF_CPP_MIN_LOG_LEVEL=2

# Upgrade pip and install dependencies
echo "🔧 Installing dependencies..."
pip install --upgrade pip wheel
pip install -r requirements.txt

# Download NLTK data
echo "📚 Downloading NLTK data..."
python -c "
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('maxent_ne_chunker_tab')
nltk.download('punkt_tab')
print('✅ NLTK data downloaded')
"

# Check for .env file
if [ ! -f ".env" ]; then
    echo "🔑 Setting up environment variables..."
    echo "Please enter your Anthropic Claude API key:"
    read -s ANTHROPIC_API_KEY
    echo "ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY" > .env
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
echo "   python -c 'from processing.comprehensive_extractor import ComprehensiveExtractor; print(\"Ready!\")'"
echo ""
echo "🔒 Your API key is safely stored in .env file"
echo "📁 Make sure to never commit .env to git!"
