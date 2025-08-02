# Domain-Agnostic RAG System

A flexible Retrieval-Augmented Generation (RAG) system built with Haystack v2 and Weaviate for querying domain-specific documents through natural language.

## 🚀 Features

- **Domain-Agnostic Design**: Works with any document collection
- **Multi-Format Support**: Processes PDF, DOCX, TXT, and Markdown files
- **Intelligent Chunking**: Smart text segmentation with overlap
- **Semantic Search**: Vector-based document retrieval using sentence-transformers
- **Natural Language Q&A**: Interactive question-answering interface
- **Docker Integration**: Containerized Weaviate vector database
- **Real-time Processing**: Fast document embedding and retrieval

## 🏗️ Architecture

```
Documents → Chunking → Embeddings → Weaviate → Retrieval → Answer Generation
```

### Core Components

- **Document Loader**: Supports multiple file formats with encoding detection
- **Text Chunker**: Sentence-based chunking with configurable overlap
- **Embedder**: SentenceTransformers for semantic representations
- **Vector Store**: Weaviate for scalable vector search
- **QA Pipeline**: Haystack v2 pipeline for end-to-end processing

## 📋 Prerequisites

- Python 3.9+
- Docker and Docker Compose
- 4GB+ RAM (for model loading)

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd rag-system
```

### 2. Create Virtual Environment
```bash
# Using conda (recommended)
conda create -n rag-env python=3.9
conda activate rag-env

# Or using venv
python -m venv rag-env
source rag-env/bin/activate  # On Windows: rag-env\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Start Weaviate (Vector Database)
```bash
docker-compose up -d
```

Wait for Weaviate to be ready:
```bash
curl http://localhost:8080/v1/.well-known/ready
```

## 📚 Usage

### 1. Prepare Your Documents

Create a `documents` folder and add your files:
```bash
mkdir documents
# Add your PDF, DOCX, TXT, or MD files to this folder
```

### 2. Run the RAG System
```bash
python weaviate_rag_pipeline_transformers.py
```

### 3. Interactive Q&A
Once the system loads, ask questions about your documents:
```
❓ You: What is the main topic of the documents?
❓ You: Who are the key stakeholders mentioned?
❓ You: Explain the methodology described
```

Type `quit` to exit.

## ⚙️ Configuration

### Chunk Size and Overlap
Modify chunking parameters in `weaviate_rag_pipeline_transformers.py`:
```python
chunks = chunk_text(content, chunk_size=500, overlap=100)
```

### Retrieval Settings
Adjust the number of retrieved documents:
```python
"retriever": {"top_k": 5}  # Retrieve top 5 relevant chunks
```

### Embedding Model
Change the sentence transformer model:
```python
model="sentence-transformers/all-MiniLM-L6-v2"  # Fast, good quality
# model="sentence-transformers/all-mpnet-base-v2"  # Slower, higher quality
```

### Neo4j Connection
Specify Neo4j connection settings in `config.yaml`:
```yaml
neo4j:
  uri: bolt://localhost:7687
  user: neo4j
  password: password
```

## 🐳 Docker Configuration

The system uses Docker Compose for Weaviate setup. Key services:

- **Weaviate**: Vector database (port 8080)
- **Transformers**: Sentence transformer inference service

### Docker Compose Services
```yaml
services:
  weaviate:
    image: cr.weaviate.io/semitechnologies/weaviate:1.31.2
    ports:
      - "8080:8080"
  
  t2v-transformers:
    image: cr.weaviate.io/semitechnologies/transformers-inference:sentence-transformers-all-MiniLM-L6-v2
```

## 📁 Project Structure

```
rag-system/
├── documents/                     # Your document collection
├── weaviate_rag_pipeline_transformers.py  # Main RAG pipeline
├── docker-compose.yml             # Weaviate configuration
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git ignore rules
├── README.md                      # This file
└── debug_chunks.py               # Debug utilities (optional)
```

## 🔧 Troubleshooting

### Common Issues

**1. Import Errors**
```bash
# Ensure correct Haystack version
pip uninstall haystack-ai -y
pip install haystack-ai==2.15.2
```

**2. Weaviate Connection Issues**
```bash
# Check if Weaviate is running
docker-compose ps
curl http://localhost:8080/v1/.well-known/ready
```

**3. Memory Issues**
- Reduce batch size in embedder
- Use smaller transformer models
- Increase Docker memory allocation

**4. Empty Responses**
- Check document chunking with `debug_chunks.py`
- Verify text cleaning isn't too aggressive
- Increase `top_k` retrieval parameter

### Performance Optimization

**For Large Document Collections:**
- Increase chunk size (500 → 800 characters)
- Use more powerful embedding models
- Configure Weaviate persistence

**For Faster Responses:**
- Use smaller embedding models
- Reduce `top_k` retrieval
- Implement embedding caching

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔗 Links

- [Haystack Documentation](https://docs.haystack.deepset.ai/)
- [Weaviate Documentation](https://weaviate.io/developers/weaviate)
- [Sentence Transformers](https://www.sbert.net/)

## 📞 Support

For questions and support:
- Create an issue in this repository
- Check the troubleshooting section above
- Review Haystack documentation for advanced configurations