import os
import re
import time
import requests
from pathlib import Path
from typing import List
import subprocess

# Haystack v2 imports
from haystack import Document, Pipeline
from haystack.components.writers import DocumentWriter
from haystack.components.embedders import SentenceTransformersTextEmbedder, SentenceTransformersDocumentEmbedder
from haystack_integrations.document_stores.weaviate import WeaviateDocumentStore
from haystack_integrations.components.retrievers.weaviate import WeaviateEmbeddingRetriever

# --- Document Loading from domain_loader.py ---
def load_text_file(file_path: str) -> str:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='latin-1') as f:
            return f.read()

def load_pdf_file(file_path: str) -> str:
    try:
        import PyPDF2
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
    except ImportError:
        print("❌ PyPDF2 not installed. Install with: pip install PyPDF2")
        return ""
    except Exception as e:
        print(f"❌ Error reading PDF {file_path}: {e}")
        return ""

def load_docx_file(file_path: str) -> str:
    try:
        import docx
        doc = docx.Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except ImportError:
        print("❌ python-docx not installed. Install with: pip install python-docx")
        return ""
    except Exception as e:
        print(f"❌ Error reading DOCX {file_path}: {e}")
        return ""

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sent_length = len(sentence)
        if current_length + sent_length > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            overlap_count = max(1, int(len(current_chunk) * 0.3))
            current_chunk = current_chunk[-overlap_count:]
            current_length = sum(len(s) for s in current_chunk)
        
        current_chunk.append(sentence)
        current_length += sent_length
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

def load_documents_from_folder(folder_path: str) -> List[Document]:
    documents = []
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"❌ Folder {folder_path} does not exist")
        return documents
    
    print(f"📁 Loading documents from: {folder_path}")
    
    supported_extensions = {
        '.txt': load_text_file,
        '.md': load_text_file,
        '.pdf': load_pdf_file,
        '.docx': load_docx_file,
    }
    
    for file_path in folder.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            print(f"📄 Processing: {file_path.name}")
            
            try:
                loader_func = supported_extensions[file_path.suffix.lower()]
                content = loader_func(str(file_path))
                
                if not content.strip():
                    print(f"⚠️  Empty content in {file_path.name}")
                    continue
                
                file_stats = file_path.stat()
                base_metadata = {
                    "filename": file_path.name,
                    "file_path": str(file_path),
                    "file_type": file_path.suffix.lower(),
                    "file_size": file_stats.st_size,
                    "modified_date": time.ctime(file_stats.st_mtime)
                }
                
                chunks = chunk_text(content, chunk_size=500, overlap=100)
                print(f"   Split into {len(chunks)} chunks")
                
                for i, chunk in enumerate(chunks):
                    chunk_metadata = base_metadata.copy()
                    chunk_metadata.update({
                        "chunk_id": i,
                        "total_chunks": len(chunks),
                        "content_type": "chunk"
                    })
                    documents.append(Document(content=chunk, meta=chunk_metadata))
                    
            except Exception as e:
                print(f"❌ Error processing {file_path.name}: {e}")
                continue
    
    print(f"✅ Loaded {len(documents)} document chunks")
    return documents

# --- Text Cleaning from simple_qa.py ---
def deep_clean_text(text):
    if not text:
        return ""
    
    # COMMENTED OUT: Keep headers for better structure
    # text = re.sub(r'^#+\s.*?$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^=+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^-+\s*$', '', text, flags=re.MULTILINE)
    # COMMENTED OUT: Keep bold formatting like **Project Partners**
    # text = re.sub(r'\*+([^*]+)\*+', r'\1', text)
    text = re.sub(r'_+([^_]+)_+', r'\1', text)
    text = re.sub(r'`+([^`]+)`+', r'\1', text)
    # COMMENTED OUT: Keep links with organization names
    # text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    text = re.sub(r'http[s]?://\S+', '', text)
    text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\|+', ' ', text)
    # COMMENTED OUT: Keep markdown characters that provide structure
    # text = re.sub(r'[#*_`<>]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n+', ' ', text)
    
    return text.strip()

def extract_quality_sentences(text):
    cleaned_text = deep_clean_text(text)
    if not cleaned_text:
        return []
    
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', cleaned_text)
    quality_sentences = []
    
    for sentence in sentences:
        sentence = sentence.strip()
        
        if (len(sentence) < 25 or
            len(sentence) > 400 or
            len(sentence.split()) < 5 or
            sentence.lower().startswith(('what', 'how', 'when', 'where', 'why', 'is there', 'are there')) or
            any(artifact in sentence.lower() for artifact in ['comprehensive guide', 'table of contents', 'click here']) or
            sentence.count('(') != sentence.count(')') or
            any(artifact in sentence for artifact in ['###', '```', '---', '==='])):
            continue
        
        sentence = re.sub(r'\s+', ' ', sentence).strip()
        if not sentence.endswith(('.', '!', '?')):
            sentence += '.'
        
        quality_sentences.append(sentence)
    
    return quality_sentences

def create_natural_answer(sentences, query):
    if not sentences:
        return "I don't know."
    
    prefix = "Here's what I found: "
    if len(sentences) == 1:
        return f"{prefix}{sentences[0]}"
    elif len(sentences) == 2:
        return f"{prefix}{sentences[0]} Also, {sentences[1]}"
    else:
        para1 = sentences[0]
        para2 = " ".join(sentences[1:3])  # Limit to avoid too long answers
        return f"{prefix}{para1} Additionally, {para2}"

# --- Main Pipeline ---
def wait_for_weaviate(url="http://localhost:8080", max_retries=30):
    for i in range(max_retries):
        try:
            response = requests.get(f"{url}/v1/.well-known/ready", timeout=5)
            if response.status_code == 200:
                print("✅ Weaviate is ready!")
                return True
        except:
            pass
        print(f"⏳ Waiting for Weaviate... ({i+1}/{max_retries})")
        time.sleep(3)
    return False

def check_docker_containers():
    """Check if required Docker containers are running"""
    try:
        result = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}"], 
            capture_output=True, text=True, check=True
        )
        running_containers = result.stdout.strip().split('\n')
        
        weaviate_running = any('weaviate' in container for container in running_containers)
        transformers_running = any('transformers' in container for container in running_containers)
        
        return weaviate_running and transformers_running
    except subprocess.CalledProcessError:
        return False

def main():
    print("🤖 Domain-Restricted RAG System (Haystack v2)")
    print("=" * 70)

    # Start infrastructure if needed
    if check_docker_containers():
        print("✅ Docker containers already running, skipping startup")
    else:
        print("🚀 Starting Docker infrastructure...")
        try:
            subprocess.run(["docker-compose", "up", "-d"], check=True)
            print("✅ Docker containers started")
            time.sleep(30)  # Reduced wait time since containers are starting fresh
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to start Docker containers: {e}")
            return

    # Wait for Weaviate
    if not wait_for_weaviate():
        print("❌ Weaviate not responding. Check docker-compose logs.")
        return

    # Initialize Weaviate Document Store
    try:
        document_store = WeaviateDocumentStore(url="http://localhost:8080")
        print("✅ Connected to Weaviate")
    except Exception as e:
        print(f"❌ Failed to connect to Weaviate: {e}")
        return

    # Load documents
    documents_folder = "documents"
    documents = load_documents_from_folder(documents_folder)
    
    if not documents:
        print("❌ No documents found to index")
        return

    # Create embedder for documents
    doc_embedder = SentenceTransformersDocumentEmbedder(
        model="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Warm up the embedder
    print("🔥 Warming up document embedder...")
    doc_embedder.warm_up()

    # Embed documents
    print(f"🔄 Embedding {len(documents)} document chunks...")
    embedded_docs = doc_embedder.run(documents)["documents"]
    
    # Index documents
    print(f"🔄 Indexing {len(embedded_docs)} document chunks...")
    indexing_pipeline = Pipeline()
    indexing_pipeline.add_component("writer", DocumentWriter(document_store=document_store))
    indexing_pipeline.run({"writer": {"documents": embedded_docs}})
    print("✅ Documents indexed successfully!")

    # Create RAG pipeline
    print("🔍 Setting up RAG pipeline...")
    
    # Create components
    text_embedder = SentenceTransformersTextEmbedder(
        model="sentence-transformers/all-MiniLM-L6-v2"
    )
    # Warm up the text embedder
    print("🔥 Warming up text embedder...")
    text_embedder.warm_up()
    retriever = WeaviateEmbeddingRetriever(document_store=document_store)
    
    # Create pipeline
    rag_pipeline = Pipeline()
    rag_pipeline.add_component("text_embedder", text_embedder)
    rag_pipeline.add_component("retriever", retriever)
    
    # Connect components
    rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")

    # Interactive QA
    print("\n💬 Natural Language Q&A")
    print("Ask any question about your documents. Type 'quit' to exit.")
    
    conversation_history = []
    
    while True:
        try:
            query = input("\n❓ You: ").strip()
            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not query:
                continue
            
            print("\n🔍 Thinking...")
            start_time = time.time()
            
            # Run the pipeline
            result = rag_pipeline.run(
                {
                    "text_embedder": {"text": query},
                    "retriever": {"top_k": 5}
                }
            )
            
            # Process results
            documents = result["retriever"]["documents"]
            sentences = []
            sources = set()

            for doc in documents:
                extracted = extract_quality_sentences(doc.content)
                sentences.extend(extracted)
                sources.add(doc.meta.get('filename', 'Unknown'))

            # DEBUG: Show what sentences were extracted
            print(f"🔍 DEBUG: Found {len(sentences)} quality sentences")
            for i, sentence in enumerate(sentences[:3]):
                print(f"   {i+1}: {sentence[:100]}...")

            # Create answer - with improved fallback logic
            partner_keywords = ['university of skövde', 'scania', 'smart eye', 'viscando']
            is_partner_query = any(keyword in query.lower() for keyword in ['partner', 'collaborator', 'organization', 'company', 'consortium'])

            relevant_sentences = []
            if sentences and is_partner_query:
                # For partner queries, check if sentences contain actual partner names
                for sentence in sentences:
                    if any(keyword in sentence.lower() for keyword in partner_keywords):
                        relevant_sentences.append(sentence)

                # If no sentences contain partner names, fall back to raw content
                if not relevant_sentences:
                    print("🔄 No sentences with partner names found, using raw content...")
                    raw_content = []
                    for doc in documents[:3]:  # Use top 3 documents
                        content = doc.content
                        lines = content.split('\n')

                        for line in lines:
                            line = line.strip()
                            # Look for lines that contain partner names or list structures
                            if (any(name in line.lower() for name in partner_keywords) or
                                ('**' in line and any(word in line.lower() for word in ['partner', 'organization'])) or
                                line.startswith('*') and any(name in line.lower() for name in partner_keywords)):
                                raw_content.append(line)

                    if raw_content:
                        answer = "Here's what I found: " + " ".join(raw_content[:5])
                    else:
                        answer = "I don't know."
                else:
                    answer = create_natural_answer(relevant_sentences[:4], query)
            elif sentences:
                # For non-partner queries, use normal logic
                answer = create_natural_answer(sentences[:4], query)
            else:
                answer = "I don't know."

            # Add sources (rest remains the same)
            if sources and answer != "I don't know.":
                source_list = sorted([s for s in sources if s != 'Unknown'])
                if source_list:
                    answer += f"\n\n📚 Sources: {', '.join(source_list)}"
            
            # Update conversation history
            conversation_history.append({"query": query, "answer": answer})
            if len(conversation_history) > 5:
                conversation_history.pop(0)
            
            elapsed = time.time() - start_time
            print(f"\n💬 Answer (in {elapsed:.2f}s):")
            print("-" * 60)
            print(answer)
            print("-" * 60)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()