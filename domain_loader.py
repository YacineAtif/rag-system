#!/usr/bin/env python3
"""
Enhanced Domain Document Loader with Semantic Chunking
Supports: PDF, TXT, DOCX, MD files
"""
import os
import re
import time
import requests
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from haystack import Document, Pipeline
from haystack.components.writers import DocumentWriter
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack_integrations.document_stores.weaviate import WeaviateDocumentStore
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

def load_text_file(file_path: str) -> str:
    """Load content from a text file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='latin-1') as f:
            return f.read()

def load_pdf_file(file_path: str) -> str:
    """Load content from a PDF file"""
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
    """Load content from a DOCX file"""
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

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
    """Split text preserving sentence boundaries with semantic overlap"""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sent_length = len(sentence)
        if current_length + sent_length > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            # Keep overlapping sentences
            overlap_count = max(1, int(len(current_chunk) * 0.3))
            current_chunk = current_chunk[-overlap_count:]
            current_length = sum(len(s) for s in current_chunk)
        current_chunk.append(sentence)
        current_length += sent_length
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

def load_documents_from_folder(folder_path: str) -> List[Document]:
    """Load all supported documents from a folder"""
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
                
                # Always chunk documents for consistency
                chunks = chunk_text(content, chunk_size=800, overlap=150)
                print(f"   Split into {len(chunks)} chunks")
                
                for i, chunk in enumerate(chunks):
                    chunk_metadata = base_metadata.copy()
                    chunk_metadata.update({
                        "chunk_id": i,
                        "total_chunks": len(chunks),
                        "content_type": "chunk"
                    })
                    
                    documents.append(Document(
                        content=chunk,
                        meta=chunk_metadata
                    ))
                
            except Exception as e:
                print(f"❌ Error processing {file_path.name}: {e}")
                continue
    
    print(f"✅ Loaded {len(documents)} document chunks")
    return documents

def wait_for_weaviate(url="http://localhost:8080", max_retries=10):
    """Wait for Weaviate to be ready"""
    for i in range(max_retries):
        try:
            response = requests.get(f"{url}/v1/meta")
            if response.status_code == 200:
                print("✅ Weaviate is ready!")
                return True
        except:
            pass
        print(f"⏳ Waiting for Weaviate... ({i+1}/{max_retries})")
        time.sleep(2)
    return False

def setup_domain_knowledge_base(documents_folder: str = "documents"):
    """Set up the complete domain knowledge base"""
    print("🚀 Domain Knowledge Loader with Semantic Chunking")
    print("=" * 70)
    
    if not wait_for_weaviate():
        print("❌ Weaviate not responding. Make sure docker-compose is running.")
        return None, None
    
    documents = load_documents_from_folder(documents_folder)
    
    if not documents:
        print("❌ No documents found to index")
        return None, None
    
    try:
        print("🔧 Initializing Weaviate document store...")
        document_store = WeaviateDocumentStore(url="http://localhost:8080")
        
        print("📚 Setting up indexing pipeline...")
        indexing_pipeline = Pipeline()
        indexing_pipeline.add_component(
            "embedder", 
            SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
        )
        indexing_pipeline.add_component("writer", DocumentWriter(document_store=document_store))
        indexing_pipeline.connect("embedder", "writer")
        
        print(f"🔄 Indexing {len(documents)} document chunks...")
        indexing_pipeline.run({"embedder": {"documents": documents}})
        print("✅ Domain documents indexed successfully!")
        
        return document_store
        
    except Exception as e:
        print(f"❌ Error setting up knowledge base: {e}")
        return None

if __name__ == "__main__":
    if not os.path.exists("documents"):
        print("❌ 'documents' folder not found. Please create it and add your domain documents.")
    else:
        setup_domain_knowledge_base("documents")