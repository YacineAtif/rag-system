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
from typing import List, Dict, Any, Tuple

try:
    import yaml
except Exception:
    yaml = None

CONFIG = {}
if yaml and Path("config.yaml").exists():
    with open("config.yaml", "r") as f:
        CONFIG = yaml.safe_load(f) or {}
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

def chunk_text(text: str, chunk_size: int = 2000, overlap: int = 200) -> List[str]:
    """Split text into larger chunks while keeping bullet lists intact."""

    bullet_re = re.compile(r"^\s*(?:[-*]|\d+\.)\s+")
    lines = text.splitlines()
    segments: List[str] = []
    buffer: List[str] = []
    in_bullet = False

    for line in lines:
        if bullet_re.match(line):
            if not in_bullet and buffer:
                segments.append(" ".join(buffer).strip())
                buffer = []
            in_bullet = True
            buffer.append(line.strip())
        else:
            if in_bullet and buffer:
                segments.append("\n".join(buffer).strip())
                buffer = []
            in_bullet = False
            buffer.append(line.strip())

    if buffer:
        if in_bullet:
            segments.append("\n".join(buffer).strip())
        else:
            segments.append(" ".join(buffer).strip())

    chunks: List[str] = []
    current_chunk: List[str] = []
    current_tokens = 0

    for seg in segments:
        seg_tokens = len(seg.split())
        if current_tokens + seg_tokens > chunk_size and current_chunk:
            chunks.append("\n".join(current_chunk))
            if overlap > 0:
                tokens = "\n".join(current_chunk).split()
                overlap_tokens = tokens[-overlap:]
                current_chunk = [" ".join(overlap_tokens)]
                current_tokens = len(overlap_tokens)
            else:
                current_chunk = []
                current_tokens = 0
        current_chunk.append(seg)
        current_tokens += seg_tokens

    if current_chunk:
        chunks.append("\n".join(current_chunk))

    return chunks


def split_into_sections(text: str, patterns: List[str]) -> List[Tuple[str, str]]:
    """Split text into (section_name, text) using heading patterns."""
    if not patterns:
        return [("unknown", text)]

    lines = text.splitlines()
    sections: List[Tuple[str, List[str]]] = []
    current_name = "unknown"
    buffer: List[str] = []

    compiled = [re.compile(p) for p in patterns]

    for line in lines:
        matched = False
        stripped = line.strip()
        for pat in compiled:
            m = pat.match(stripped)
            if m:
                if buffer:
                    sections.append((current_name, buffer))
                    buffer = []
                current_name = m.group(1).strip().lower()
                matched = True
                break
        if not matched:
            buffer.append(line)

    if buffer:
        sections.append((current_name, buffer))

    return [(name, "\n".join(lines).strip()) for name, lines in sections]

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
    
    patterns = CONFIG.get("chunk_processing", {}).get("section_patterns", [])

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

                sections = split_into_sections(content, patterns)
                chunk_pairs = []
                for section_name, section_text in sections:
                    chs = chunk_text(
                        section_text,
                        chunk_size=CONFIG.get("chunk_size", 2000),
                        overlap=CONFIG.get("chunk_overlap", 200),
                    )
                    for ch in chs:
                        chunk_pairs.append((section_name, ch))

                print(f"   Split into {len(chunk_pairs)} chunks")

                for i, (sec_name, chunk) in enumerate(chunk_pairs):
                    chunk_metadata = base_metadata.copy()
                    chunk_metadata.update({
                        "chunk_id": i,
                        "total_chunks": len(chunk_pairs),
                        "content_type": "chunk",
                        "section_name": sec_name
                    })

                    documents.append(Document(content=chunk, meta=chunk_metadata))

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
