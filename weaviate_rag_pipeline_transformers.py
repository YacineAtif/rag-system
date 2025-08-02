import os
import re
import time
import requests
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from backend.config import Config
from backend.qa_models import ClaudeQA
from backend.llm_generator import LLMGenerator
from neo4j import GraphDatabase
import subprocess

CONFIG = Config()

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
    
    patterns = CONFIG.chunk_processing.section_patterns

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
                    chs = chunk_text(section_text, chunk_size=500, overlap=100)
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

# --- Text Cleaning from simple_qa.py ---
class QueryClassifier:
    """Simple heuristic-based query classifier with config support."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or CONFIG
        pri_cfg = {}
        sp = getattr(self.config, "section_priorities", {})
        if hasattr(sp, "queries"):
            pri_cfg = sp.queries.get("partnership_queries", {})
        elif isinstance(sp, dict):
            pri_cfg = sp.get("partnership_queries", {})
        self.partnership_keywords = [kw.lower() for kw in pri_cfg.get("keywords", [])]

    def classify(self, query: str) -> str:
        q = query.lower().strip()

        if any(word in q for word in self.partnership_keywords):
            return "partnership"

        entity_keywords = [
            "who",
            "organization",
            "organizations",
            "company",
            "companies",
            "partner",
            "collaborator",
            "team",
            "group",
            "member",
            "participant",
            "contributor",
            "stakeholder",
            "department",
            "division",
            "institution",
        ]

        if any(word in q for word in entity_keywords):
            return "entity"

        if q.startswith("what is") or q.startswith("define") or "definition" in q:
            return "definition"


        procedural_keywords = ["how", "step", "procedure", "process"]
        if any(word in q for word in procedural_keywords):
            return "procedural"

        comparison_keywords = ["compare", "difference", " vs ", " versus "]
        if any(word in q for word in comparison_keywords):
            return "comparison"

        return "general"


class TextProcessor:
    """Utility class for advanced text cleaning and processing."""

    def clean_text(self, text: str, strategy: str = "balanced") -> str:
        if not text:
            return ""

        if strategy == "preserve_structure":
            text = re.sub(r"http[s]?://\S+", "", text)
            text = re.sub(r"\s+", " ", text)
            return text.strip()

        if strategy == "clean_moderate":
            text = re.sub(r"http[s]?://\S+", "", text)
            text = re.sub(r"[_*`]+", "", text)
            text = re.sub(r"\s+", " ", text)
            text = re.sub(r"\n+", " ", text)
            return text.strip()

        if strategy == "preserve_lists":
            text = re.sub(r"http[s]?://\S+", "", text)
            text = re.sub(r"[_*`]+", "", text)
            text = re.sub(r"\r\n", "\n", text)
            text = re.sub(r"\n{3,}", "\n\n", text)
            return text.strip()

        # balanced strategy
        text = re.sub(r"http[s]?://\S+", "", text)
        text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
        text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)
        text = re.sub(r"[#*_`<>|]", " ", text)
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\n+", " ", text)
        return text.strip()

    def improve_sentence_boundary_detection(self, text: str) -> List[str]:
        text = text.strip()
        if not text:
            return []
        return re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)

    def extract_entities(self, text: str) -> List[str]:
        cleaned = self.clean_text(text, strategy="preserve_structure")
        bullets = re.findall(r"^[\-*+]\s*(.+)", cleaned, flags=re.MULTILINE)
        headings = re.findall(r"^#+\s*(.+)", cleaned, flags=re.MULTILINE)
        capitalized = re.findall(r"\b([A-Z][A-Za-z0-9&]*(?:\s+[A-Z][A-Za-z0-9&]*){0,3})", cleaned)
        names = bullets + headings + capitalized

        seen = set()
        unique = []
        for name in names:
            n = name.strip()
            if n:
                l = n.lower()
                if l not in seen:
                    seen.add(l)
                    unique.append(n)
        return unique

    def preserve_context_formatting(self, text: str) -> str:
        text = re.sub(r"\r\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def extract_quality_sentences(self, text: str, strategy: str = "balanced") -> List[str]:
        cleaned_text = self.clean_text(text, strategy=strategy)
        if not cleaned_text:
            return []

        sentences = self.improve_sentence_boundary_detection(cleaned_text)
        quality_sentences = []

        for sentence in sentences:
            sentence = sentence.strip()

            if (
                len(sentence) < 25
                or len(sentence) > 400
                or len(sentence.split()) < 5
                or sentence.lower().startswith(
                    (
                        "what",
                        "how",
                        "when",
                        "where",
                        "why",
                        "is there",
                        "are there",
                    )
                )
                or any(
                    artifact in sentence.lower()
                    for artifact in ["comprehensive guide", "table of contents", "click here"]
                )
                or sentence.count("(") != sentence.count(")")
                or any(artifact in sentence for artifact in ["###", "```", "---", "==="])
            ):
                continue

            sentence = re.sub(r"\s+", " ", sentence).strip()
            if not sentence.endswith((".", "!", "?")):
                sentence += "."

            quality_sentences.append(sentence)

        return quality_sentences


class AnswerGenerator:
    """Generate answers using Claude Haiku."""

    def __init__(self, processor: TextProcessor, config: Optional[Config] = None):
        self.processor = processor
        self.config = config or Config()

    def _score(self, sentence: str, query: str) -> float:
        q_words = set(query.lower().split())
        s_words = set(sentence.lower().split())
        if not q_words:
            return 0.0
        return len(q_words & s_words) / len(q_words)

    def _select_sentences(self, sentences: List[str], query: str, limit: int = 6) -> List[str]:
        scored = [(self._score(s, query), s) for s in sentences]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in scored[:limit] if s]

    def generate(self, query: str, sentences: List[str], query_type: str, history: List[Dict]) -> str:
        if not sentences:
            if history:
                return "I'm not sure. Previously we discussed: " + history[-1]["answer"]
            return "I don't know."

        top_sentences = self._select_sentences(sentences, query)

        instruction = None
        try:
            instruction = self.config.prompting.context_instructions.get(query_type)
        except Exception:
            instruction = None


        claude = ClaudeQA(self.config)
        res = claude.generate(query, top_sentences[:8], instruction=instruction)
        if res.get("answer"):
            return res["answer"]

        if query_type in {"entity", "procedural", "comparison", "general"}:
            if query_type == "entity":
                entities: List[str] = []
                for s in top_sentences:
                    entities.extend(self.processor.extract_entities(s))
                entities = list(dict.fromkeys(entities))
                if entities:
                    return "\n".join([
                        "Entities mentioned:",
                        "- " + "\n- ".join(entities),
                    ])

            if query_type == "procedural":
                steps = [f"{i+1}. {self.processor.clean_text(s, 'preserve_lists')}" for i, s in enumerate(top_sentences)]
                return "Here are the steps:\n" + "\n".join(steps)

            if query_type == "comparison":
                return "Comparison:\n" + "\n".join(f"- {s}" for s in top_sentences[:4])

            if not res.get("answer"):
                res = claude.generate(query, top_sentences[:8], instruction=instruction)
                if res.get("answer"):
                    return res["answer"]


        claude = ClaudeQA(self.config)
        res = claude.generate(query, top_sentences[:8], instruction=instruction)
        return res.get("answer", "")


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


class Neo4jGraphBuilder:
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="password"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.claude_extractor = LLMGenerator(model="claude-3-5-haiku-20241022")

    def extract_entities_and_relationships(self, text_chunk, source_doc):
        """Use Claude to extract entities and relationships from text"""
        extraction_prompt = """
        Extract entities and relationships from this text. Return JSON format:
        {{
            "entities": [
                {{"name": "Entity Name", "type": "Person|Organization|Technology|Concept", "properties": {{}}}}
            ],
            "relationships": [
                {{"source": "Entity1", "target": "Entity2", "relationship": "WORKS_WITH|DEVELOPS|USES", "properties": {{}}}}
            ]
        }}

        Text: {text}
        """

        # Extract using Claude
        result = self.claude_extractor.generate(
            query=extraction_prompt.format(text=text_chunk),
            context_sentences=[],
            system_prompt="You are an expert at extracting structured knowledge from text."
        )

        # Parse JSON response
        try:
            return json.loads(result)
        except:
            return {"entities": [], "relationships": []}

    def populate_graph(self, documents):
        """Phase 1: Build knowledge graph from documents"""
        with self.driver.session() as session:
            for doc in documents:
                extracted = self.extract_entities_and_relationships(doc.content, doc.meta.get("source", ""))

                # Create entities
                for entity in extracted["entities"]:
                    session.run(
                        "MERGE (e:Entity {name: $name, type: $type, source: $source})",
                        name=entity["name"],
                        type=entity["type"],
                        source=doc.meta.get("source", "")
                    )

                # Create relationships
                for rel in extracted["relationships"]:
                    session.run(
                        """
                        MATCH (a:Entity {name: $source}), (b:Entity {name: $target})
                        MERGE (a)-[r:RELATES {type: $rel_type, source: $doc_source}]->(b)
                        """,
                        source=rel["source"],
                        target=rel["target"],
                        rel_type=rel["relationship"],
                        doc_source=doc.meta.get("source", "")
                    )

    def query_graph(self, query_text):
        """Phase 2: Query knowledge graph for relationships"""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (e1:Entity)-[r]->(e2:Entity)
                WHERE e1.name CONTAINS $query OR e2.name CONTAINS $query
                RETURN e1.name as source, r.type as relationship, e2.name as target
                LIMIT 10
                """,
                query=query_text
            )
            return [
                {"source": record["source"], "relationship": record["relationship"], "target": record["target"]}
                for record in result
            ]


class HybridQueryRouter:
    def __init__(self, graph_builder, vector_retriever):
        self.graph_builder = graph_builder
        self.vector_retriever = vector_retriever
        self.claude = LLMGenerator()

    def should_use_graph(self, query):
        """Determine if query needs graph traversal"""
        graph_keywords = ["partner", "collaborate", "relationship", "connect", "work with", "develop", "contribute"]
        return any(keyword in query.lower() for keyword in graph_keywords)

    def hybrid_retrieve(self, query):
        """Route query to appropriate retrieval method"""
        results = {"vector_results": [], "graph_results": []}

        # Always get vector results
        # Generate query embedding first
        query_embedding = self.text_embedder.run(text=query)["embedding"]
        vector_docs = self.vector_retriever.run(query_embedding=query_embedding)
        results["vector_results"] = vector_docs.get("documents", [])

        # Get graph results if needed
        if self.should_use_graph(query):
            graph_results = self.graph_builder.query_graph(query)
            results["graph_results"] = graph_results

        return results

    def synthesize_answer(self, query, hybrid_results):
        """Combine vector and graph results for final answer"""
        context_parts = []

        # Add vector context
        for doc in hybrid_results["vector_results"][:5]:
            context_parts.append(doc.content)

        # Add graph context
        if hybrid_results["graph_results"]:
            graph_context = "Relationships found: "
            for rel in hybrid_results["graph_results"]:
                graph_context += f"{rel['source']} {rel['relationship']} {rel['target']}. "
            context_parts.append(graph_context)

        # Generate answer using Claude
        return self.claude.generate(
            query=query,
            context_sentences=context_parts,
            system_prompt="Combine information from documents and relationship data to provide comprehensive answers."
        )


def boost_documents(documents: List[Document], query_type: str) -> List[Document]:
    """Apply section-based score boosts to retrieved documents."""
    retrieval_cfg = getattr(CONFIG, "retrieval", {})
    if hasattr(retrieval_cfg, "section_name_boost"):
        base_boost = retrieval_cfg.section_name_boost
    elif isinstance(retrieval_cfg, dict):
        base_boost = retrieval_cfg.get("section_name_boost", 1.0)
    else:
        base_boost = 1.0

    sp = getattr(CONFIG, "section_priorities", {})
    if hasattr(sp, "queries"):
        priority_cfg = sp.queries.get(f"{query_type}_queries", {})
    elif isinstance(sp, dict):
        priority_cfg = sp.get(f"{query_type}_queries", {})
    else:
        priority_cfg = {}
    priority_sections = [s.lower() for s in priority_cfg.get("priority_sections", [])]
    section_factor = priority_cfg.get("boost_factor", 1.0)

    for doc in documents:
        score = doc.score or 0.0
        section = ((doc.meta or {}).get("section_name") or "").lower()
        if section:
            score *= base_boost
            if section in priority_sections:
                score *= section_factor
        doc.score = score

    documents.sort(key=lambda d: d.score or 0.0, reverse=True)
    return documents

class RAGPipeline:
    def __init__(self, document_store, retriever):
        # Existing components
        self.document_store = document_store
        self.retriever = retriever

        # Add Neo4j integration
        self.graph_builder = Neo4jGraphBuilder()
        self.hybrid_router = HybridQueryRouter(self.graph_builder, self.retriever)
        self.graph_populated = False

    def populate_knowledge_graph(self):
        """Phase 1: Build knowledge graph (run once after indexing)"""
        if not self.graph_populated:
            print("🔄 Phase 1: Building Knowledge Graph...")
            print("🤖 Extracting entities and relationships with Claude...")

            # Get all documents from Weaviate
            all_docs = self.document_store.filter_documents()
            self.graph_builder.populate_graph(all_docs)

            print("📊 Knowledge graph populated!")
            self.graph_populated = True

    def query_with_graph(self, query):
        """Phase 2: Hybrid query using both vector search and graph traversal"""
        print("🔍 Phase 2: Hybrid retrieval (Vector + Graph)...")

        # Route query through hybrid system
        hybrid_results = self.hybrid_router.hybrid_retrieve(query)

        # Synthesize final answer
        answer = self.hybrid_router.synthesize_answer(query, hybrid_results)

        return {
            "answer": answer,
            "vector_results": len(hybrid_results["vector_results"]),
            "graph_results": len(hybrid_results["graph_results"])
        }


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

    # Create retriever for hybrid queries
    text_embedder = SentenceTransformersTextEmbedder(
        model="sentence-transformers/all-MiniLM-L6-v2",
    )
    print("🔥 Warming up text embedder...")
    text_embedder.warm_up()
    retriever = WeaviateEmbeddingRetriever(document_store=document_store)

    # Initialize RAG pipeline with Neo4j integration
    pipeline = RAGPipeline(document_store, retriever)

    # PHASE 1: Build Knowledge Graph (one-time after indexing)
    pipeline.populate_knowledge_graph()

    # PHASE 2: Interactive Q&A with hybrid retrieval
    print("\n💬 Hybrid Q&A (Vector + Knowledge Graph)")
    print("Ask questions about partnerships, collaborations, or general content.")

    while True:
        query = input("\n❓ You: ").strip()
        if query.lower() == 'quit':
            break

        start_time = time.time()
        result = pipeline.query_with_graph(query)
        elapsed = time.time() - start_time

        print(f"\n💬 Answer (in {elapsed:.2f}s):")
        print("-" * 60)
        print(result["answer"])
        print(f"\n📊 Retrieved: {result['vector_results']} vector + {result['graph_results']} graph results")
        print("-" * 60)


if __name__ == "__main__":
    main()