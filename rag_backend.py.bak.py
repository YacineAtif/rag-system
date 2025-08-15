"""Shared backend logic for RAG system.

This module exposes a `RAGBackend` class that wraps the existing
`RAGPipeline` from `weaviate_rag_pipeline_transformers` so it can be
reused by both the command line interface and the new web application.

The implementation mirrors the setup performed in the CLI's `main`
function but packages it in a class that can be instantiated and queried
programmatically.
"""
from __future__ import annotations

import json
import time
import subprocess
import os
import re
from typing import Optional, List, Dict, Tuple
from neo4j import GraphDatabase

import yaml

from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack_integrations.document_stores.weaviate import WeaviateDocumentStore
from haystack_integrations.components.retrievers.weaviate import (
    WeaviateEmbeddingRetriever,
)

# Import the updated Config class
from backend.config import Config

# Reuse the existing pipeline and utilities from the CLI module.
from weaviate_rag_pipeline_transformers import (
    RAGPipeline,
    check_docker_containers,
    setup_logging,
    wait_for_weaviate,
)


class I2ConnectOODDetector:
    """Enhanced OOD detection specifically for I2Connect domain"""
    
    def __init__(self):
        # I2Connect domain indicators with weights
        self.domain_indicators = {
            # Core I2Connect concepts - HIGH WEIGHT
            'i2connect': 3.0,
            'evidence theory': 3.0,
            'dempster-shafer': 3.0,
            'belief function': 2.5,
            'mass function': 2.5,
            'plausibility': 2.5,
            
            # Traffic safety - HIGH WEIGHT
            'traffic safety': 2.5,
            'intersection': 2.0,
            'collision': 2.0,
            'driver monitoring': 2.0,
            'driver behavior': 2.0,
            'gaze tracking': 2.0,
            
            # Technology partners - MEDIUM WEIGHT
            'smart eye': 2.0,
            'viscando': 2.0,
            'scania': 2.0,
            'adas': 2.0,
            'hmi': 1.5,
            
            # Organizations - MEDIUM WEIGHT
            'university of skövde': 2.0,
            'skövde': 1.5,
            'consortium': 1.5,
            
            # Safety concepts - MEDIUM WEIGHT
            'safety concept': 2.0,
            'concept 1': 1.5,
            'concept 2': 1.5,
            'actor-focused': 1.5,
            'risk assessment': 2.0,
            'hazard detection': 1.5,
            
            # Technical terms - LOW-MEDIUM WEIGHT
            'sensor fusion': 1.5,
            'uncertainty quantification': 1.5,
            'probability': 1.0,
            'uncertainty': 1.0,
            'monitoring system': 1.0,
            'vehicle': 1.0,
            'automotive': 1.0
        }
        
        # Strong OOD indicators (topics clearly outside I2Connect)
        self.ood_indicators = [
            'cooking', 'recipe', 'food', 'restaurant', 'cuisine',
            'weather', 'climate', 'temperature', 'rain', 'snow',
            'sports', 'football', 'basketball', 'soccer', 'tennis',
            'entertainment', 'movie', 'film', 'music', 'concert',
            'celebrity', 'actor', 'actress', 'singer',
            'politics', 'election', 'government', 'president',
            'fashion', 'clothing', 'style', 'beauty',
            'finance', 'stock', 'investment', 'banking',
            'health', 'medicine', 'doctor', 'hospital',
            'travel', 'vacation', 'hotel', 'tourism'
        ]
    
    def is_context_relevant(self, context_sentences: List[str], 
                          min_score_threshold: float = 0.8,
                          max_ood_ratio: float = 0.3) -> Tuple[bool, Dict]:
        """
        Check if retrieved context is I2Connect relevant
        
        Args:
            context_sentences: List of retrieved context sentences
            min_score_threshold: Minimum relevance score (lowered for I2Connect)
            max_ood_ratio: Maximum ratio of OOD indicators allowed
            
        Returns:
            (is_relevant, diagnostic_info)
        """
        if not context_sentences:
            return False, {"reason": "empty_context", "score": 0.0}
        
        # Combine and preprocess context
        context_text = " ".join(context_sentences).lower()
        context_text = re.sub(r'[^\w\s]', ' ', context_text)
        
        # Calculate relevance score
        relevance_score = self._calculate_relevance_score(context_text)
        
        # Check for OOD indicators
        ood_count = sum(1 for indicator in self.ood_indicators 
                       if indicator in context_text)
        total_indicators = len(self.ood_indicators)
        ood_ratio = ood_count / total_indicators if total_indicators > 0 else 0
        
        # Check context length and quality
        word_count = len(context_text.split())
        
        # Decision logic - more permissive for I2Connect
        is_relevant = (
            relevance_score >= min_score_threshold and
            ood_ratio <= max_ood_ratio and
            word_count >= 5  # Very low threshold for minimum content
        )
        
        # Diagnostic information
        diagnostic_info = {
            "relevance_score": relevance_score,
            "ood_ratio": ood_ratio,
            "word_count": word_count,
            "matched_indicators": self._get_matched_indicators(context_text),
            "ood_matches": [ind for ind in self.ood_indicators if ind in context_text],
            "decision_factors": {
                "score_pass": relevance_score >= min_score_threshold,
                "ood_pass": ood_ratio <= max_ood_ratio,
                "length_pass": word_count >= 5
            }
        }
        
        return is_relevant, diagnostic_info
    
    def _calculate_relevance_score(self, context_text: str) -> float:
        """Calculate weighted relevance score based on I2Connect indicators"""
        total_score = 0.0
        
        for indicator, weight in self.domain_indicators.items():
            count = context_text.count(indicator)
            if count > 0:
                # Diminishing returns for multiple occurrences
                score_contribution = weight * min(count, 3) * (1 / (1 + count * 0.1))
                total_score += score_contribution
        
        return total_score
    
    def _get_matched_indicators(self, context_text: str) -> List[Tuple[str, int, float]]:
        """Get list of matched indicators with their counts and contributions"""
        matches = []
        for indicator, weight in self.domain_indicators.items():
            count = context_text.count(indicator)
            if count > 0:
                contribution = weight * min(count, 3) * (1 / (1 + count * 0.1))
                matches.append((indicator, count, contribution))
        
        return sorted(matches, key=lambda x: x[2], reverse=True)
    
    def analyze_context_quality(self, context_sentences: List[str]) -> Dict:
        """Detailed analysis for debugging"""
        is_relevant, diagnostics = self.is_context_relevant(context_sentences)
        
        return {
            "is_relevant": is_relevant,
            "overall_assessment": "IN_DOMAIN" if is_relevant else "OUT_OF_DOMAIN",
            "diagnostics": diagnostics,
            "top_matches": diagnostics["matched_indicators"][:5] if diagnostics["matched_indicators"] else [],
            "recommendations": self._generate_recommendations(diagnostics)
        }
    
    def _generate_recommendations(self, diagnostics: Dict) -> List[str]:
        """Generate recommendations based on analysis"""
        recommendations = []
        
        if diagnostics["relevance_score"] < 0.5:
            recommendations.append("Very low relevance score - consider improving retrieval strategy")
        elif diagnostics["relevance_score"] < 1.0:
            recommendations.append("Low relevance score - context may need enhancement")
        
        if diagnostics["ood_ratio"] > 0.2:
            recommendations.append(f"High OOD ratio ({diagnostics['ood_ratio']:.2f}) - context contains irrelevant content")
        
        if diagnostics["word_count"] < 10:
            recommendations.append("Very short context - may need more comprehensive retrieval")
        
        return recommendations


class RAGBackend:
    """Backend helper that wraps the heavy RAG pipeline."""

    def __init__(self, config_path: str = "config.yaml", force_rebuild: bool = False):
        # Initialize OOD detector
        self.ood_detector = I2ConnectOODDetector()
        
        # Load configuration using the updated Config class
        self.config = Config(config_path)
        print(f"🔧 RAGBackend using environment: {self.config.environment}")
        
        # Set up logging using the raw YAML for setup_logging function compatibility
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        setup_logging(config_dict)

        self._ensure_infrastructure()

        # Get environment-aware Weaviate URL
        weaviate_url = self._get_current_weaviate_url()
        print(f"🔗 Connecting to Weaviate at: {weaviate_url}")

        if not wait_for_weaviate(url=weaviate_url):
            raise RuntimeError(f"Weaviate not responding at {weaviate_url}. Check configuration.")

        # Build Haystack components with environment-aware URLs
        self.document_store = WeaviateDocumentStore(url=weaviate_url)
        self.text_embedder = SentenceTransformersTextEmbedder(
            model="sentence-transformers/all-MiniLM-L6-v2",
            progress_bar=False,
            encode_kwargs={"convert_to_tensor": False},
        )
        print("🔥 Warming up text embedder...")
        self.text_embedder.warm_up()
        retriever = WeaviateEmbeddingRetriever(document_store=self.document_store)

        # Initialize pipeline with the updated Config object
        self.pipeline = RAGPipeline(self.document_store, retriever, self.text_embedder)
        
        # CRITICAL FIX: Make sure the pipeline uses our environment-aware config
        self.pipeline.graph_builder.config = self.config
        
        # Update the global CONFIG to match our environment
        global CONFIG
        from weaviate_rag_pipeline_transformers import CONFIG as GLOBAL_CONFIG
        GLOBAL_CONFIG.environment = self.config.environment
        GLOBAL_CONFIG.neo4j = self.config.neo4j
        
        print(f"🔗 Neo4j connection will use: {self._get_current_neo4j_uri()}")

        # Check processing mode
        use_existing_data = os.getenv("USE_EXISTING_DATA", "false").lower() == "true"
        skip_processing = os.getenv("SKIP_DOCUMENT_PROCESSING", "false").lower() == "true"

        if use_existing_data:
            print("📊 Using existing data (USE_EXISTING_DATA=true)")
        elif skip_processing:
            print("⏭️ Skipping document processing (SKIP_DOCUMENT_PROCESSING=true)")
        else:
            print("📄 Processing documents from scratch...")
            self.pipeline.process_documents_intelligently()
            if force_rebuild:
                self.pipeline.force_rebuild_graph()
            else:
                self.pipeline.populate_knowledge_graph()

    def _get_current_weaviate_url(self) -> str:
        """Get the Weaviate URL based on current environment."""
        env = self.config.environment
        
        if hasattr(self.config, 'weaviate_environments') and env in self.config.weaviate_environments:
            return self.config.weaviate_environments[env]["url"]
        
        return self.config.weaviate.url

    def _get_current_neo4j_uri(self) -> str:
        """Get the Neo4j URI that will be used based on current environment."""
        env = self.config.environment
        
        if env == 'railway_weaviate_test':
            if hasattr(self.config.neo4j, 'railway_weaviate_test') and self.config.neo4j.railway_weaviate_test:
                uri = self.config.neo4j.railway_weaviate_test.uri
                print(f"🧪 Railway Weaviate Test - Using Railway Neo4j: {uri}")
                return uri
            else:
                uri = self.config.neo4j.local.uri
                print(f"🧪 Railway Weaviate Test - Using Local Neo4j (fallback): {uri}")
                return uri
        elif env == 'railway' and hasattr(self.config.neo4j, 'railway') and self.config.neo4j.railway:
            uri = self.config.neo4j.railway.uri
            print(f"🚀 Using Railway Neo4j: {uri}")
            return uri
        elif env == 'local' and hasattr(self.config.neo4j, 'local') and self.config.neo4j.local:
            uri = self.config.neo4j.local.uri
            print(f"🏠 Using Local Neo4j: {uri}")
            return uri
        elif env == 'aura' and hasattr(self.config.neo4j, 'aura') and self.config.neo4j.aura:
            uri = self.config.neo4j.aura.uri
            print(f"☁️ Using Aura Neo4j: {uri}")
            return uri
        elif env == 'production' and hasattr(self.config.neo4j, 'production') and self.config.neo4j.production:
            uri = self.config.neo4j.production.uri
            print(f"🏭 Using Production Neo4j: {uri}")
            return uri
        else:
            uri = self.config.neo4j.uri
            return uri

    def _ensure_infrastructure(self) -> None:
        """Ensure required infrastructure is available."""
        environment = self.config.environment
        
        if environment in ["railway", "aura", "production", "railway_weaviate_test"]:
            print(f"🌍 Running in {environment} environment - infrastructure managed externally")
            return
        
        if environment == "local":
            if check_docker_containers():
                print("✅ Docker containers already running")
                return
            
            print("🚀 Starting Docker infrastructure...")
            try:
                subprocess.run(["docker-compose", "up", "-d"], check=True)
                print("✅ Docker containers started")
                time.sleep(30)
            except subprocess.CalledProcessError as exc:
                raise RuntimeError("Failed to start Docker containers") from exc
        else:
            print(f"⚠️ Unknown environment '{environment}' - assuming infrastructure is available")

    def _format_answer(self, text: str) -> str:
        """Normalize whitespace and preserve paragraph breaks."""
        lines = [line.rstrip() for line in text.splitlines()]
        return "\n".join(line for line in lines if line)
    
    def _is_query_domain_relevant(self, query: str) -> bool:
        """Check if the query itself is about I2Connect domain topics"""
        query_lower = query.lower().strip()
        
        # I2Connect domain keywords - more comprehensive list
        domain_keywords = [
            # Core project terms
            'i2connect', 'evidence theory', 'dempster-shafer', 'belief function',
            'mass function', 'plausibility',
            
            # Traffic safety
            'traffic', 'safety', 'intersection', 'collision', 'driver', 'vehicle',
            'autonomous', 'adas', 'monitoring', 'gaze', 'tracking',
            
            # Technology partners
            'smart eye', 'viscando', 'scania', 'university', 'skövde',
            
            # Risk and assessment
            'risk', 'assessment', 'hazard', 'safety concept', 'uncertainty',
            'probability', 'sensor', 'fusion',
            
            # Research terms
            'research', 'project', 'consortium', 'partner', 'organization',
            
            # Technical architecture terms (when in research context)
            'architecture', 'system', 'data', 'framework', 'design',
            'implementation', 'methodology', 'approach', 'technology'
        ]
        
        # Check for direct keyword matches
        if any(keyword in query_lower for keyword in domain_keywords):
            return True
        
        # Check for question patterns about domain topics (more permissive)
        domain_question_patterns = [
            ('what is', ['evidence', 'risk', 'safety', 'dempster', 'belief', 'system', 'architecture', 'data']),
            ('how does', ['evidence', 'theory', 'risk', 'safety', 'system', 'architecture', 'work']),
            ('how is', ['risk', 'assessment', 'data', 'system']),
            ('tell me about', ['i2connect', 'project', 'research', 'safety', 'system', 'architecture']),
            ('who are', ['partners', 'consortium', 'organizations', 'contributors']),
            ('what are', ['safety concepts', 'risks', 'technologies', 'components']),
            ('describe', ['system', 'architecture', 'approach', 'methodology']),
            ('explain', ['system', 'architecture', 'approach', 'theory'])
        ]
        
        for pattern, terms in domain_question_patterns:
            if pattern in query_lower:
                if any(term in query_lower for term in terms):
                    return True
        
        # If the query contains "i2connect" anywhere, it's probably domain-relevant
        if 'i2connect' in query_lower or 'i2 connect' in query_lower:
            return True
        
        # Common non-domain queries that should be clearly rejected
        non_domain_indicators = [
            'weather', 'temperature', 'rain', 'sunny', 'cloudy', 'forecast',
            'cooking', 'recipe', 'food', 'restaurant', 'meal', 'kitchen', 'paella', 'pasta',
            'sports', 'football', 'basketball', 'game', 'match', 'score',
            'movie', 'music', 'entertainment', 'celebrity', 'actor', 'song',
            'politics', 'election', 'president', 'government',
            'shopping', 'price', 'buy', 'sell', 'market', 'store',
            'health', 'medicine', 'doctor', 'hospital', 'disease',
            'travel', 'vacation', 'hotel', 'flight', 'tourism'
        ]
        
        # Only reject if it's clearly about non-domain topics
        if any(indicator in query_lower for indicator in non_domain_indicators):
            # Double-check: reject unless it also contains core domain terms
            core_domain_terms = ['i2connect', 'evidence theory', 'traffic', 'safety', 'risk']
            if not any(term in query_lower for term in core_domain_terms):
                return False
        
        # Default to ACCEPT for ambiguous queries - let context check handle it
        return True
        """Simple Weaviate connection check"""
        return True

    def generate_answer_with_enhanced_ood(self, query: str, context_sentences: List[str]) -> Dict:
        """Generate answer with enhanced OOD detection"""
        try:
            # FIRST: Check if the query itself is I2Connect related
            query_is_domain_relevant = self._is_query_domain_relevant(query)
            
            # THEN: Check if context is relevant using enhanced OOD detection
            is_relevant, ood_diagnostics = self.ood_detector.is_context_relevant(
                context_sentences,
                min_score_threshold=1.0,  # More strict threshold
                max_ood_ratio=0.2  # Lower tolerance for OOD indicators
            )
            
            print(f"🔍 Query Domain Check - Relevant: {query_is_domain_relevant}")
            print(f"🔍 Context OOD Analysis - Relevant: {is_relevant}, Score: {ood_diagnostics['relevance_score']:.2f}")
            if ood_diagnostics['matched_indicators']:
                print(f"📋 Top matches: {[match[0] for match in ood_diagnostics['matched_indicators'][:3]]}")
            
            # Combined decision: BOTH query AND context must be relevant
            overall_relevant = query_is_domain_relevant and is_relevant
            
            # If either query or context is not relevant, return OOD response
            if not overall_relevant:
                rejection_reason = []
                if not query_is_domain_relevant:
                    rejection_reason.append("query not I2Connect-related")
                if not is_relevant:
                    rejection_reason.append("retrieved context not domain-relevant")
                
                return {
                    "answer": "I can only help with I2Connect topics.",
                    "is_ood": True,
                    "ood_diagnostics": ood_diagnostics,
                    "rejection_reason": "; ".join(rejection_reason),
                    "confidence": 0.0
                }
                return {
                    "answer": "I can only provide information about the I2Connect traffic safety research project, including evidence theory, risk assessment, driver monitoring, and related safety concepts. Could you please rephrase your question to focus on these topics?",
                    "is_ood": True,
                    "ood_diagnostics": ood_diagnostics,
                    "confidence": 0.0
                }
            
            # Generate answer using existing LLM generator
            from backend.llm_generator import LLMGenerator
            claude = LLMGenerator(model="claude-3-5-haiku-20241022")

            # Enhanced system prompt
            system_prompt = """You are an expert on the I2Connect traffic safety research project with comprehensive knowledge.

Answer the user's question based on the provided context. Be confident and informative:

1. Use the context to provide specific, detailed answers about I2Connect topics
2. Focus on evidence theory, risk assessment, driver monitoring, traffic safety, and related concepts
3. When discussing technical concepts like Dempster-Shafer theory or belief functions, explain them clearly
4. Reference specific organizations, partners, or technologies when mentioned in context
5. Structure your response clearly and comprehensively
6. Use definitive language when the context supports it

Key I2Connect areas include: evidence theory, Dempster-Shafer methods, traffic safety, intersection risk assessment, driver monitoring systems, Smart Eye gaze tracking, Viscando sensors, Scania vehicles, ADAS systems, and safety concepts."""

            claude_response = claude.generate(
                query=query,
                context_sentences=context_sentences,
                system_prompt=system_prompt
            )

            # Calculate confidence based on relevance score
            confidence = min(ood_diagnostics["relevance_score"] / 3.0, 1.0)

            return {
                "answer": claude_response,
                "is_ood": False,
                "ood_diagnostics": ood_diagnostics,
                "confidence": confidence
            }

        except Exception as e:
            print(f"⚠️ Answer generation failed: {e}")
            return {
                "answer": f"I encountered an error while processing your I2Connect-related query: {str(e)}",
                "is_ood": False,
                "error": str(e),
                "confidence": 0.0
            }

    # Updated legacy method for backward compatibility
    def bypass_ood_detection(self, query: str, context_sentences: list) -> str:
        """Legacy method - now uses enhanced OOD detection internally"""
        result = self.generate_answer_with_enhanced_ood(query, context_sentences)
        return result["answer"]
    
    def query(self, query: str, enable_enhanced_ood: bool = True) -> dict:
        """Execute a query with enhanced OOD detection"""

        try:
            print(f"🔍 Processing query: {query}")

            # EARLY OOD CHECK - before any retrieval
            if enable_enhanced_ood:
                query_is_domain_relevant = self._is_query_domain_relevant(query)
                print(f"🔍 Early Query Check - Relevant: {query_is_domain_relevant}")
                
                if not query_is_domain_relevant:
                    return {
                        "answer": "I can only help with I2Connect topics.",
                        "is_ood": True,
                        "ood_diagnostics": {"reason": "query_not_domain_relevant", "early_rejection": True},
                        "confidence": 0.0,
                        "vector_results": 0,
                        "graph_results": 0,
                        "context_sources": 0,
                        "environment": self.config.environment,
                        "neo4j_uri": self._get_current_neo4j_uri(),
                        "weaviate_url": self._get_current_weaviate_url()
                    }

            # Only proceed with retrieval if query passes early check
            print("🔍 Attempting vector search...")
            query_embedding = self.pipeline.text_embedder.run(text=query)["embedding"]
            vector_docs = self.pipeline.retriever.run(query_embedding=query_embedding)
            vector_results = vector_docs.get("documents", [])

            print(f"📊 Vector search found {len(vector_results)} documents")

            # Try graph search
            print("🔍 Attempting graph search...")
            graph_results = self.pipeline.graph_builder.graph_search(query)
            print(f"📊 Graph search found {len(graph_results)} results")

            # Prepare context from both sources
            context_parts = []

            # Add vector context (prioritize relevant docs)
            for doc in vector_results[:5]:
                if len(doc.content.strip()) > 50:
                    context_parts.append(doc.content)

            # Add graph context
            for result in graph_results[:3]:
                context_parts.append(f"{result['source']} {result['relationship']} {result['target']}")

            print(f"🔍 Combined context from {len(context_parts)} sources")

            # Generate answer with enhanced OOD detection
            if context_parts:
                if enable_enhanced_ood:
                    generation_result = self.generate_answer_with_enhanced_ood(query, context_parts)
                    
                    return {
                        "answer": generation_result["answer"],
                        "is_ood": generation_result["is_ood"],
                        "ood_diagnostics": generation_result.get("ood_diagnostics", {}),
                        "confidence": generation_result.get("confidence", 0.0),
                        "vector_results": len(vector_results),
                        "graph_results": len(graph_results),
                        "context_sources": len(context_parts),
                        "environment": self.config.environment,
                        "neo4j_uri": self._get_current_neo4j_uri(),
                        "weaviate_url": self._get_current_weaviate_url()
                    }
                else:
                    # Use legacy method for backward compatibility
                    answer = self.bypass_ood_detection(query, context_parts)
                    return {
                        "answer": answer,
                        "vector_results": len(vector_results),
                        "graph_results": len(graph_results),
                        "context_sources": len(context_parts),
                        "environment": self.config.environment,
                        "neo4j_uri": self._get_current_neo4j_uri(),
                        "weaviate_url": self._get_current_weaviate_url()
                    }

            # If no meaningful context found
            return {
                "answer": "I found limited information about your query. Could you try rephrasing your question or asking about specific aspects of the I2Connect project?",
                "is_ood": False,
                "vector_results": len(vector_results),
                "graph_results": len(graph_results),
                "context_sources": 0,
                "environment": self.config.environment,
                "neo4j_uri": self._get_current_neo4j_uri(),
                "weaviate_url": self._get_current_weaviate_url()
            }

        except Exception as e:
            print(f"❌ Query processing error: {e}")
            return {
                "answer": f"I encountered an error while processing your query: {str(e)}",
                "error": str(e),
                "environment": self.config.environment,
                "neo4j_uri": self._get_current_neo4j_uri(),
                "weaviate_url": self._get_current_weaviate_url()
            }

    def analyze_context_for_debug(self, context_sentences: List[str]) -> Dict:
        """Debug method to analyze context quality"""
        return self.ood_detector.analyze_context_quality(context_sentences)
        
    def generate_answer_with_context(self, query: str, context_parts: list) -> str:
        """Generate answer with more permissive logic - kept for backward compatibility"""
        result = self.generate_answer_with_enhanced_ood(query, context_parts)
        return result["answer"]
        
    def get_stats(self) -> dict:
        """Get statistics from the knowledge graph."""
        try:
            with self.pipeline.graph_builder.driver.session() as session:
                entity_result = session.run("MATCH (n:Entity) RETURN count(n) as count")
                entity_count = entity_result.single()["count"]
                
                rel_result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
                rel_count = rel_result.single()["count"]
            
            return {
                "entities": entity_count,
                "relationships": rel_count,
                "environment": self.config.environment,
                "neo4j_uri": self._get_current_neo4j_uri(),
                "weaviate_url": self._get_current_weaviate_url()
            }
        except Exception as e:
            return {
                "entities": 0,
                "relationships": 0,
                "environment": self.config.environment,
                "error": str(e)
            }

    def get_health(self) -> dict:
        """Get health status with environment information."""
        return {
            "status": "ok",
            "environment": self.config.environment,
            "neo4j_uri": self._get_current_neo4j_uri(),
            "weaviate_url": self._get_current_weaviate_url()
        }


__all__ = ["RAGBackend"]