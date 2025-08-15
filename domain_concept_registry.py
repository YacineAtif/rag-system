import logging
import re
from typing import Dict, List, Optional, Sequence, Tuple, Any

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
try:
    from sentence_transformers import SentenceTransformer
    _HAS_TRANSFORMER = True
except Exception:  # pragma: no cover - optional dependency
    _HAS_TRANSFORMER = False

logger = logging.getLogger(__name__)


class DomainConceptRegistry:
    """
    Enhanced Registry of known domain concepts and their embeddings.
    Optimized for Evidence Theory + I2Connect + Traffic Safety domains
    Enhanced with explicit Safety Concept support
    """

    def __init__(self, concepts: Sequence[str], synonyms: Optional[Dict[str, str]] = None):
        self.concepts = [c.lower() for c in concepts]
        self.synonyms = {k.lower(): v.lower() for k, v in (synonyms or {}).items()}
        
        # Enhanced I2Connect domain concepts with explicit Safety Concept support
        self.domain_concepts = {
            'evidence_theory': [
                'evidence theory', 'dempster-shafer', 'belief functions', 'plausibility',
                'basic probability assignment', 'bpa', 'mass function', 'uncertainty',
                'frame of discernment', 'focal elements', 'combination rules'
            ],
            'project_management': [
                'i2connect', 'work package', 'wp', 'deliverable', 'milestone',
                'task', 'evaluation', 'demonstrator', 'prototype', 'proof of concept',
                'ffi', 'vinnova', 'requirements', 'methodology'
            ],
            'safety_systems': [
                'adas', 'advanced driver assistance', 'traffic safety', 'collision',
                'risk assessment', 'driver monitoring', 'dms', 'hmi', 'v2x',
                'situation awareness', 'safety confidence', 'threat assessment'
            ],
            # NEW: Explicit high-priority domain concepts (configurable)
            'high_priority_concepts': [
                'concept 1', 'concept 2', 'safety concept', 'safety concept 1', 'safety concept 2',
                'first concept', 'second concept', 'comparative analysis', 'comparative summary'
            ],
            'organizations': [
                'university of skövde', 'his', 'scania', 'smart eye', 'viscando',
                'traton', 'vti', 'safer', 'euroncap', 'unece'
            ],
            'technologies': [
                'cloud platform', 'real-time processing', 'data fusion',
                'information fusion', 'machine learning', 'ai', 'iot'
            ]
        }
        
        # Flatten all domain concepts
        all_domain_concepts = []
        for domain_list in self.domain_concepts.values():
            all_domain_concepts.extend(domain_list)
        
        # Combine provided concepts with domain concepts
        self.all_concepts = list(set(self.concepts + all_domain_concepts))
        
        if _HAS_TRANSFORMER:
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
            try:
                self._embedder = SentenceTransformer(model_name, )
                self._use_transformer = True
            except Exception as exc:  # pragma: no cover - network failure fallback
                logger.warning("Falling back to HashingVectorizer embeddings due to: %s", exc)
                self._use_transformer = False
                self._vectorizer = HashingVectorizer(n_features=384, alternate_sign=False, norm="l2")
        else:  # pragma: no cover - sklearn fallback
            self._use_transformer = False
            self._vectorizer = HashingVectorizer(n_features=384, alternate_sign=False, norm="l2")
        
        # Create embeddings for all concepts
        self.embeddings = {c: self._embed(c) for c in self.all_concepts}
        
        # Create domain-specific boost factors (configurable)
        self.domain_boosts = {
            'high_priority_concepts': 3.5,  # Configurable high-priority concepts
            'evidence_theory': 3.0,
            'project_management': 2.8,
            'safety_systems': 2.7,
            'organizations': 2.4,
            'technologies': 2.2
        }

    def _embed(self, text: str) -> np.ndarray:
        if self._use_transformer:
            logger.debug("Embedding text '%s'", text)
            return np.asarray(
                self._embedder.encode(
                    text,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    convert_to_tensor=False,
                ),
                dtype=np.float32,
            )

        return self._vectorizer.transform([text]).toarray()[0]

    def resolve(self, term: str) -> str:
        """Resolve term using synonyms and domain-specific mappings"""
        term_l = term.lower()
        
        # Check explicit synonyms first
        if term_l in self.synonyms:
            return self.synonyms[term_l]
        
        # Enhanced domain-specific mappings (configurable)
        domain_mappings = {
            'his': 'university of skövde',
            'wp': 'work package',
            'dms': 'driver monitoring system',
            'hmi': 'human-machine interface',
            'bpa': 'basic probability assignment',
            'ds': 'dempster-shafer',
            'et': 'evidence theory'
        }
        
        return domain_mappings.get(term_l, term_l)

    def match(self, term: str) -> Tuple[str, float]:
        """Return the closest concept and similarity score with domain boosting"""
        term = self.resolve(term)
        emb = self._embed(term)
        best, best_score = "", 0.0
        
        for concept, c_emb in self.embeddings.items():
            denom = (np.linalg.norm(emb) * np.linalg.norm(c_emb)) or 1.0
            score = float(np.dot(emb, c_emb) / denom)
            
            # Apply domain boosting
            domain_boost = self._get_domain_boost(concept)
            boosted_score = score * domain_boost
            
            if boosted_score > best_score:
                best, best_score = concept, boosted_score
        
        # Normalize score back to 0-1 range
        max_boost = max(self.domain_boosts.values())
        normalized_score = min(best_score / max_boost, 1.0)
        
        return best, normalized_score

    def _get_domain_boost(self, concept: str) -> float:
        """Get domain-specific boost factor for a concept"""
        for domain, concepts in self.domain_concepts.items():
            if concept in concepts:
                return self.domain_boosts.get(domain, 1.0)
        return 1.0

    def extract_from_query(self, query: str) -> List[str]:
        """Enhanced query concept extraction with domain awareness and configurable priority patterns"""
        tokens = re.findall(r"\w+", query.lower())
        matched: List[str] = []
        
        # Also check for multi-word concepts
        query_lower = query.lower()
        
        # Check for multi-word domain concepts first
        for domain, concepts in self.domain_concepts.items():
            for concept in concepts:
                if concept in query_lower:
                    matched.append(concept)
        
        # Check individual tokens
        for tok in tokens:
            concept, score = self.match(tok)
            if score >= 0.6:  # Lower threshold for better recall
                matched.append(concept)
        
        return list(set(matched))  # Remove duplicates

    def classify_query_domain(self, query: str) -> Tuple[str, float]:
        """
        Classify query into primary domain with confidence score
        Domain-agnostic with configurable high-priority detection
        """
        concepts = self.extract_from_query(query)
        domain_scores = {domain: 0.0 for domain in self.domain_concepts.keys()}
        
        for concept in concepts:
            for domain, domain_concepts in self.domain_concepts.items():
                if concept in domain_concepts:
                    domain_scores[domain] += self.domain_boosts.get(domain, 1.0)
        
        if not any(domain_scores.values()):
            return 'general', 0.0
        
        best_domain = max(domain_scores.items(), key=lambda x: x[1])
        total_score = sum(domain_scores.values())
        confidence = best_domain[1] / total_score if total_score > 0 else 0.0
        
        return best_domain[0], confidence

    def get_related_concepts(self, concept: str, threshold: float = 0.7) -> List[Tuple[str, float]]:
        """Get concepts related to the input concept"""
        if concept not in self.embeddings:
            concept, _ = self.match(concept)
        
        if not concept or concept not in self.embeddings:
            return []
        
        concept_emb = self.embeddings[concept]
        related = []
        
        for other_concept, other_emb in self.embeddings.items():
            if other_concept != concept:
                denom = (np.linalg.norm(concept_emb) * np.linalg.norm(other_emb)) or 1.0
                similarity = float(np.dot(concept_emb, other_emb) / denom)
                
                if similarity >= threshold:
                    related.append((other_concept, similarity))
        
        return sorted(related, key=lambda x: x[1], reverse=True)

    def enhance_chunk_metadata(self, chunk_text: str, existing_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance chunk metadata with domain concept analysis
        Domain-agnostic with configurable content detection
        """
        enhanced_metadata = existing_metadata.copy()
        
        # Extract concepts from chunk
        concepts = self.extract_from_query(chunk_text)
        enhanced_metadata['extracted_concepts'] = concepts
        
        # Classify primary domain
        primary_domain, confidence = self.classify_query_domain(chunk_text)
        enhanced_metadata['primary_domain'] = primary_domain
        enhanced_metadata['domain_confidence'] = confidence
        
        # Calculate domain-specific scores
        domain_scores = {}
        for domain in self.domain_concepts.keys():
            domain_concepts = [c for c in concepts if c in self.domain_concepts[domain]]
            domain_scores[domain] = len(domain_concepts)
        
        enhanced_metadata['domain_scores'] = domain_scores
        
        # Calculate concept density
        total_words = len(chunk_text.split())
        concept_density = len(concepts) / total_words if total_words > 0 else 0.0
        enhanced_metadata['concept_density'] = concept_density
        
        # Identify key relationships
        relationships = []
        for concept in concepts:
            related = self.get_related_concepts(concept, threshold=0.8)
            for related_concept, similarity in related[:3]:  # Top 3 related concepts
                if related_concept in concepts:  # Only if both concepts are in this chunk
                    relationships.append({
                        'source': concept,
                        'target': related_concept,
                        'similarity': similarity,
                        'type': 'conceptual_similarity'
                    })
        
        enhanced_metadata['concept_relationships'] = relationships
        
        return enhanced_metadata

    def get_query_boost_factors(self, query: str) -> Dict[str, float]:
        """
        Get boost factors for query based on detected concepts and domains
        Enhanced with Safety Concept support
        """
        concepts = self.extract_from_query(query)
        primary_domain, confidence = self.classify_query_domain(query)
        
        boost_factors = {}
        
        # Base boost factors from domain classification
        if primary_domain == 'high_priority_concepts' and confidence > 0.5:
            boost_factors['high_priority_queries'] = 3.5
        elif primary_domain == 'evidence_theory' and confidence > 0.5:
            boost_factors['evidence_theory_queries'] = 3.0
        elif primary_domain == 'project_management' and confidence > 0.5:
            boost_factors['project_implementation_queries'] = 2.8
        elif primary_domain == 'safety_systems' and confidence > 0.5:
            boost_factors['safety_system_queries'] = 2.7
        elif primary_domain == 'organizations' and confidence > 0.5:
            boost_factors['partnership_queries'] = 2.4
        
        # Additional boosts for specific concept combinations
        if any(c in concepts for c in ['evidence theory', 'dempster-shafer', 'belief functions']):
            if any(c in concepts for c in ['risk assessment', 'safety', 'adas']):
                boost_factors['cross_domain_integration'] = 2.5
        
        if any(c in concepts for c in ['work package', 'demonstrator', 'evaluation']):
            if any(c in concepts for c in ['adas', 'safety', 'driver monitoring']):
                boost_factors['project_technical_integration'] = 2.3
        
        # High-priority concept specific boosts (configurable)
        high_priority_concepts = self.domain_concepts.get('high_priority_concepts', [])
        if any(c in concepts for c in high_priority_concepts):
            boost_factors['high_priority_focus'] = 3.5
            if any(c in concepts for c in ['comparative', 'comparison', 'vs']):
                boost_factors['comparative_analysis'] = 3.2
        
        return boost_factors

    def suggest_related_queries(self, query: str, max_suggestions: int = 5) -> List[str]:
        """
        Suggest related queries based on concept analysis
        Enhanced with Safety Concept suggestions
        """
        concepts = self.extract_from_query(query)
        primary_domain, _ = self.classify_query_domain(query)
        
        suggestions = []
        
        # High-priority concept specific suggestions (configurable)
        if primary_domain == 'high_priority_concepts':
            suggestions.extend([
                "What are the key concepts?",
                "How do the concepts differ?",
                "Compare the different concepts",
                "What are the technical specifications?",
                "Implementation differences between concepts"
            ])
        
        # Domain-specific query suggestions
        elif primary_domain == 'evidence_theory':
            suggestions.extend([
                "How does Evidence Theory work in risk assessment?",
                "What are Belief Functions and Plausibility?",
                "Explain Dempster-Shafer combination rules",
                "Evidence Theory mathematical framework"
            ])
        
        elif primary_domain == 'project_management':
            suggestions.extend([
                "What are the I2Connect work packages?",
                "I2Connect project deliverables and milestones",
                "How is the I2Connect evaluation conducted?",
                "I2Connect demonstrator development"
            ])
        
        elif primary_domain == 'safety_systems':
            suggestions.extend([
                "How do ADAS systems work?",
                "Driver monitoring system capabilities",
                "V2X communication in traffic safety",
                "Risk assessment in collision avoidance"
            ])
        
        # Cross-domain suggestions (configurable)
        high_priority_concepts = self.domain_concepts.get('high_priority_concepts', [])
        if len(set(concepts) & set(high_priority_concepts)) > 0:
            if len(set(concepts) & set(self.domain_concepts['evidence_theory'])) > 0:
                suggestions.append("How do key concepts use Evidence Theory?")
        
        if len(set(concepts) & set(high_priority_concepts)) > 0:
            if len(set(concepts) & set(self.domain_concepts['safety_systems'])) > 0:
                suggestions.append("Key concepts in system implementation")
        
        return suggestions[:max_suggestions]


# Enhanced factory function for I2Connect domain with explicit Safety Concept support
def create_i2connect_concept_registry() -> DomainConceptRegistry:
    """
    Factory function to create a pre-configured concept registry for I2Connect domain
    Enhanced with explicit Safety Concept support
    """
    
    # Core I2Connect concepts with enhanced Safety Concept support
    core_concepts = [
        # Evidence Theory
        "evidence theory", "dempster-shafer", "belief functions", "plausibility",
        "basic probability assignment", "bpa", "mass function", "uncertainty",
        
        # Project Management  
        "i2connect", "work package", "deliverable", "milestone", "evaluation",
        "demonstrator", "prototype", "ffi", "vinnova",
        
        # Safety Systems
        "adas", "advanced driver assistance", "traffic safety", "collision",
        "risk assessment", "driver monitoring", "situation awareness",
        
        # High-priority domain concepts (configurable per domain)
        "concept 1", "concept 2", "safety concept", "comparative analysis",
        
        # Organizations
        "university of skövde", "scania", "smart eye", "viscando", "traton",
        
        # Technologies
        "v2x", "hmi", "cloud platform", "data fusion", "real-time processing"
    ]
    
    # Enhanced synonyms and abbreviations (configurable per domain)
    synonyms = {
        "his": "university of skövde",
        "wp": "work package", 
        "wp1": "work package 1",
        "wp2": "work package 2",
        "wp3": "work package 3",
        "wp4": "work package 4",
        "wp5": "work package 5",
        "dms": "driver monitoring system",
        "hmi": "human-machine interface",
        "bpa": "basic probability assignment",
        "ds": "dempster-shafer",
        "et": "evidence theory",
        "bf": "belief functions",
        "pf": "plausibility functions",
        "ra": "risk assessment",
        "sa": "situation awareness",
        "v2x": "vehicle-to-everything",
        "iot": "internet of things",
        "ai": "artificial intelligence",
        "ml": "machine learning",
        "cv": "computer vision"
    }
    
    return DomainConceptRegistry(concepts=core_concepts, synonyms=synonyms)


if __name__ == "__main__":
    # Test the enhanced registry with Safety Concept queries
    registry = create_i2connect_concept_registry()
    
    test_queries = [
        "What is Safety Concept 1?",
        "Concept 1 features",
        "Compare Concept 1 and Concept 2",
        "Safety Concept implementation"
    ]
    
    print("Testing Enhanced Safety Concept Registry:")
    print("=" * 50)
    
    for query in test_queries:
        concepts = registry.extract_from_query(query)
        domain, confidence = registry.classify_query_domain(query)
        boost_factors = registry.get_query_boost_factors(query)
        
        print(f"\nQuery: {query}")
        print(f"Concepts: {concepts}")
        print(f"Domain: {domain} (confidence: {confidence:.2f})")
        print(f"Boost Factors: {boost_factors}")
