#!/usr/bin/env python3
"""
I2Connect Query Classifier
Routes queries to appropriate document domains and applies boost factors
"""
import re
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass

@dataclass
class QueryClassification:
    """Query classification result"""
    primary_domain: str
    confidence: float
    boost_factors: Dict[str, float]
    enhanced_query: str
    routing_keywords: List[str]

class I2ConnectQueryClassifier:
    """
    Classify queries for I2Connect domain and route to appropriate chunks
    """
    
    def __init__(self):
        self.setup_domain_patterns()
        self.setup_boost_factors()
        self.setup_query_expansions()
    
    def setup_domain_patterns(self):
        """Define patterns for different query domains"""
        
        self.domain_patterns = {
            'evidence_theory': {
                'keywords': [
                    'evidence theory', 'dempster', 'shafer', 'belief function', 'plausibility',
                    'bpa', 'basic probability assignment', 'mass function', 'uncertainty',
                    'mathematical', 'theorem', 'proof', 'equation', 'formula'
                ],
                'patterns': [
                    r'(?i)\b(evidence theory|dempster.shafer|belief function|plausibility)\b',
                    r'(?i)\b(mathematical framework|uncertainty|theorem|proof)\b',
                    r'(?i)\b(bpa|mass function|combination rule)\b'
                ]
            },
            
            'project_management': {
                'keywords': [
                    'i2connect', 'project', 'work package', 'wp', 'deliverable', 'milestone',
                    'partners', 'collaboration', 'consortium', 'evaluation', 'methodology',
                    'scania', 'smart eye', 'viscando', 'university', 'skövde'
                ],
                'patterns': [
                    r'(?i)\b(i2connect|work package|wp\d+|deliverable)\b',
                    r'(?i)\b(partners?|collaboration|consortium|evaluation)\b',
                    r'(?i)\b(scania|smart eye|viscando|university.*skövde)\b'
                ]
            },
            
            'safety_systems': {
                'keywords': [
                    'adas', 'safety', 'collision', 'traffic', 'driver monitoring',
                    'risk assessment', 'v2x', 'hmi', 'intersection', 'accident',
                    'warning', 'detection', 'prevention'
                ],
                'patterns': [
                    r'(?i)\b(adas|advanced driver assistance|safety system)\b',
                    r'(?i)\b(collision|traffic safety|driver monitoring)\b',
                    r'(?i)\b(risk assessment|intersection|accident)\b'
                ]
            }
        }
    
    def setup_boost_factors(self):
        """Define boost factors for different query types"""
        
        self.boost_factors = {
            'evidence_theory': 3.0,
            'project_management': 2.8,
            'safety_systems': 2.7,
            'cross_domain': 2.5,
            'general': 1.0
        }
    
    def setup_query_expansions(self):
        """Define query expansions for better matching"""
        
        self.query_expansions = {
            # Project partner queries
            'partners': ['scania', 'smart eye', 'viscando', 'university of skövde', 'consortium', 'collaboration'],
            'project': ['i2connect', 'work package', 'deliverable', 'milestone'],
            
            # Evidence theory queries  
            'evidence theory': ['dempster-shafer', 'belief functions', 'plausibility', 'bpa'],
            'mathematical': ['theorem', 'proof', 'equation', 'formula'],
            
            # Safety queries
            'safety': ['adas', 'collision', 'traffic', 'driver monitoring', 'risk assessment'],
            'adas': ['advanced driver assistance', 'safety system', 'collision avoidance']
        }
    
    def classify_query(self, query: str) -> QueryClassification:
        """
        Classify a query and return routing information
        """
        
        query_lower = query.lower()
        
        # Calculate domain scores
        domain_scores = {}
        matched_keywords = []
        
        for domain, config in self.domain_patterns.items():
            score = 0
            domain_keywords = []
            
            # Check keyword matches
            for keyword in config['keywords']:
                if keyword in query_lower:
                    score += 2
                    domain_keywords.append(keyword)
                    matched_keywords.append(keyword)
            
            # Check pattern matches
            for pattern in config['patterns']:
                matches = re.findall(pattern, query)
                if matches:
                    score += len(matches) * 1.5
                    domain_keywords.extend(matches)
            
            domain_scores[domain] = score
        
        # Determine primary domain
        if not any(domain_scores.values()):
            primary_domain = 'general'
            confidence = 0.0
        else:
            primary_domain = max(domain_scores.items(), key=lambda x: x[1])[0]
            total_score = sum(domain_scores.values())
            confidence = domain_scores[primary_domain] / total_score if total_score > 0 else 0.0
        
        # Generate boost factors
        boost_factors = self._calculate_boost_factors(primary_domain, domain_scores)
        
        # Enhance query with domain-specific terms
        enhanced_query = self._enhance_query(query, primary_domain, matched_keywords)
        
        return QueryClassification(
            primary_domain=primary_domain,
            confidence=confidence,
            boost_factors=boost_factors,
            enhanced_query=enhanced_query,
            routing_keywords=matched_keywords
        )
    
    def _calculate_boost_factors(self, primary_domain: str, domain_scores: Dict[str, float]) -> Dict[str, float]:
        """Calculate boost factors based on domain classification"""
        
        boosts = {}
        
        # Primary domain boost
        if primary_domain in self.boost_factors:
            boosts[f'{primary_domain}_queries'] = self.boost_factors[primary_domain]
        
        # Secondary domain boosts
        for domain, score in domain_scores.items():
            if domain != primary_domain and score > 0:
                boosts[f'{domain}_queries'] = self.boost_factors.get(domain, 1.0) * 0.7
        
        # Cross-domain boost if multiple domains detected
        active_domains = [d for d, s in domain_scores.items() if s > 0]
        if len(active_domains) > 1:
            boosts['cross_domain_queries'] = self.boost_factors['cross_domain']
        
        return boosts
    
    def _enhance_query(self, query: str, primary_domain: str, matched_keywords: List[str]) -> str:
        """Enhance query with domain-specific terms for better retrieval"""
        
        enhanced_query = query
        
        # Add domain-specific expansions
        query_words = query.lower().split()
        
        for word in query_words:
            if word in self.query_expansions:
                expansion_terms = self.query_expansions[word]
                # Add most relevant expansion terms
                for term in expansion_terms[:2]:  # Limit to avoid query bloat
                    if term not in enhanced_query.lower():
                        enhanced_query += f" {term}"
        
        return enhanced_query
    
    def get_metadata_filters(self, classification: QueryClassification) -> Dict[str, Any]:
        """Generate metadata filters based on classification"""
        
        filters = {}
        
        # Document type filter
        if classification.primary_domain == 'evidence_theory':
            filters['document_type'] = 'mathematical'
        elif classification.primary_domain == 'project_management':
            filters['document_type'] = 'project'
        elif classification.primary_domain == 'safety_systems':
            filters['document_type'] = ['technical', 'project']  # Safety spans both
        
        # Content-specific filters
        if 'partners' in classification.enhanced_query.lower():
            filters['contains_organizations'] = True
        
        if any(term in classification.enhanced_query.lower() for term in ['equation', 'formula', 'theorem']):
            filters['contains_equations'] = True
        
        if any(term in classification.enhanced_query.lower() for term in ['work package', 'wp', 'deliverable']):
            filters['contains_work_packages'] = True
        
        return filters

# Usage example
def test_query_classifier():
    """Test the query classifier with sample queries"""
    
    classifier = I2ConnectQueryClassifier()
    
    test_queries = [
        "What are the I2Connect project partners?",
        "How does Evidence Theory work?",
        "Explain ADAS safety systems",
        "What is Dempster-Shafer theory?",
        "I2Connect work packages and deliverables",
        "Risk assessment using belief functions"
    ]
    
    print("🧪 Testing Query Classification:")
    print("=" * 60)
    
    for query in test_queries:
        classification = classifier.classify_query(query)
        
        print(f"\n📝 Query: {query}")
        print(f"   Domain: {classification.primary_domain} (confidence: {classification.confidence:.2f})")
        print(f"   Enhanced: {classification.enhanced_query}")
        print(f"   Boosts: {classification.boost_factors}")
        print(f"   Keywords: {classification.routing_keywords}")

if __name__ == "__main__":
    test_query_classifier()