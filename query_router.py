#!/usr/bin/env python3
"""
Enhanced RAG Integration
Integrates query classification with your existing RAG backend
"""

from query_classifier import I2ConnectQueryClassifier
from response_formatter import ResponseFormatter  # NEW: Add response formatter
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class QueryRouter:
    """
    Query Router with intelligent domain classification
    Routes queries to your existing RAG backend with domain-aware processing
    """
    
    def __init__(self, existing_rag_backend):
        """
        Initialize with your existing RAG backend
        
        Args:
            existing_rag_backend: Your current RAG system (rag_backend.py)
        """
        self.rag_backend = existing_rag_backend
        self.query_classifier = I2ConnectQueryClassifier()
        self.response_formatter = ResponseFormatter()  # NEW: Initialize formatter
        
    def route_query(self, query: str) -> Dict[str, Any]:
        """
        Route query with domain-aware classification
        
        Args:
            query: User query
            
        Returns:
            Routed results with classification info and enhanced formatting
        """
        
        # Step 1: Classify the query
        classification = self.query_classifier.classify_query(query)
        
        logger.info(f"Query classified as: {classification.primary_domain} "
                   f"(confidence: {classification.confidence:.2f})")
        
        # Step 2: Use enhanced query for better retrieval
        search_query = classification.enhanced_query
        
        # Step 3: Get metadata filters
        metadata_filters = self.query_classifier.get_metadata_filters(classification)
        
        # Step 4: Retrieve with domain-aware parameters
        try:
            # Use your existing RAG backend but with enhanced parameters
            if hasattr(self.rag_backend, 'query_with_filters'):
                # If your backend supports metadata filtering
                results = self.rag_backend.query_with_filters(
                    search_query, 
                    filters=metadata_filters
                )
            else:
                # Fallback to standard query
                results = self.rag_backend.query(search_query)
                
                # Post-process results based on classification
                results = self._post_process_results(results, classification, metadata_filters)
            
            # NEW: Step 5: Format the response for better readability
            domain = classification.primary_domain
            results = self._format_response(results, domain)
            
            # Step 6: Package routed response with classification metadata
            routed_results = {
                'query': query,
                'enhanced_query': search_query,
                'classification': {
                    'domain': classification.primary_domain,
                    'confidence': classification.confidence,
                    'keywords': classification.routing_keywords
                },
                'results': results,
                'boost_factors_applied': classification.boost_factors
            }
            
            return routed_results
            
        except Exception as e:
            logger.error(f"Query routing failed: {e}")
            # Fallback to original query
            fallback_results = self.rag_backend.query(query)
            # NEW: Format fallback response too
            fallback_results = self._format_response(fallback_results, 'general')
            
            return {
                'query': query,
                'results': fallback_results,
                'fallback': True,
                'error': str(e)
            }
    
    def _format_response(self, result: Dict[str, Any], domain: str) -> Dict[str, Any]:
        """
        NEW: Format the response for better readability
        
        Args:
            result: RAG backend result dictionary
            domain: Domain classification for domain-specific formatting
            
        Returns:
            Result dictionary with formatted answer
        """
        
        if not result or not isinstance(result, dict):
            return result
        
        # Format the answer if it exists
        if 'answer' in result and isinstance(result['answer'], str):
            formatted_answer = self.response_formatter.format_response(
                result['answer'], 
                domain
            )
            
            # Update the result with formatted answer
            result_copy = result.copy()
            result_copy['answer'] = formatted_answer
            result_copy['formatting_applied'] = True
            result_copy['formatted_for_domain'] = domain
            
            logger.info(f"Applied {domain} formatting to response")
            return result_copy
        
        return result
    
    def _post_process_results(self, results: List[Dict], classification, filters: Dict[str, Any]) -> List[Dict]:
        """Post-process results based on classification when backend doesn't support filtering"""
        
        if not results:
            return results
        
        scored_results = []
        
        for result in results:
            score = result.get('score', 0.0)
            metadata = result.get('metadata', {})
            
            # Apply domain-specific boosting
            domain_boost = 1.0
            
            # Document type boosting
            doc_type = metadata.get('document_type')
            if classification.primary_domain == 'evidence_theory' and doc_type == 'mathematical':
                domain_boost *= 1.5
            elif classification.primary_domain == 'project_management' and doc_type == 'project':
                domain_boost *= 1.5
            elif classification.primary_domain == 'safety_systems' and doc_type in ['technical', 'project']:
                domain_boost *= 1.3
            
            # Content-specific boosting
            if filters.get('contains_equations') and metadata.get('contains_equations'):
                domain_boost *= 1.2
            if filters.get('contains_work_packages') and metadata.get('contains_work_packages'):
                domain_boost *= 1.2
            if filters.get('contains_organizations') and metadata.get('key_entities'):
                # Check if result contains organization names
                entities = metadata.get('key_entities', [])
                org_names = ['Scania', 'Smart Eye', 'Viscando', 'University of Skövde']
                if any(org in str(entities) for org in org_names):
                    domain_boost *= 1.4
            
            # Apply boost to score
            boosted_score = score * domain_boost
            
            result_copy = result.copy()
            result_copy['original_score'] = score
            result_copy['boosted_score'] = boosted_score
            result_copy['domain_boost'] = domain_boost
            
            scored_results.append(result_copy)
        
        # Re-sort by boosted score
        scored_results.sort(key=lambda x: x.get('boosted_score', 0), reverse=True)
        
        return scored_results
    
    def explain_routing(self, query: str) -> Dict[str, Any]:
        """Explain how a query would be routed (for debugging)"""
        
        classification = self.query_classifier.classify_query(query)
        metadata_filters = self.query_classifier.get_metadata_filters(classification)
        
        return {
            'original_query': query,
            'enhanced_query': classification.enhanced_query,
            'domain_classification': {
                'primary_domain': classification.primary_domain,
                'confidence': classification.confidence,
                'matched_keywords': classification.routing_keywords
            },
            'boost_factors': classification.boost_factors,
            'metadata_filters': metadata_filters,
            'routing_explanation': self._explain_routing_logic(classification),
            'formatting_will_apply': classification.primary_domain  # NEW: Show what formatting will be applied
        }
    
    def _explain_routing_logic(self, classification) -> str:
        """Generate human-readable explanation of routing logic"""
        
        explanations = {
            'evidence_theory': "Routed to mathematical/theoretical content with equation preservation and mathematical formatting",
            'project_management': "Routed to project documents with work package and partner information, plus organizational formatting", 
            'safety_systems': "Routed to technical safety systems and ADAS content with technical term highlighting",
            'general': "No specific domain detected, using general retrieval and formatting"
        }
        
        base_explanation = explanations.get(classification.primary_domain, "Unknown routing")
        
        if classification.confidence > 0.7:
            confidence_note = "High confidence routing"
        elif classification.confidence > 0.4:
            confidence_note = "Medium confidence routing"
        else:
            confidence_note = "Low confidence routing - may need query refinement"
        
        return f"{base_explanation}. {confidence_note}."


# Integration with your existing app.py
def integrate_with_flask_app():
    """
    Example of how to integrate QueryRouter with your Flask app
    """
    
    # In your app.py, add the router to your existing RAG:
    
    example_code = '''
    # Add this to your app.py:
    
    from rag_backend import RAGBackend  # Your existing backend
    from query_router import QueryRouter
    
    # Initialize query router (wraps your existing RAG)
    rag_backend = RAGBackend()  # Your existing RAG backend (unchanged)
    query_router = QueryRouter(rag_backend)  # NEW: Smart routing wrapper with formatting
    
    # In your query endpoint:
    @app.route('/query', methods=['POST'])
    def query():
        user_query = request.json.get('query', '')
        
        # Route the query intelligently (instead of direct rag_backend.query)
        routed_results = query_router.route_query(user_query)
        
        return jsonify({
            'query': routed_results['query'],
            'routed_query': routed_results.get('enhanced_query'),
            'domain': routed_results.get('classification', {}).get('domain'),
            'confidence': routed_results.get('classification', {}).get('confidence'),
            'formatted': routed_results['results'].get('formatting_applied', False),
            'answer': routed_results['results']  # Your existing result format (now formatted!)
        })
    '''
    
    return example_code

if __name__ == "__main__":
    # Test the query router with formatting
    print("🧪 Query Router with Response Formatting Test")
    print("=" * 60)
    
    # This would use your actual RAG backend
    print("Enhanced features:")
    print("✅ Domain-aware query routing")
    print("✅ Content-specific boosting")
    print("✅ Response formatting for better readability")
    print("✅ Sequential numbering fixes")
    print("✅ Improved indentation")
    print()
    print("To integrate:")
    print("1. Save query_classifier.py")
    print("2. Save response_formatter.py") 
    print("3. Replace query_router.py with this enhanced version")
    print("4. Restart your app: python app.py")
    print("5. Test with: 'What are the I2Connect project partners?'")
    
    print("\nIntegration code:")
    print(integrate_with_flask_app())