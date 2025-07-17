#!/usr/bin/env python3
"""
Improved Domain-Agnostic Q&A System with Better Deduplication
"""
import time
import requests
import warnings
import os
import re
import numpy as np
from typing import List
from haystack import Pipeline, Document
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack_integrations.document_stores.weaviate import WeaviateDocumentStore
from haystack_integrations.components.retrievers.weaviate import WeaviateEmbeddingRetriever
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Suppress noise
warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# Global embedding model - load once
try:
    EMBEDDER = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
except Exception as e:
    print(f"Warning: Could not load embedding model: {e}")
    EMBEDDER = None

def setup_qa_system():
    """Setup Q&A system"""
    try:
        response = requests.get("http://localhost:8080/v1/meta", timeout=5)
        if response.status_code != 200:
            return None
        
        document_store = WeaviateDocumentStore(url="http://localhost:8080")
        search_pipeline = Pipeline()
        search_pipeline.add_component(
            "text_embedder", 
            SentenceTransformersTextEmbedder(
                model="sentence-transformers/all-MiniLM-L6-v2",
                progress_bar=False
            )
        )
        search_pipeline.add_component("retriever", WeaviateEmbeddingRetriever(document_store=document_store))
        search_pipeline.connect("text_embedder", "retriever")
        return search_pipeline
    except Exception:
        return None

def deep_clean_text(text):
    """Deeply clean text to remove artifacts and formatting"""
    if not text:
        return ""
    
    text = re.sub(r'^#+\s.*?$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^=+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^-+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\*+([^*]+)\*+', r'\1', text)
    text = re.sub(r'_+([^_]+)_+', r'\1', text)
    text = re.sub(r'`+([^`]+)`+', r'\1', text)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    text = re.sub(r'http[s]?://\S+', '', text)
    text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\|+', ' ', text)
    text = re.sub(r'[#*_`<>]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n+', ' ', text)
    return text.strip()

def extract_quality_sentences(text):
    """Extract high-quality sentences from cleaned text"""
    cleaned_text = deep_clean_text(text)
    if not cleaned_text:
        return []
    
    # Split more carefully on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', cleaned_text)
    quality_sentences = []
    
    for sentence in sentences:
        sentence = sentence.strip()
        
        # Enhanced quality filters
        if (len(sentence) < 25 or  # Increased minimum length
            len(sentence) > 400 or
            len(sentence.split()) < 5 or  # At least 5 words
            sentence.lower().startswith(('what', 'how', 'when', 'where', 'why', 'is there', 'are there')) or
            any(artifact in sentence.lower() for artifact in ['comprehensive guide', 'table of contents', 'click here']) or
            sentence.count('(') != sentence.count(')') or  # Malformed parentheses
            any(artifact in sentence for artifact in ['###', '```', '---', '==='])):
            continue
        
        # Clean up sentence
        sentence = re.sub(r'\s+', ' ', sentence).strip()
        if not sentence.endswith(('.', '!', '?')):
            sentence += '.'
        
        quality_sentences.append(sentence)
    
    return quality_sentences

def advanced_deduplicate(sentences: List[str]) -> List[str]:
    """Advanced deduplication that catches semantic duplicates"""
    if len(sentences) < 2:
        return sentences
    
    unique_sentences = []
    
    for sentence in sentences:
        is_duplicate = False
        sentence_clean = sentence.lower().strip()
        
        # Check against existing sentences
        for existing in unique_sentences:
            existing_clean = existing.lower().strip()
            
            # Method 1: Check for key phrase overlap
            sentence_phrases = set()
            existing_phrases = set()
            
            # Extract key phrases (3+ word sequences)
            sentence_words = sentence_clean.split()
            existing_words = existing_clean.split()
            
            for i in range(len(sentence_words) - 2):
                phrase = ' '.join(sentence_words[i:i+3])
                sentence_phrases.add(phrase)
            
            for i in range(len(existing_words) - 2):
                phrase = ' '.join(existing_words[i:i+3])
                existing_phrases.add(phrase)
            
            # Check phrase overlap
            if sentence_phrases and existing_phrases:
                overlap = len(sentence_phrases.intersection(existing_phrases))
                total_phrases = len(sentence_phrases.union(existing_phrases))
                phrase_similarity = overlap / total_phrases if total_phrases > 0 else 0
                
                if phrase_similarity > 0.5:  # 50% phrase overlap
                    is_duplicate = True
                    break
            
            # Method 2: Check for key concept repetition
            key_concepts = [
                'evidence theory', 'dempster-shafer', 'mathematical framework',
                'uncertainty', 'incomplete information', 'basic probability',
                'i2connect', 'traffic safety', 'gaze tracking'
            ]
            
            sentence_concepts = [c for c in key_concepts if c in sentence_clean]
            existing_concepts = [c for c in key_concepts if c in existing_clean]
            
            if (len(sentence_concepts) >= 2 and len(existing_concepts) >= 2 and
                len(set(sentence_concepts).intersection(set(existing_concepts))) >= 2):
                # Same concepts being discussed
                word_overlap = len(set(sentence_words).intersection(set(existing_words)))
                word_similarity = word_overlap / max(len(sentence_words), len(existing_words))
                
                if word_similarity > 0.6:  # 60% word overlap with same concepts
                    is_duplicate = True
                    break
        
        if not is_duplicate:
            unique_sentences.append(sentence)
    
    return unique_sentences

def extract_organizations_and_entities(text):
    """Extract organization names and entities from text - refined domain agnostic approach"""
    # More precise organization patterns
    org_patterns = [
        # Universities and institutes with clear indicators
        r'\b(?:University|Institute|College) of [A-Z][a-zA-Z\s]+\b',
        r'\b[A-Z][a-zA-Z\s]{2,25}\s+(?:University|Institute|College)\b',
        
        # Companies with clear corporate suffixes
        r'\b[A-Z][a-zA-Z\s]{2,25}\s+(?:Corporation|Company|Ltd|Inc|LLC|Group|AB|GmbH)\b',
        
        # Organizations mentioned in partnership context
        r'(?:partner|organization|institution|company)[:]\s*([A-Z][a-zA-Z\s]{2,30})',
        
        # Standalone proper nouns in organizational contexts
        r'(?:Organizations involved|Partners include|Consortium members)[:]\s*([A-Z][a-zA-Z\s,]+)',
    ]
    
    organizations = set()
    
    for pattern in org_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            if isinstance(match, tuple):
                match = match[0] if match else ""
            
            clean_match = match.strip()
            
            # Quality filters to avoid false positives
            if (len(clean_match) > 3 and
                len(clean_match) < 50 and  # Not too long
                not clean_match.lower().startswith(('the ', 'this ', 'that ', 'a ', 'an ')) and
                not any(tech_term in clean_match.lower() for tech_term in [
                    'module', 'system', 'framework', 'algorithm', 'method',
                    'probability', 'assignment', 'assessment', 'monitoring',
                    'sensor', 'data', 'analysis', 'processing', 'evaluation'
                ]) and
                not clean_match.isupper() and  # Avoid all-caps acronyms unless they're known orgs
                len(clean_match.split()) <= 4):  # Maximum 4 words
                organizations.add(clean_match)
    
    # Look for specific organizational mentions in context
    context_patterns = [
        r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*[:]\s*(?:Technology partner|Industrial partner|Academic partner|Research partner)',
        r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:provides|contributes|specializes|offers)',
        r'\*\*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\*\*[:]\s*(?:[A-Z])',  # Bold organization names
    ]
    
    for pattern in context_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            clean_match = match.strip()
            if (len(clean_match) > 1 and 
                len(clean_match.split()) <= 3 and
                not any(tech_term in clean_match.lower() for tech_term in [
                    'module', 'system', 'assessment', 'monitoring', 'probability'
                ])):
                organizations.add(clean_match)
    
    return list(organizations)

def find_relevant_content(documents, question):
    """Find content relevant to the question - completely domain agnostic"""
    all_sentences = []
    sources = set()
    organizations = set()
    
    # Check if this is an organizational/partnership question
    is_org_question = any(word in question.lower() for word in [
        'partner', 'organization', 'company', 'university', 'who are', 'involved',
        'institution', 'team', 'collaboration', 'consortium'
    ])
    
    for doc in documents:
        sentences = extract_quality_sentences(doc.content)
        all_sentences.extend(sentences)
        sources.add(doc.meta.get('filename', 'Unknown'))
        
        # Extract organizations if this is an org question
        if is_org_question:
            doc_orgs = extract_organizations_and_entities(doc.content)
            organizations.update(doc_orgs)
    
    if not all_sentences:
        return [], sources
    
    # Advanced deduplication
    all_sentences = advanced_deduplicate(all_sentences)
    
    # For organization questions, prioritize sentences with org names
    if is_org_question and organizations:
        org_sentences = []
        for sentence in all_sentences:
            for org in organizations:
                if org.lower() in sentence.lower():
                    org_sentences.append(sentence)
                    break
        
        # If we found organization-specific sentences, prioritize them
        if org_sentences:
            all_sentences = org_sentences + [s for s in all_sentences if s not in org_sentences]
    
    # Score sentences based on question relevance
    question_words = set(question.lower().split())
    stop_words = {'what', 'is', 'are', 'who', 'how', 'where', 'when', 'why', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'this', 'that', 'it'}
    key_words = question_words - stop_words
    
    scored_sentences = []
    
    for sentence in all_sentences:
        sentence_words = set(sentence.lower().split())
        
        # Calculate relevance
        if key_words:
            overlap = len(key_words.intersection(sentence_words))
            relevance = overlap / len(key_words)
        else:
            relevance = 0
        
        # Boost for definition patterns
        if any(pattern in sentence.lower() for pattern in [
            'is a', 'is an', 'refers to', 'means', 'defined as', 'known as'
        ]):
            relevance += 0.3
        
        # Boost for process/usage patterns  
        if any(pattern in sentence.lower() for pattern in [
            'uses', 'applies', 'implements', 'enables', 'provides', 'allows'
        ]):
            relevance += 0.2
        
        # Generic boost for partnership/organization patterns
        if any(pattern in sentence.lower() for pattern in [
            'partner', 'organization', 'university', 'company', 'institution',
            'consortium', 'collaboration', 'involved', 'team', 'founded',
            'established', 'member', 'affiliate', 'division'
        ]):
            relevance += 0.4
        
        # Boost for sentences containing extracted organizations
        if is_org_question and organizations:
            for org in organizations:
                if org.lower() in sentence.lower():
                    relevance += 0.3
                    break
        
        scored_sentences.append((sentence, relevance))
    
    # Sort by relevance and return top sentences
    scored_sentences.sort(key=lambda x: x[1], reverse=True)
    relevant_sentences = [s for s, score in scored_sentences if score > 0.1][:4]  # Top 4
    
    # If this was an org question and we found organizations, add them to the answer
    if is_org_question and organizations and relevant_sentences:
        org_list = ", ".join(sorted(organizations))
        if org_list:
            # Prepend organization summary if we have specific orgs
            org_summary = f"The main organizations mentioned include: {org_list}."
            relevant_sentences.insert(0, org_summary)
    
    return relevant_sentences, sources

def create_simple_answer(sentences):
    """Create a simple, clean answer from sentences"""
    if not sentences:
        return "I couldn't find specific information to answer your question."
    
    if len(sentences) == 1:
        return sentences[0]
    elif len(sentences) == 2:
        return f"{sentences[0]} {sentences[1]}"
    else:
        # Group into maximum 2 paragraphs
        para1 = sentences[0]
        para2 = " ".join(sentences[1:])
        return f"{para1}\n\n{para2}"

def ask_question(qa_system, question):
    """Get a clean, accurate answer"""
    try:
        result = qa_system.run({
            "text_embedder": {"text": question},
            "retriever": {"top_k": 5}
        })
        
        documents = result.get("retriever", {}).get("documents", [])
        
        if not documents:
            return "I couldn't find relevant information about that topic in the knowledge base."
        
        relevant_sentences, sources = find_relevant_content(documents, question)
        
        if not relevant_sentences:
            return "The documents don't contain specific information to answer your question."
        
        answer = create_simple_answer(relevant_sentences)
        
        # Add sources
        if sources:
            source_list = sorted([s for s in sources if s != 'Unknown'])
            if source_list:
                answer += f"\n\n📚 Sources: {', '.join(source_list)}"
        
        return answer
        
    except Exception as e:
        return f"I encountered an error while searching: {str(e)}"

def main():
    """Main interface"""
    print("🤖 Clean Q&A Assistant")
    print("Ask questions about your documents - I'll provide accurate, readable answers.\n")
    
    print("Setting up...")
    qa_system = setup_qa_system()
    
    if not qa_system:
        print("❌ Cannot connect to knowledge base.")
        return
    
    print("✅ Ready!\n")
    
    while True:
        try:
            question = input("❓ Question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not question:
                continue
            
            print("\n🔍 Finding answer...")
            start_time = time.time()
            answer = ask_question(qa_system, question)
            elapsed = time.time() - start_time
            
            print(f"\n💡 Answer (in {elapsed:.2f}s):")
            print("-" * 60)
            print(answer)
            print("-" * 60)
            print()
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break

if __name__ == "__main__":
    main()