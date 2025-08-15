#!/usr/bin/env python3
"""
Response Formatter for Enhanced RAG Output
Improves formatting of responses for better readability
"""

import re
from typing import Dict, Any

class ResponseFormatter:
    """Format RAG responses for better readability"""
    
    def __init__(self):
        self.setup_formatting_rules()
    
    def setup_formatting_rules(self):
        """Define formatting rules for different content types"""
        
        # Patterns for different list types
        self.list_patterns = {
            'numbered': re.compile(r'^(\d+)\.?\s+(.+)$', re.MULTILINE),
            'bulleted': re.compile(r'^[•*-]\s+(.+)$', re.MULTILINE),
            'sub_bulleted': re.compile(r'^\s+[•*-]\s+(.+)$', re.MULTILINE)
        }
        
        # Patterns for structure improvement
        self.structure_patterns = {
            'headers': re.compile(r'^([A-Z][^.!?]*):$', re.MULTILINE),
            'emphasis': re.compile(r'\*\*([^*]+)\*\*'),
            'code_blocks': re.compile(r'```([^`]+)```')
        }
    
    def format_response(self, response: str, domain: str = None) -> str:
        """
        Format a response based on content type and domain
        
        Args:
            response: Raw response text
            domain: Domain classification (evidence_theory, project_management, etc.)
            
        Returns:
            Formatted response with improved structure
        """
        
        if not response or not response.strip():
            return response
        
        # Apply domain-specific formatting
        if domain == 'evidence_theory':
            formatted = self._format_mathematical_content(response)
        elif domain == 'project_management':
            formatted = self._format_project_content(response)
        elif domain == 'safety_systems':
            formatted = self._format_technical_content(response)
        else:
            formatted = self._format_general_content(response)
        
        # Apply general formatting improvements
        formatted = self._fix_list_numbering(formatted)
        formatted = self._improve_indentation(formatted)
        formatted = self._enhance_structure(formatted)
        
        return formatted.strip()
    
    def _format_mathematical_content(self, text: str) -> str:
        """Format mathematical/Evidence Theory content"""
        
        # Preserve mathematical notation
        text = self._preserve_mathematical_notation(text)
        
        # Format definitions and theorems
        text = re.sub(
            r'(Definition|Theorem|Proposition|Lemma)(\s*\d*):?\s*',
            r'**\1\2:**\n\n',
            text,
            flags=re.IGNORECASE
        )
        
        # Format equations
        text = re.sub(
            r'(bel\([^)]+\)|pl\([^)]+\)|m\([^)]+\))\s*=\s*([^\n]+)',
            r'`\1 = \2`',
            text
        )
        
        return text
    
    def _format_project_content(self, text: str) -> str:
        """Format project management content"""
        
        # Format work packages
        text = re.sub(
            r'(Work Package|WP)\s*(\d+)',
            r'**\1 \2**',
            text,
            flags=re.IGNORECASE
        )
        
        # Format deliverables
        text = re.sub(
            r'(Deliverable|D)(\d+\.\d+)',
            r'**\1 \2**',
            text,
            flags=re.IGNORECASE
        )
        
        # Format organization names
        orgs = ['Scania', 'Smart Eye', 'Viscando', 'University of Skövde']
        for org in orgs:
            text = re.sub(
                f'\\b({re.escape(org)})\\b',
                f'**{org}**',
                text,
                flags=re.IGNORECASE
            )
        
        return text
    
    def _format_technical_content(self, text: str) -> str:
        """Format technical/safety systems content"""
        
        # Format technical terms
        tech_terms = ['ADAS', 'V2X', 'HMI', 'DMS', 'Driver Monitoring']
        for term in tech_terms:
            text = re.sub(
                f'\\b({re.escape(term)})\\b',
                f'**{term}**',
                text,
                flags=re.IGNORECASE
            )
        
        return text
    
    def _format_general_content(self, text: str) -> str:
        """Format general content"""
        return text
    
    def _fix_list_numbering(self, text: str) -> str:
        """Fix sequential numbering in lists - simplified approach"""

        # Simple regex to find lines that start with "number. text:"
        pattern = r'^(\d+)\.\s+([^:\n]+):\s*(.*)$'

        lines = text.split('\n')
        formatted_lines = []
        current_number = 1

        for line in lines:
            stripped = line.strip()

            # Check if this line matches the numbered pattern
            match = re.match(pattern, stripped)

            if match:
                # Extract the title and content parts
                title = match.group(2)
                content = match.group(3) if match.group(3) else ""

                # Replace with correct sequential number
                if content:
                    formatted_lines.append(f"{current_number}. {title}: {content}")
                else:
                    formatted_lines.append(f"{current_number}. {title}:")

                current_number += 1
            else:
                # Not a numbered item, keep as is
                formatted_lines.append(line)

        return '\n'.join(formatted_lines)
    
    def _improve_indentation(self, text: str) -> str:
        """Improve indentation for nested lists"""
        
        lines = text.split('\n')
        formatted_lines = []
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Look for lines that should be indented as sub-items
            if self._should_be_sub_item(stripped, lines, i):
                # Check if it's already properly indented
                if not line.startswith('  '):
                    formatted_lines.append(f"  {stripped}")
                else:
                    formatted_lines.append(line)
            else:
                formatted_lines.append(line)
        
        return '\n'.join(formatted_lines)
    
    def _should_be_sub_item(self, line: str, all_lines: list, index: int) -> bool:
        """Determine if a line should be indented as a sub-item"""
        
        # Check for patterns that indicate sub-items
        sub_item_patterns = [
            r'^[•*-]\s+',  # Bullet point
            r'^[A-Z][^.!?]*:$',  # Category header followed by items
            r'^(Computed|Proximity|Driver)\s+',  # Specific sub-items
        ]
        
        for pattern in sub_item_patterns:
            if re.match(pattern, line):
                # Check if previous line suggests this should be indented
                if index > 0:
                    prev_line = all_lines[index - 1].strip()
                    if prev_line.endswith(':') or 'based on:' in prev_line.lower():
                        return True
        
        return False
    
    def _enhance_structure(self, text: str) -> str:
        """Enhance overall text structure"""
        
        # Add spacing around headers
        text = re.sub(
            r'^([A-Z][^.!?]*):$',
            r'\n**\1:**\n',
            text,
            flags=re.MULTILINE
        )
        
        # Improve spacing around lists
        text = re.sub(
            r'\n(\d+\.|\*|\-)\s+',
            r'\n\n\1 ',
            text
        )
        
        # Clean up excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text
    
    def _preserve_mathematical_notation(self, text: str) -> str:
        """Preserve mathematical notation and formatting"""
        
        # Protect mathematical expressions
        math_patterns = [
            r'bel\([^)]+\)',
            r'pl\([^)]+\)', 
            r'm\([^)]+\)',
            r'[∑∫∀∃]',
            r'\b[a-zA-Z]\s*=\s*[^,\n]+',
        ]
        
        for pattern in math_patterns:
            text = re.sub(
                f'({pattern})',
                r'`\1`',
                text,
                flags=re.IGNORECASE
            )
        
        return text

# Integration function
def format_rag_response(response_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format a RAG response dictionary with enhanced formatting
    
    Args:
        response_dict: Dictionary containing response and metadata
        
    Returns:
        Dictionary with formatted response
    """
    
    formatter = ResponseFormatter()
    
    # Extract response and domain info
    raw_answer = response_dict.get('answer', '')
    domain = response_dict.get('classification', {}).get('domain', 'general')
    
    # Format the response
    formatted_answer = formatter.format_response(raw_answer, domain)
    
    # Update the response dictionary
    response_dict['answer'] = formatted_answer
    response_dict['formatting_applied'] = True
    response_dict['formatted_for_domain'] = domain
    
    return response_dict

# Example usage
if __name__ == "__main__":
    # Test the formatter
    formatter = ResponseFormatter()
    
    test_response = """
Based on the context provided, here's a summary:

1. Scania
Industrial partner
Provides real-world validation

3. Smart Eye
Technology partner
Adapts alerts based on:
Computed risk thresholds
Proximity of road actors
Driver gaze engagement

4. Viscando
Technology partner
"""
    
    print("Original:")
    print(test_response)
    print("\n" + "="*50 + "\n")
    
    formatted = formatter.format_response(test_response, 'project_management')
    print("Formatted:")
    print(formatted)