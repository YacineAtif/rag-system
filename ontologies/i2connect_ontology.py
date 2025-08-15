"""
I2Connect Domain Ontology for Enhanced Entity Recognition and Relationship Modeling
"""

from dataclasses import dataclass
from typing import Dict, List, Set, Optional, Tuple
from enum import Enum
import re
import json

class EntityType(Enum):
    ORGANIZATION = "Organization"
    SAFETY_CONCEPT = "SafetyConcept"
    TECHNOLOGY = "Technology"
    DELIVERABLE = "Deliverable"
    PERSON = "Person"
    LOCATION = "Location"
    TIMEFRAME = "TimeFrame"
    RISK_TYPE = "RiskType"
    SYSTEM_COMPONENT = "SystemComponent"
    RESEARCH_AREA = "ResearchArea"

class RelationshipType(Enum):
    CONTRIBUTES_TO = "CONTRIBUTES_TO"
    DEVELOPS = "DEVELOPS"
    IMPLEMENTS = "IMPLEMENTS"
    ADDRESSES = "ADDRESSES"
    PARTNERS_WITH = "PARTNERS_WITH"
    LEADS = "LEADS"
    PARTICIPATES_IN = "PARTICIPATES_IN"
    USES_TECHNOLOGY = "USES_TECHNOLOGY"
    HAS_DELIVERABLE = "HAS_DELIVERABLE"
    LOCATED_IN = "LOCATED_IN"
    SCHEDULED_FOR = "SCHEDULED_FOR"
    DEPENDS_ON = "DEPENDS_ON"
    MITIGATES = "MITIGATES"
    INTEGRATES_WITH = "INTEGRATES_WITH"

@dataclass
class EntityDefinition:
    canonical_name: str
    aliases: List[str]
    entity_type: EntityType
    attributes: Dict[str, str]
    description: str

@dataclass
class RelationshipDefinition:
    source_type: EntityType
    target_type: EntityType
    relationship_type: RelationshipType
    description: str
    confidence_indicators: List[str]

class I2ConnectOntology:
    """Structured domain ontology for I2Connect traffic safety project"""
    
    def __init__(self):
        self.entities = self._initialize_entities()
        self.relationships = self._initialize_relationships()
        self.extraction_patterns = self._initialize_patterns()
        
    def _initialize_entities(self) -> Dict[str, EntityDefinition]:
        """Initialize domain-specific entities with aliases and attributes"""
        entities = {}
        
        # Organizations
        orgs = [
            EntityDefinition(
                canonical_name="University of Skövde",
                aliases=["University of Skovde", "HiS", "Skövde University", "Skovde University"],
                entity_type=EntityType.ORGANIZATION,
                attributes={"type": "university", "country": "Sweden", "role": "research_lead"},
                description="Swedish university leading I2Connect research"
            ),
            EntityDefinition(
                canonical_name="Scania",
                aliases=["Scania AB", "Scania CV", "Scania Group"],
                entity_type=EntityType.ORGANIZATION,
                attributes={"type": "industry", "country": "Sweden", "role": "vehicle_manufacturer"},
                description="Swedish truck and bus manufacturer, industry partner"
            ),
            EntityDefinition(
                canonical_name="Smart Eye",
                aliases=["Smart Eye AB", "SmartEye"],
                entity_type=EntityType.ORGANIZATION,
                attributes={"type": "technology", "country": "Sweden", "role": "sensor_provider"},
                description="Eye tracking and driver monitoring technology provider"
            ),
            EntityDefinition(
                canonical_name="Viscando",
                aliases=["Viscando AB"],
                entity_type=EntityType.ORGANIZATION,
                attributes={"type": "technology", "country": "Sweden", "role": "traffic_analytics"},
                description="Traffic analytics and sensor technology provider"
            )
        ]
        
        # Safety Concepts
        safety_concepts = [
            EntityDefinition(
                canonical_name="Safety Concept 1",
                aliases=["Concept 1", "Safety Concept I", "Concept I"],
                entity_type=EntityType.SAFETY_CONCEPT,
                attributes={"priority": "high", "focus": "proactive_risk_assessment"},
                description="Proactive risk assessment safety concept"
            ),
            EntityDefinition(
                canonical_name="Safety Concept 2", 
                aliases=["Concept 2", "Safety Concept II", "Concept II"],
                entity_type=EntityType.SAFETY_CONCEPT,
                attributes={"priority": "high", "focus": "actor_focused_risk"},
                description="Actor-focused risk assessment safety concept"
            ),
            EntityDefinition(
                canonical_name="Actor-Focused Risk Assessment",
                aliases=["Actor Focused Risk", "AFRA", "Actor-based Risk"],
                entity_type=EntityType.SAFETY_CONCEPT,
                attributes={"category": "risk_methodology", "complexity": "advanced"},
                description="Risk assessment methodology focusing on individual actors"
            )
        ]
        
        # Technologies
        technologies = [
            EntityDefinition(
                canonical_name="Evidence Theory",
                aliases=["Dempster-Shafer Theory", "Belief Functions", "DS Theory"],
                entity_type=EntityType.TECHNOLOGY,
                attributes={"category": "mathematical_framework", "application": "uncertainty_reasoning"},
                description="Mathematical framework for reasoning under uncertainty"
            ),
            EntityDefinition(
                canonical_name="ADAS",
                aliases=["Advanced Driver Assistance Systems", "Driver Assistance"],
                entity_type=EntityType.TECHNOLOGY,
                attributes={"category": "vehicle_system", "application": "safety_assistance"},
                description="Advanced driver assistance systems"
            ),
            EntityDefinition(
                canonical_name="V2X Communication",
                aliases=["V2X", "Vehicle-to-Everything", "V2V", "V2I"],
                entity_type=EntityType.TECHNOLOGY,
                attributes={"category": "communication", "application": "cooperative_systems"},
                description="Vehicle-to-everything communication technology"
            ),
            EntityDefinition(
                canonical_name="Driver Monitoring System",
                aliases=["DMS", "Driver Monitoring", "Eye Tracking"],
                entity_type=EntityType.TECHNOLOGY,
                attributes={"category": "monitoring_system", "provider": "Smart Eye"},
                description="System for monitoring driver attention and behavior"
            )
        ]
        
        # Risk Types
        risk_types = [
            EntityDefinition(
                canonical_name="Intersection Risk",
                aliases=["Junction Risk", "Crossroad Risk"],
                entity_type=EntityType.RISK_TYPE,
                attributes={"severity": "high", "location_type": "intersection"},
                description="Risk associated with intersection scenarios"
            ),
            EntityDefinition(
                canonical_name="Collision Risk",
                aliases=["Crash Risk", "Accident Risk"],
                entity_type=EntityType.RISK_TYPE,
                attributes={"severity": "critical", "impact": "safety_critical"},
                description="Risk of vehicle collisions"
            )
        ]
        
        # Deliverables/Work Packages
        deliverables = [
            EntityDefinition(
                canonical_name="Work Package 1",
                aliases=["WP1", "WP 1", "Work Package I"],
                entity_type=EntityType.DELIVERABLE,
                attributes={"phase": "1", "focus": "requirements_analysis"},
                description="Requirements analysis and system design work package"
            ),
            EntityDefinition(
                canonical_name="Work Package 2", 
                aliases=["WP2", "WP 2", "Work Package II"],
                entity_type=EntityType.DELIVERABLE,
                attributes={"phase": "2", "focus": "implementation"},
                description="System implementation work package"
            )
        ]
        
        # Combine all entities
        all_entities = orgs + safety_concepts + technologies + risk_types + deliverables
        
        # Create lookup dictionary
        for entity in all_entities:
            entities[entity.canonical_name] = entity
            # Add aliases for lookup
            for alias in entity.aliases:
                entities[alias] = entity
                
        return entities
    
    def _initialize_relationships(self) -> List[RelationshipDefinition]:
        """Define valid relationships between entity types"""
        return [
            RelationshipDefinition(
                source_type=EntityType.ORGANIZATION,
                target_type=EntityType.DELIVERABLE,
                relationship_type=RelationshipType.CONTRIBUTES_TO,
                description="Organization contributes to deliverable",
                confidence_indicators=["leads", "contributes", "responsible for", "develops"]
            ),
            RelationshipDefinition(
                source_type=EntityType.ORGANIZATION,
                target_type=EntityType.TECHNOLOGY,
                relationship_type=RelationshipType.DEVELOPS,
                description="Organization develops technology",
                confidence_indicators=["develops", "creates", "builds", "provides"]
            ),
            RelationshipDefinition(
                source_type=EntityType.SAFETY_CONCEPT,
                target_type=EntityType.RISK_TYPE,
                relationship_type=RelationshipType.ADDRESSES,
                description="Safety concept addresses risk type",
                confidence_indicators=["addresses", "mitigates", "handles", "manages"]
            ),
            RelationshipDefinition(
                source_type=EntityType.TECHNOLOGY,
                target_type=EntityType.SAFETY_CONCEPT,
                relationship_type=RelationshipType.IMPLEMENTS,
                description="Technology implements safety concept",
                confidence_indicators=["implements", "enables", "supports", "realizes"]
            ),
            RelationshipDefinition(
                source_type=EntityType.ORGANIZATION,
                target_type=EntityType.ORGANIZATION,
                relationship_type=RelationshipType.PARTNERS_WITH,
                description="Organization partners with another organization",
                confidence_indicators=["partners with", "collaborates with", "works with"]
            )
        ]
    
    def _initialize_patterns(self) -> Dict[EntityType, List[str]]:
        """Initialize regex patterns for entity recognition"""
        return {
            EntityType.ORGANIZATION: [
                r'\b(University of Sk[öo]vde|Scania|Smart Eye|Viscando)\b',
                r'\b([A-Z][a-z]+ University)\b',
                r'\b([A-Z][A-Z]+ AB)\b'
            ],
            EntityType.SAFETY_CONCEPT: [
                r'\b(Safety )?Concept [12I]+\b',
                r'\bActor[- ]?Focused Risk\b',
                r'\bRisk Assessment\b'
            ],
            EntityType.TECHNOLOGY: [
                r'\b(Evidence Theory|Dempster[- ]?Shafer|ADAS|V2[XIVx])\b',
                r'\bDriver Monitoring\b',
                r'\bBelief Functions?\b'
            ],
            EntityType.DELIVERABLE: [
                r'\bW[Pp] ?[12]\b',
                r'\bWork Package [12I]+\b',
                r'\bDeliverable [D]?\d+\b'
            ]
        }
    
    def extract_entities(self, text: str) -> List[Tuple[str, EntityDefinition, float]]:
        """Extract entities from text using ontology with domain-agnostic matching"""
        found_entities = []

        # Case-insensitive direct matching with word boundaries
        for name, entity_def in self.entities.items():
            # Create regex pattern with word boundaries
            pattern = re.compile(r'\b' + re.escape(name) + r'\b', re.IGNORECASE)
            if pattern.search(text):
                # Calculate confidence based on exact match vs alias
                confidence = 1.0 if name == entity_def.canonical_name else 0.9
                found_entities.append((name, entity_def, confidence))

        # Pattern-based extraction
        for entity_type, patterns in self.extraction_patterns.items():
            for pattern in patterns:
                try:
                    matches = re.finditer(pattern, text, re.IGNORECASE)
                    for match in matches:
                        matched_text = match.group(0)
                        # Check if we already found this entity through direct matching
                        if not any(e[0].lower() == matched_text.lower() for e in found_entities):
                            # Create temporary entity if not in ontology
                            if matched_text not in self.entities:
                                temp_entity = EntityDefinition(
                                    canonical_name=matched_text,
                                    aliases=[],
                                    entity_type=entity_type,
                                    attributes={},
                                    description=f"Pattern-detected {entity_type.value}"
                                )
                                found_entities.append((matched_text, temp_entity, 0.7))
                except re.error:
                    # Handle invalid regex patterns gracefully
                    continue

        # Remove duplicates and sort by confidence
        unique_entities = {}
        for text, entity, conf in found_entities:
            key = entity.canonical_name.lower()  # Case-insensitive deduplication
            if key not in unique_entities or unique_entities[key][2] < conf:
                unique_entities[key] = (text, entity, conf)

        return sorted(unique_entities.values(), key=lambda x: x[2], reverse=True)
    
    def extract_relationships(self, text: str, entities: List[Tuple[str, EntityDefinition, float]]) -> List[Dict]:
        relationships = []
        text_lower = text.lower()

        # Get entity names for pattern matching
        entity_names = [e[0] for e in entities]

        # Find co-occurrences within sentence distance
        sentences = re.split(r'(?<=[.!?])\s+', text)
        for sentence in sentences:
            sent_lower = sentence.lower()
            present_entities = [name for name in entity_names if name.lower() in sent_lower]

            if len(present_entities) < 2:
                continue

            # Check for all valid relationship combinations
            for i, source in enumerate(present_entities):
                for j, target in enumerate(present_entities):
                    if i == j:
                        continue

                    source_def = self.entities.get(source)
                    target_def = self.entities.get(target)

                    if not source_def or not target_def:
                        continue

                    # Find matching relationship definitions
                    valid_rels = [
                        rel for rel in self.relationships 
                        if (rel.source_type == source_def.entity_type and 
                            rel.target_type == target_def.entity_type)
                    ]

                    for rel_def in valid_rels:
                        # Add relationship even without explicit indicator
                        relationships.append({
                            "source": source_def.canonical_name,
                            "target": target_def.canonical_name,
                            "relationship": rel_def.relationship_type.value,
                            "confidence": 0.7,  # Medium confidence
                            "evidence": "co-occurrence",
                            "description": rel_def.description
                        })

        return relationships
    
    def enhance_extraction_prompt(self, base_prompt: str) -> str:
        """Enhance the Claude extraction prompt with ontology information"""
        entity_types = "\n".join([f"- {et.value}" for et in EntityType])
        relationship_types = "\n".join([f"- {rt.value}" for rt in RelationshipType])
        
        key_entities = "\n".join([
            f"- {name}: {defn.description}" 
            for name, defn in list(self.entities.items())[:20]  # Top 20 entities
            if name == defn.canonical_name  # Only canonical names
        ])
        
        enhanced_prompt = f"""{base_prompt}

DOMAIN-SPECIFIC ONTOLOGY FOR I2CONNECT TRAFFIC SAFETY PROJECT:

Entity Types:
{entity_types}

Relationship Types:
{relationship_types}

Key Domain Entities:
{key_entities}

EXTRACTION GUIDELINES:
1. Prioritize entities from the ontology above
2. Use canonical names when possible
3. Recognize aliases and variations
4. Focus on I2Connect project context
5. Pay attention to safety concepts, organizations, and technologies
6. Link entities through meaningful relationships

"""
        return enhanced_prompt
    
    def validate_extraction(self, extracted_entities: List[Dict], extracted_relationships: List[Dict]) -> Dict:
        """Validate extracted entities and relationships against ontology"""
        validation_results = {
            "valid_entities": [],
            "invalid_entities": [],
            "valid_relationships": [],
            "invalid_relationships": [],
            "suggestions": []
        }
        
        # Validate entities
        for entity in extracted_entities:
            entity_name = entity.get("name", "")
            if entity_name in self.entities:
                validation_results["valid_entities"].append(entity)
            else:
                validation_results["invalid_entities"].append(entity)
                # Suggest similar entities
                suggestions = self._find_similar_entities(entity_name)
                if suggestions:
                    validation_results["suggestions"].append({
                        "original": entity_name,
                        "suggestions": suggestions
                    })
        
        # Validate relationships
        for rel in extracted_relationships:
            source_name = rel.get("source", "")
            target_name = rel.get("target", "")
            rel_type = rel.get("relationship", "")
            
            if (source_name in self.entities and 
                target_name in self.entities and 
                self._is_valid_relationship(source_name, target_name, rel_type)):
                validation_results["valid_relationships"].append(rel)
            else:
                validation_results["invalid_relationships"].append(rel)
        
        return validation_results
    
    def _find_similar_entities(self, entity_name: str, threshold: float = 0.6) -> List[str]:
        """Find similar entities using simple string similarity"""
        from difflib import SequenceMatcher
        
        similarities = []
        for name, entity_def in self.entities.items():
            if name == entity_def.canonical_name:  # Only canonical names
                similarity = SequenceMatcher(None, entity_name.lower(), name.lower()).ratio()
                if similarity >= threshold:
                    similarities.append((name, similarity))
        
        return [name for name, _ in sorted(similarities, key=lambda x: x[1], reverse=True)[:3]]
    
    def _is_valid_relationship(self, source_name: str, target_name: str, rel_type: str) -> bool:
        """Check if a relationship is valid according to ontology"""
        if source_name not in self.entities or target_name not in self.entities:
            return False
        
        source_type = self.entities[source_name].entity_type
        target_type = self.entities[target_name].entity_type
        
        for rel_def in self.relationships:
            if (rel_def.source_type == source_type and 
                rel_def.target_type == target_type and 
                rel_def.relationship_type.value == rel_type):
                return True
        
        return False
    
    def export_schema(self) -> Dict:
        """Export ontology schema for Neo4j or other systems"""
        return {
            "entity_types": [et.value for et in EntityType],
            "relationship_types": [rt.value for rt in RelationshipType],
            "entities": {
                name: {
                    "type": defn.entity_type.value,
                    "aliases": defn.aliases,
                    "attributes": defn.attributes,
                    "description": defn.description
                }
                for name, defn in self.entities.items()
                if name == defn.canonical_name
            },
            "relationship_definitions": [
                {
                    "source_type": rel.source_type.value,
                    "target_type": rel.target_type.value,
                    "relationship_type": rel.relationship_type.value,
                    "description": rel.description
                }
                for rel in self.relationships
            ]
        }


# Usage example and integration
# Updated EnhancedEntityExtractor class for i2connect_ontology.py

class EnhancedEntityExtractor:
    """Enhanced entity extractor using I2Connect ontology"""
    
    def __init__(self, claude_extractor, config=None):
        self.claude_extractor = claude_extractor
        self.ontology = I2ConnectOntology()
        self.config = config
    
    def extract_with_ontology(self, text_chunk: str, source_doc: str) -> Dict:
        """Extract entities and relationships using ontology guidance"""
        
        # Step 1: Quick ontology-based extraction
        ontology_entities = self.ontology.extract_entities(text_chunk)
        
        # Step 2: Extract relationships from ontology entities
        ontology_relationships = self.ontology.extract_relationships(text_chunk, ontology_entities)
        
        # Step 3: Enhanced Claude prompt with ontology
        base_prompt = """Extract ALL entities and their relationships from this text comprehensively.

TEXT TO ANALYZE:
{text}

Focus on:
1. Organizations (companies, universities, research institutions)  
2. Technologies (systems, frameworks, concepts)
3. Safety concepts and risk types
4. People and their roles
5. Projects and deliverables

For each entity, provide:
- name: The entity name
- type: Entity type (Organization, Technology, SafetyConcept, RiskType, Person, etc.)
- properties: Additional attributes

For relationships, look for:
- Partnership relationships (works with, collaborates with, partners with)
- Development relationships (develops, creates, builds, implements)
- Usage relationships (uses, applies, employs)
- Contribution relationships (contributes to, leads, participates in)
- Risk relationships (addresses, mitigates, causes)

Return a JSON object with this structure:
{{
    "entities": [
        {{
            "name": "Entity Name",
            "type": "EntityType", 
            "properties": {{"key": "value"}}
        }}
    ],
    "relationships": [
        {{
            "source": "Source Entity",
            "target": "Target Entity",
            "relationship": "RELATIONSHIP_TYPE",
            "properties": {{"confidence": 0.8}}
        }}
    ]
}}

Be aggressive in finding relationships - if two entities appear in the same context, there's likely a relationship."""

        enhanced_prompt = self.ontology.enhance_extraction_prompt(base_prompt)
        
        # Step 4: Claude extraction with enhanced prompt
        try:
            claude_result = self.claude_extractor.generate(
                query=enhanced_prompt.format(text=text_chunk),
                context_sentences=[],
                system_prompt="Use the provided ontology to guide extraction. Return valid JSON only."
            )
            
            # Step 5: Parse Claude results
            claude_extracted = self._parse_claude_response(claude_result)
            
        except Exception as e:
            print(f"⚠️ Claude extraction failed: {e}")
            claude_extracted = {"entities": [], "relationships": []}
        
        # Step 6: Merge ontology and Claude results
        merged_entities = self._merge_entity_results(ontology_entities, claude_extracted.get("entities", []))
        merged_relationships = self._merge_relationship_results(ontology_relationships, claude_extracted.get("relationships", []))
        
        # Step 7: Validate against ontology
        validation = self.ontology.validate_extraction(merged_entities, merged_relationships)
        
        return {
            "entities": validation["valid_entities"] + validation["invalid_entities"],  # Include both for now
            "relationships": validation["valid_relationships"] + validation["invalid_relationships"],  # Include both for now
            "ontology_matches": len([e for e in merged_entities if e.get("source") == "ontology"]),
            "validation_suggestions": validation["suggestions"]
        }
    
    def _parse_claude_response(self, response: str) -> Dict:
        """Parse Claude response using JSON extraction"""
        import re
        
        response_text = response.strip()
        json_patterns = [
            r"```json\s*([\s\S]*?)\s*```",
            r"```\s*([\s\S]*?)\s*```", 
            r"\{[\s\S]*\}",
        ]

        for pattern in json_patterns:
            matches = re.findall(pattern, response_text)
            if matches:
                json_text = matches[0] if isinstance(matches[0], str) else matches[0]
                try:
                    return json.loads(json_text)
                except json.JSONDecodeError:
                    continue

        if "Based on" in response_text and "{" in response_text:
            start_idx = response_text.find("{")
            end_idx = response_text.rfind("}") + 1
            if start_idx >= 0 and end_idx > start_idx:
                json_text = response_text[start_idx:end_idx]
                try:
                    return json.loads(json_text)
                except json.JSONDecodeError:
                    pass

        return {"entities": [], "relationships": []}
    
    def _merge_entity_results(self, ontology_entities: List, claude_entities: List) -> List[Dict]:
        """Merge ontology and Claude entity results"""
        merged = []
        seen_entities = set()
        
        # Add ontology entities (high confidence)
        for text, entity_def, confidence in ontology_entities:
            canonical_name = entity_def.canonical_name.lower()
            if canonical_name not in seen_entities:
                merged.append({
                    "name": entity_def.canonical_name,
                    "type": entity_def.entity_type.value,
                    "confidence": confidence,
                    "source": "ontology",
                    "properties": entity_def.attributes
                })
                seen_entities.add(canonical_name)
        
        # Add Claude entities (if not already covered)
        for entity in claude_entities:
            entity_name = entity.get("name", "").lower()
            if entity_name and entity_name not in seen_entities:
                merged.append({
                    "name": entity.get("name"),
                    "type": entity.get("type", "Unknown"),
                    "confidence": 0.7,
                    "source": "claude",
                    "properties": entity.get("properties", {})
                })
                seen_entities.add(entity_name)
        
        return merged
    
    def _merge_relationship_results(self, ontology_relationships: List, claude_relationships: List) -> List[Dict]:
        """Merge ontology and Claude relationship results"""
        merged = []
        seen_relationships = set()
        
        # Add ontology relationships
        for rel in ontology_relationships:
            rel_key = f"{rel['source'].lower()}-{rel['target'].lower()}-{rel['relationship']}"
            if rel_key not in seen_relationships:
                merged.append(rel)
                seen_relationships.add(rel_key)
        
        # Add Claude relationships (if not duplicates)
        for rel in claude_relationships:
            source = rel.get("source", "").lower()
            target = rel.get("target", "").lower()
            relationship = rel.get("relationship", "")
            rel_key = f"{source}-{target}-{relationship}"
            
            if rel_key not in seen_relationships and source and target:
                merged.append(rel)
                seen_relationships.add(rel_key)
        
        return merged