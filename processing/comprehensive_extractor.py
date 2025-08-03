"""Comprehensive multi-pass knowledge extraction utilities.

This module implements a strategy that performs structure analysis, semantic
chunking and completeness validation to ensure that documents are fully
processed. It relies on an LLM-like ``ClaudeQA`` interface but can be easily
mocked for tests.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import nltk
from backend.qa_models import ClaudeQA


@dataclass
class ExtractionResult:
    """Container for entities and relationships."""

    entities: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]


class ComprehensiveExtractor:
    """High level extractor implementing a multi-pass strategy.

    The class is intentionally lightweight.  All calls to the underlying LLM are
    routed through ``claude_extractor`` which exposes a ``generate`` method.  In
    tests this object can be replaced with a simple stub.
    """

    def __init__(self, claude_extractor: Optional[ClaudeQA] = None) -> None:
        self.claude_extractor = claude_extractor or ClaudeQA()
        # default prompt used when no specialised prompt is available
        self.default_extraction_prompt = "Extract entities and relationships as JSON"
        self._ensure_nltk()

    # ------------------------------------------------------------------
    # Helper utilities
    def extract_json_from_claude_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Return JSON from a Claude response dictionary.

        ClaudeQA returns a dictionary with an ``answer`` field that may contain a
        JSON blob.  This helper safely parses the JSON and falls back to an empty
        dictionary on error.
        """

        text = response.get("answer", "") if isinstance(response, dict) else ""
        try:
            return json.loads(text)
        except Exception:
            return {}

    def _ensure_nltk(self) -> None:
        """Ensure required NLTK data packages are available."""

        packages = [
            "punkt",
            "punkt_tab",
            "averaged_perceptron_tagger",
            "averaged_perceptron_tagger_eng",
            "maxent_ne_chunker",
            "maxent_ne_chunker_tab",
            "words",
        ]
        for pkg in packages:
            nltk.download(pkg, quiet=True)

    # ------------------------------------------------------------------
    # NLTK based entity utilities ---------------------------------------------
    def _looks_like_organization(
        self, name: str, prev_token: Optional[str], next_token: Optional[str]
    ) -> bool:
        """Return True if the span likely refers to an organisation."""

        keywords = {
            "university",
            "institute",
            "technologies",
            "technology",
            "tech",
            "solutions",
            "systems",
            "company",
            "corporation",
            "corp",
            "inc",
            "ltd",
            "gmbh",
            "ab",
            "group",
            "centre",
            "center",
            "college",
            "lab",
            "labs",
        }
        known_orgs = {"scania", "smart eye", "viscando technologies", "viscando"}

        lower_name = name.lower()
        if any(k in lower_name for k in keywords) or lower_name in known_orgs:
            return True
        for tok in (prev_token, next_token):
            if tok and tok.lower() in keywords:
                return True
        return False

    def map_nltk_entity_type(self, label: str) -> Optional[str]:
        """Map NLTK NE labels to the extractor schema."""

        mapping = {
            "ORGANIZATION": "Organization",
            "PERSON": "Person",
            "GPE": "Location",
            "LOCATION": "Location",
            "FACILITY": "Organization",
        }
        return mapping.get(label)

    def extract_entities_nltk(self, text: str) -> List[Dict[str, Any]]:
        """Use NLTK's NER to detect entities in ``text``."""

        tokens = nltk.word_tokenize(text)
        tagged = nltk.pos_tag(tokens)
        tree = nltk.ne_chunk(tagged)
        flat = list(tree)
        entities: List[Dict[str, Any]] = []
        i = 0
        while i < len(flat):
            node = flat[i]
            if isinstance(node, nltk.Tree):
                label = node.label()
                name_tokens = [tok for tok, _ in node.leaves()]
                j = i + 1
                # extend span with following capitalised tokens or patterns like
                # "of <Capitalised>"
                while j < len(flat):
                    nxt = flat[j]
                    if isinstance(nxt, tuple):
                        word, pos = nxt
                        if word.lower() in {"of", "for", "the"}:
                            if j + 1 < len(flat):
                                nxt2 = flat[j + 1]
                                if isinstance(nxt2, nltk.Tree) or (
                                    isinstance(nxt2, tuple)
                                    and nxt2[0][0].isupper()
                                ):
                                    name_tokens.append(word)
                                    j += 1
                                    continue
                            break
                        if word[0].isupper() and re.match(r"NNP", pos):
                            name_tokens.append(word)
                            j += 1
                            continue
                        break
                    elif isinstance(nxt, nltk.Tree):
                        name_tokens.extend(tok for tok, _ in nxt.leaves())
                        j += 1
                        continue
                    else:
                        break

                name = " ".join(name_tokens)
                prev_word = (
                    flat[i - 1][0] if i > 0 and isinstance(flat[i - 1], tuple) else None
                )
                next_word = (
                    flat[j][0] if j < len(flat) and isinstance(flat[j], tuple) else None
                )
                mapped = self.map_nltk_entity_type(label) or ""
                if mapped != "Organization" and self._looks_like_organization(
                    name, prev_word, next_word
                ):
                    mapped = "Organization"
                if mapped:
                    entities.append({"name": name, "type": mapped})
                i = j
            else:
                i += 1
        return entities

    # ------------------------------------------------------------------
    # Pass 1: document structure -------------------------------------------------
    def analyze_document_structure(self, document_content: str) -> Dict[str, Any]:
        """Ask the LLM to analyse the document structure.

        Only the first couple of thousand characters are required for structure
        analysis which keeps the prompt small.
        """

        structure_prompt = (
            """
            Analyze this document structure and identify ALL key sections that contain entities and relationships.\n\n"""
            "Return comprehensive section analysis as JSON."
        )

        query = structure_prompt + f"\nDocument: {document_content[:2000]}"
        result = self.claude_extractor.generate(
            query=query,
            system_prompt=(
                "You are an expert at analyzing document structure and identifying critical information sections."
            ),
        )
        return self.extract_json_from_claude_response(result)

    # ------------------------------------------------------------------
    # Pass 2: section specific extraction ---------------------------------------
    def extract_entities_with_specialized_prompt(self, text: str, prompt: str, source: str) -> ExtractionResult:
        """Helper calling the LLM with a specialised prompt."""

        query = f"{prompt}\n\n{text}"
        response = self.claude_extractor.generate(query=query, system_prompt=source)
        data = self.extract_json_from_claude_response(response)
        return ExtractionResult(data.get("entities", []), data.get("relationships", []))

    def extract_by_section_type(self, text_chunk: str, section_type: str, source_doc: str) -> ExtractionResult:
        """Use specialised prompts depending on section type."""

        section_prompts = {
            "organizational_info": (
                "CRITICAL: Extract ALL organizational entities and partnerships from this text."\
                " Focus on partner organizations, consortium members and affiliations."
            ),
            "technical_content": (
                "Extract technical entities and their relationships such as technologies, systems or methodologies."
            ),
            "project_details": (
                "Extract project related entities: project names, objectives and timelines."
            ),
        }
        base_prompt = section_prompts.get(section_type, self.default_extraction_prompt)
        return self.extract_entities_with_specialized_prompt(text_chunk, base_prompt, source_doc)

    # default extraction path
    def extract_entities_and_relationships(
        self, text_chunk: str, source_doc: str, section_type: str = "general"
    ) -> ExtractionResult:
        return self.extract_by_section_type(text_chunk, section_type, source_doc)

    def extract_entities_forced(self, chunk: Dict[str, Any]) -> ExtractionResult:
        """Always attempt extraction even if chunk is large or repetitive."""

        return self.extract_entities_and_relationships(
            chunk["content"], chunk.get("source", ""), chunk.get("type", "general")
        )

    # ------------------------------------------------------------------
    # Pass 3: validation ---------------------------------------------------------
    def validate_extraction_completeness(self, document_content: str, extracted_entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Ask the model to cross check the extraction for missing information."""

        validation_prompt = (
            "Compare this document content with the extracted entities to identify any MISSING critical information."\
            " Return analysis as JSON."
        )
        query = validation_prompt + f"\nDocument content: {document_content[:3000]}\nExtracted entities: {json.dumps(extracted_entities)[:1000]}"
        response = self.claude_extractor.generate(
            query=query,
            system_prompt="You are an expert at validating information extraction completeness.",
        )
        return self.extract_json_from_claude_response(response)

    # ------------------------------------------------------------------
    # Semantic aware chunking ----------------------------------------------------
    def identify_document_sections(self, document_content: str, markers: Iterable[str]) -> List[Dict[str, Any]]:
        """Very small helper that identifies sections based on heading markers."""

        lines = document_content.splitlines()
        sections: List[Dict[str, Any]] = []
        current: Dict[str, Any] = {"content": []}

        for line in lines:
            matched_type: Optional[str] = None
            for pat in markers:
                if re.match(pat, line, flags=re.IGNORECASE):
                    # close previous
                    if current.get("content"):
                        sections.append(
                            {
                                "content": "\n".join(current["content"]),
                                "type": current.get("type", "general"),
                            }
                        )
                    current = {"content": [], "type": self._classify_section(line)}
                    matched_type = current["type"]
                    break
            if matched_type is None:
                current.setdefault("content", []).append(line)
        if current.get("content"):
            sections.append(
                {"content": "\n".join(current["content"]), "type": current.get("type", "general")}
            )
        if not sections:
            sections.append({"content": document_content, "type": "general"})
        return sections

    def _classify_section(self, heading: str) -> str:
        """Map a heading to a known section type."""

        h = heading.lower()
        if any(k in h for k in ["collaborators", "partners", "consortium", "team", "organizations"]):
            return "organizational_info"
        if any(k in h for k in ["project", "system", "architecture", "components"]):
            return "technical_content"
        if any(k in h for k in ["methodology", "approach", "framework"]):
            return "technical_content"
        if any(k in h for k in ["results", "conclusions"]):
            return "project_details"
        return "general"

    def context_aware_split(self, content: str, section_type: str) -> List[Dict[str, Any]]:
        """Split large sections while preserving context."""

        limit = 2000
        if len(content) <= limit:
            return [{"content": content, "type": section_type, "priority": "high"}]
        chunks: List[Dict[str, Any]] = []
        start = 0
        while start < len(content):
            end = start + limit
            chunk_text = content[start:end]
            chunks.append({"content": chunk_text, "type": section_type, "priority": "high"})
            start = end
        return chunks

    def intelligent_semantic_chunking(self, document_content: str) -> List[Dict[str, Any]]:
        """Chunk document while preserving semantic sections."""

        section_markers = [
            r"##?\s*\d*\.?\s*(Collaborators?|Partners?|Consortium|Team|Organizations?)",
            r"##?\s*\d*\.?\s*(Project|System|Architecture|Components?)",
            r"##?\s*\d*\.?\s*(Methodology|Approach|Framework)",
            r"##?\s*\d*\.?\s*(Results?|Findings?|Conclusions?)",
            r"##?\s*\d*\.?\s*(References?|Bibliography|Citations?)",
        ]
        sections = self.identify_document_sections(document_content, section_markers)
        intelligent_chunks: List[Dict[str, Any]] = []
        for section in sections:
            if section["type"] == "organizational_info":
                intelligent_chunks.append(
                    {"content": section["content"], "type": section["type"], "priority": "critical"}
                )
            elif len(section["content"]) > 2000:
                intelligent_chunks.extend(self.context_aware_split(section["content"], section["type"]))
            else:
                intelligent_chunks.append(
                    {"content": section["content"], "type": section["type"], "priority": "medium"}
                )
        return intelligent_chunks

    # ------------------------------------------------------------------
    # Priority based processing -----------------------------------------------
    def process_chunks_by_priority(self, chunks: List[Dict[str, Any]]) -> ExtractionResult:
        """Process critical chunks first and accumulate results."""

        priority_order = ["critical", "high", "medium", "low"]
        sorted_chunks = sorted(
            chunks, key=lambda x: priority_order.index(x.get("priority", "medium"))
        )
        entities: List[Dict[str, Any]] = []
        relationships: List[Dict[str, Any]] = []
        for chunk in sorted_chunks:
            result = self.extract_entities_and_relationships(
                chunk["content"], chunk.get("source", ""), chunk.get("type", "general")
            )
            entities.extend(result.entities)
            relationships.extend(result.relationships)
        return ExtractionResult(entities, relationships)

    # ------------------------------------------------------------------
    # Relationship utilities -----------------------------------------------------
    def check_existing_relationship(
        self, source: str, target: str, relationships: List[Dict[str, Any]]
    ) -> bool:
        return any(
            r.get("source") == source and r.get("target") == target
            for r in relationships
        )

    def infer_relationship_type(self, organization: Dict[str, Any], project: Dict[str, Any]) -> str:
        org_name = organization.get("name", "").lower()
        if "university" in org_name or "academic" in org_name:
            return "ACADEMIC_PARTNER"
        if any(word in org_name for word in ["company", "corp", "ltd", "inc"]):
            return "INDUSTRIAL_PARTNER"
        if any(word in org_name for word in ["technology", "tech", "systems"]):
            return "TECHNOLOGY_PARTNER"
        return "COLLABORATES_WITH"

    def build_project_centric_relationships(
        self,
        entities: List[Dict[str, Any]],
        existing_relationships: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        existing_relationships = existing_relationships or []
        main_projects = [e for e in entities if e.get("type") == "Project"]
        organizations = [e for e in entities if e.get("type") == "Organization"]
        additional: List[Dict[str, Any]] = []
        for org in organizations:
            for project in main_projects:
                if not self.check_existing_relationship(
                    org.get("name"), project.get("name"), existing_relationships
                ):
                    rel = {
                        "source": org.get("name"),
                        "target": project.get("name"),
                        "relationship": self.infer_relationship_type(org, project),
                        "properties": {"inferred": "true", "confidence": "high"},
                    }
                    additional.append(rel)
        return additional

    # ------------------------------------------------------------------
    # Completeness validation helpers -------------------------------------------
    def extract_missing_organizations(
        self, missing_orgs: Iterable[str], document_content: str
    ) -> List[Dict[str, Any]]:
        """Return entity stubs for missing organisations."""

        entities = []
        for name in missing_orgs:
            entities.append({"name": name, "type": "Organization"})
        return entities

    def validate_organizational_coverage(
        self, document_content: str, extracted_entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Ensure all mentioned organisations appear in the extracted set."""

        nltk_entities = self.extract_entities_nltk(document_content)
        mentioned = {
            e["name"] for e in nltk_entities if e.get("type") == "Organization"
        }
        all_extracted = {e.get("name") for e in extracted_entities}
        missing = mentioned - all_extracted
        if missing:
            return self.extract_missing_organizations(missing, document_content)
        return []

    # ------------------------------------------------------------------
    def comprehensive_knowledge_extraction(self, documents: Iterable[Any]) -> Dict[str, Any]:
        """Main orchestration method implementing the multi-pass strategy."""

        all_entities: List[Dict[str, Any]] = []
        all_relationships: List[Dict[str, Any]] = []

        for doc in documents:
            content = getattr(doc, "content", "")
            source = getattr(getattr(doc, "meta", {}), "get", lambda k, d=None: d)("source", "document")

            # Step 1: analyse structure (result unused but stored for completeness)
            self.analyze_document_structure(content)

            # Step 2: chunking
            chunks = self.intelligent_semantic_chunking(content)

            # Step 3: extraction
            result = self.process_chunks_by_priority(chunks)

            # Step 4: completeness validation
            self.validate_extraction_completeness(content, result.entities)

            # Step 5: organisational coverage check
            missing_entities = self.validate_organizational_coverage(content, result.entities)
            result.entities.extend(missing_entities)

            # Step 6: relationship building
            additional = self.build_project_centric_relationships(
                result.entities, result.relationships
            )
            result.relationships.extend(additional)

            all_entities.extend(result.entities)
            all_relationships.extend(result.relationships)

        return {"entities": all_entities, "relationships": all_relationships}

