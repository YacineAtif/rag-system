"""Tests for natural language query parsing for the knowledge graph."""

from processing.knowledge_graph import _extract_entity_name, _classify_query


def test_extract_entity_name_role_question():
    assert _extract_entity_name("what is Scania role?") == "Scania"


def test_extract_entity_name_partners_question():
    text = "who are the partners of Smart Eye?"
    assert _extract_entity_name(text) == "Smart Eye"


def test_extract_entity_name_direct():
    assert _extract_entity_name("University of Skövde") == "University of Skövde"


def test_extract_entity_name_simple_what_is():
    assert _extract_entity_name("what is evidence theory") == "evidence theory"


def test_extract_entity_name_numbers():
    assert _extract_entity_name("what is concept 1") == "concept 1"


def test_extract_entity_name_preposition():
    assert _extract_entity_name("how is applied in I2Connect") == "I2Connect"


def test_classify_role_as_graph():
    assert _classify_query("what is Smart Eye role?") == "graph"

