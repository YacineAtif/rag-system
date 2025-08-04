"""Tests for natural language query parsing for the knowledge graph."""

from processing.knowledge_graph import (
    _classify_query,
    _extract_entity_name,
    _graph_query_weight,
)


def test_extract_entity_name_role_question():
    assert _extract_entity_name("what is Scania role?") == "Scania"


def test_extract_entity_name_partners_question():
    text = "who are the partners of Smart Eye?"
    assert _extract_entity_name(text) == "Smart Eye"


def test_extract_entity_name_partnerships_does_have():
    text = "what partnerships does Scania have"
    assert _extract_entity_name(text) == "Scania"


def test_extract_entity_name_direct():
    assert _extract_entity_name("University of Skövde") == "University of Skövde"


def test_extract_entity_name_simple_what_is():
    assert _extract_entity_name("what is evidence theory") == "evidence theory"


def test_extract_entity_name_numbers():
    assert _extract_entity_name("what is concept 1") == "concept 1"


def test_extract_entity_name_preposition():
    assert _extract_entity_name("how is applied in I2Connect") == "I2Connect"



def test_extract_entity_name_who_is():
    assert _extract_entity_name("who is Alan Turing?") == "Alan Turing"


def test_extract_entity_name_where_is():
    text = "where is the University of Skövde?"
    assert _extract_entity_name(text) == "University of Skövde"


def test_extract_entity_name_when_is():
    assert _extract_entity_name("when is the conference?") == "conference"


def test_extract_entity_name_which_organization():
    text = "which organization developed the Transformer model?"
    assert _extract_entity_name(text) == "developed the Transformer model"


def test_extract_entity_name_how_does_relate():
    text = "how does the University of Skövde relate to Smart Eye?"
    assert _extract_entity_name(text) == "University of Skövde"


def test_extract_entity_name_how_does_work():
    assert _extract_entity_name("how does blockchain work?") == "blockchain"


def test_classify_role_as_graph():
    assert _classify_query("what is Smart Eye role?") == "graph"

def test_graph_query_weight_role_question():
    assert _graph_query_weight("what is Smart Eye role?") > 1.0


