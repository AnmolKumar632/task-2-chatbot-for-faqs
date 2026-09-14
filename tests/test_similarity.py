"""Tests for the TF-IDF + cosine similarity matching engine."""

from pathlib import Path

import numpy as np
import pytest

from nlp.similarity_engine import SimilarityEngine
from services.chatbot_service import load_faqs

FAQ_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "faqs.json"


@pytest.fixture(scope="module")
def faqs():
    return load_faqs(FAQ_DATA_PATH)


@pytest.fixture(scope="module")
def engine(faqs):
    return SimilarityEngine(faqs)


def test_engine_initializes_with_faq_data(engine, faqs):
    assert engine.document_count == len(faqs)


def test_faq_matrix_contains_expected_document_count(engine, faqs):
    from sklearn.feature_extraction.text import TfidfVectorizer

    assert engine.document_count == 30
    assert isinstance(engine._vectorizer, TfidfVectorizer)


def test_user_query_returns_best_match_structure(engine):
    result = engine.find_best_match("How can I reset my password?")
    assert "faq" in result
    assert "score" in result
    assert isinstance(result["score"], float)


def test_exact_faq_question_matches_itself(engine):
    result = engine.find_best_match("How can I reset my password?")
    assert result["faq"]["id"] == 1
    assert result["score"] == pytest.approx(1.0, abs=1e-9)


def test_paraphrased_question_matches_faq(engine):
    result = engine.find_best_match("I forgot my password. How can I change it?")
    assert result["faq"]["id"] == 1
    assert 0.0 < result["score"] <= 1.0


def test_second_paraphrase_matches_faq(engine):
    result = engine.find_best_match("Where can I change my password?")
    assert result["faq"]["id"] == 1


def test_download_certificate_query_matches_certificate_faq(engine):
    result = engine.find_best_match("How do I download my certificate?")
    assert result["faq"]["id"] == 22


def test_unrelated_query_does_not_fabricate_match(engine):
    result = engine.find_best_match("What is the weather today?")
    assert isinstance(result["score"], float)
    assert result["score"] == 0.0
    assert result["faq"] is None


def test_out_of_vocabulary_query_is_safe(engine):
    result = engine.find_best_match("xyzabc123 unknownterm")
    assert result["faq"] is None
    assert result["score"] == 0.0


def test_empty_query_is_handled(engine):
    result = engine.find_best_match("")
    assert result["faq"] is None
    assert result["score"] == 0.0


def test_whitespace_query_is_handled(engine):
    result = engine.find_best_match("   ")
    assert result["faq"] is None
    assert result["score"] == 0.0


@pytest.mark.parametrize("bad_input", [None, 123, [], {}])
def test_invalid_input_raises_type_error(engine, bad_input):
    with pytest.raises(TypeError):
        engine.find_best_match(bad_input)


def test_empty_dataset_raises():
    with pytest.raises(ValueError):
        SimilarityEngine([])


def test_scores_are_numeric_and_in_range(engine):
    scores = engine.score_questions("How can I reset my password?")
    assert len(scores) == engine.document_count
    for score in scores:
        assert isinstance(score, (float, np.floating))
        assert 0.0 <= float(score) <= 1.0 + 1e-9


def test_engine_is_deterministic(engine):
    query = "I forgot my password. How can I change it?"
    first = engine.find_best_match(query)
    second = engine.find_best_match(query)
    assert first["faq"]["id"] == second["faq"]["id"]
    assert first["score"] == second["score"]


@pytest.mark.parametrize(
    "query",
    [
        "How can I reset my password?",
        "I forgot my password. How can I change it?",
        "What payment options do you support?",
        "What is the weather today?",
        "Tell me a joke",
    ],
)
def test_processed_path_matches_raw_path(engine, query):
    from nlp.text_processor import preprocess_text

    raw = engine.find_best_match(query)
    processed = engine.find_best_match_processed(preprocess_text(query))
    assert processed == raw


def test_processed_path_handles_empty_text(engine):
    result = engine.find_best_match_processed("")
    assert result["faq"] is None
    assert result["score"] == 0.0