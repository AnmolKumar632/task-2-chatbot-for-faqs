"""Lightweight performance / initialization smoke tests.

These verify that expensive work (NLTK preprocessing of the FAQ corpus and
TF-IDF construction) happens once at startup and that repeated requests do
not rebuild the vectorizer. No full benchmarking framework is introduced.

Timing bounds are deliberately generous so the tests stay deterministic on
slow machines; they exist only to catch pathological regressions.
"""

import time

import pytest

from nlp import similarity_engine
from nlp.similarity_engine import SimilarityEngine


@pytest.fixture(scope="module")
def engine(faqs):
    return SimilarityEngine(faqs)


def test_faq_corpus_is_preprocessed_once_at_initialization(faqs, monkeypatch):
    calls = []

    original = similarity_engine.preprocess_text

    def counting_preprocess(text):
        calls.append(text)
        return original(text)

    monkeypatch.setattr(similarity_engine, "preprocess_text", counting_preprocess)

    engine = SimilarityEngine(faqs)
    corpus_calls = len(calls)
    assert corpus_calls == len(faqs)

    engine.find_best_match("How can I reset my password?")
    engine.find_best_match("How do I submit an assignment?")
    query_calls = len(calls) - corpus_calls
    assert query_calls == 2


def test_vectorizer_and_matrix_are_reused_across_queries(engine):
    vectorizer = engine._vectorizer
    matrix = engine._faq_matrix
    for query in (
        "How can I reset my password?",
        "I forgot my password",
        "What payment options do you support?",
        "Tell me a joke",
    ):
        engine.find_best_match(query)
    assert engine._vectorizer is vectorizer
    assert engine._faq_matrix is matrix


def test_same_query_returns_identical_best_match_each_time(engine):
    query = "Where can I download my certificate?"
    results = [engine.find_best_match(query) for _ in range(5)]
    assert all(r["score"] == results[0]["score"] for r in results)
    assert all(r["faq"]["id"] == results[0]["faq"]["id"] for r in results)


def test_get_response_preprocesses_accepted_query_exactly_once(
    chatbot_service, monkeypatch
):
    import services.chatbot_service as service_module
    from nlp import similarity_engine

    calls = []

    original = service_module.preprocess_tokens

    def counting_preprocess(text):
        calls.append(text)
        return original(text)

    engine_calls = []

    original_engine_preprocess = similarity_engine.preprocess_text

    def counting_engine_preprocess(text):
        engine_calls.append(text)
        return original_engine_preprocess(text)

    monkeypatch.setattr(service_module, "preprocess_tokens", counting_preprocess)
    monkeypatch.setattr(similarity_engine, "preprocess_text", counting_engine_preprocess)

    response = chatbot_service.get_response(
        "I forgot my password. How can I change it?"
    )

    assert response["matched"] is True
    assert response["faq"]["id"] == 1
    # One NLTK pass for the token-count rule, reused by the engine: the
    # query is preprocessed once, never twice.
    assert len(calls) == 1
    assert len(engine_calls) == 0


def test_get_response_preprocesses_rejected_short_query_exactly_once(
    chatbot_service, monkeypatch
):
    import services.chatbot_service as service_module

    calls = []

    original = service_module.preprocess_tokens

    def counting_preprocess(text):
        calls.append(text)
        return original(text)

    monkeypatch.setattr(service_module, "preprocess_tokens", counting_preprocess)

    response = chatbot_service.get_response("password")

    assert response["matched"] is False
    assert len(calls) == 1


def test_repeated_api_requests_are_stable_and_prompt(client):
    payload = {"question": "How can I reset my password?"}
    responses = []
    start = time.monotonic()
    for _ in range(30):
        response = client.post("/api/chat", json=payload)
        assert response.status_code == 200
        responses.append(response.get_json())
    elapsed = time.monotonic() - start

    assert all(r == responses[0] for r in responses)
    assert elapsed < 10.0, f"30 chat requests took {elapsed:.2f}s"