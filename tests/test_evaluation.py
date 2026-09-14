"""Integrity and regression tests for the matching-quality evaluation dataset.

These tests validate that tests/data/matching_evaluation.json is consistent
with the production FAQ dataset and assert stable, behavior-based expectations:

    * every expected FAQ id exists in the real dataset
    * exact queries match their FAQ exactly
    * vocabulary queries resolve to the expected FAQ
    * short / unrelated / OOV queries are rejected (no false positives)
"""

import json
from pathlib import Path

import pytest

from services.chatbot_service import FAQChatbot, load_faqs

BASE_DIR = Path(__file__).resolve().parents[1]
FAQ_DATA_PATH = BASE_DIR / "data" / "faqs.json"
EVALUATION_DATA_PATH = Path(__file__).resolve().parent / "data" / "matching_evaluation.json"

ALLOWED_TYPES = {"exact", "paraphrase", "vocabulary", "short", "ambiguous", "unrelated", "oov"}


@pytest.fixture(scope="module")
def evaluation_cases():
    with open(EVALUATION_DATA_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


@pytest.fixture(scope="module")
def faqs():
    return load_faqs(FAQ_DATA_PATH)


@pytest.fixture(scope="module")
def chatbot_service(faqs):
    return FAQChatbot(faqs, threshold=0.35)


# --- Evaluation dataset integrity ---


def test_evaluation_file_exists():
    assert EVALUATION_DATA_PATH.is_file()


def test_evaluation_cases_have_required_fields(evaluation_cases):
    assert isinstance(evaluation_cases, list)
    assert len(evaluation_cases) > 0
    for case in evaluation_cases:
        assert set(case) >= {"query", "expected_faq_id", "type"}, case


def test_evaluation_queries_are_non_empty_unique_strings(evaluation_cases):
    queries = [case["query"] for case in evaluation_cases]
    assert all(isinstance(q, str) and q.strip() for q in queries)
    assert len(queries) == len(set(queries))


def test_evaluation_types_are_known(evaluation_cases):
    for case in evaluation_cases:
        assert case["type"] in ALLOWED_TYPES, case["type"]


def test_expected_faq_ids_exist_in_dataset(evaluation_cases, faqs):
    faq_ids = {faq["id"] for faq in faqs}
    for case in evaluation_cases:
        expected = case["expected_faq_id"]
        if expected is not None:
            assert expected in faq_ids, f"expected_faq_id {expected} is not a real FAQ"


def test_evaluation_dataset_covers_multiple_query_types(evaluation_cases):
    types = {case["type"] for case in evaluation_cases}
    assert {"exact", "paraphrase", "unrelated"} <= types


# --- Behavior-based regression assertions ---


def test_exact_queries_match_their_faq_id_exactly(chatbot_service, evaluation_cases):
    for case in evaluation_cases:
        if case["type"] != "exact":
            continue
        response = chatbot_service.get_response(case["query"])
        assert response["matched"] is True, case["query"]
        assert response["faq"]["id"] == case["expected_faq_id"], case["query"]
        assert response["score"] == pytest.approx(1.0, abs=1e-3), case["query"]


def test_vocabulary_queries_resolve_to_expected_faq(chatbot_service, evaluation_cases):
    for case in evaluation_cases:
        if case["type"] != "vocabulary":
            continue
        response = chatbot_service.get_response(case["query"])
        assert response["matched"] is True, case["query"]
        assert response["faq"]["id"] == case["expected_faq_id"], case["query"]


def test_unrelated_queries_are_rejected(chatbot_service, evaluation_cases):
    for case in evaluation_cases:
        if case["type"] != "unrelated":
            continue
        response = chatbot_service.get_response(case["query"])
        assert response["matched"] is False, f"false positive: {case['query']!r}"
        assert response["faq"] is None
        assert response["message"] == chatbot_service.fallback_message


def test_short_queries_are_rejected(chatbot_service, evaluation_cases):
    for case in evaluation_cases:
        if case["type"] != "short":
            continue
        response = chatbot_service.get_response(case["query"])
        assert response["matched"] is False, f"false positive: {case['query']!r}"
        assert response["score"] == 0.0


def test_oov_queries_are_rejected(chatbot_service, evaluation_cases):
    for case in evaluation_cases:
        if case["type"] != "oov":
            continue
        response = chatbot_service.get_response(case["query"])
        assert response["matched"] is False, f"false positive: {case['query']!r}"
        assert response["score"] == 0.0


def test_all_expected_rejections_are_rejected(chatbot_service, evaluation_cases):
    for case in evaluation_cases:
        if case["expected_faq_id"] is not None:
            continue
        response = chatbot_service.get_response(case["query"])
        assert response["matched"] is False, f"false positive: {case['query']!r}"


def test_all_exact_and_vocabulary_cases_are_accepted(chatbot_service, evaluation_cases):
    for case in evaluation_cases:
        if case["type"] not in {"exact", "vocabulary"}:
            continue
        response = chatbot_service.get_response(case["query"])
        assert response["matched"] is True, f"false negative: {case['query']!r}"