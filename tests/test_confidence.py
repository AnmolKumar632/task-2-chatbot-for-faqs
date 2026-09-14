"""Tests for the confidence threshold and fallback behavior."""

from pathlib import Path

import pytest

from services.chatbot_service import (
    DEFAULT_FALLBACK_MESSAGE,
    FAQChatbot,
    is_confident_match,
    load_faqs,
    normalize_threshold,
)

FAQ_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "faqs.json"
FAQ_ANSWERS = [faq["answer"] for faq in load_faqs(FAQ_DATA_PATH)]


@pytest.fixture(scope="module")
def faqs():
    return load_faqs(FAQ_DATA_PATH)


@pytest.fixture(scope="module")
def chatbot(faqs):
    return FAQChatbot(faqs)


# --- Accept / reject decisions ---


def test_related_query_is_accepted(chatbot):
    response = chatbot.get_response("I forgot my password. How can I change it?")
    assert response["matched"] is True
    assert response["faq"]["id"] == 1
    assert response["message"] == response["faq"]["answer"]


def test_weakly_related_query_is_accepted(chatbot):
    response = chatbot.get_response("I can't remember my login password")
    assert response["matched"] is True
    assert response["faq"]["id"] == 1


def test_exact_match_is_accepted(chatbot):
    response = chatbot.get_response("How can I reset my password?")
    assert response["matched"] is True
    assert response["faq"]["id"] == 1
    assert response["score"] == pytest.approx(1.0, abs=1e-3)


def test_get_response_is_deterministic(chatbot):
    query = "I forgot my password. How can I change it?"
    first = chatbot.get_response(query)
    second = chatbot.get_response(query)
    assert first["matched"] == second["matched"]
    assert first["faq"] == second["faq"]
    assert first["score"] == second["score"]
    assert first["message"] == second["message"]


def test_accepted_response_structure(chatbot):
    response = chatbot.get_response("How can I reset my password?")
    assert {"matched", "faq", "score", "message"} <= set(response)
    assert response["matched"] is True
    assert response["faq"] is not None
    assert isinstance(response["score"], float)
    assert isinstance(response["message"], str)


def test_rejected_response_structure(chatbot):
    response = chatbot.get_response("What is the weather today?")
    assert {"matched", "faq", "score", "message"} <= set(response)
    assert response["matched"] is False
    assert response["faq"] is None
    assert response["score"] == 0.0


def test_unrelated_query_is_rejected(chatbot):
    response = chatbot.get_response("What is the weather today?")
    assert response["matched"] is False
    assert response["faq"] is None
    assert response["message"] == chatbot.fallback_message


# --- Threshold boundary ---


def test_below_threshold_rejected():
    assert is_confident_match(0.34, 0.35) is False


def test_equal_to_threshold_accepted():
    assert is_confident_match(0.35, 0.35) is True


def test_above_threshold_accepted():
    assert is_confident_match(0.36, 0.35) is True


@pytest.mark.parametrize("bad_threshold", [-0.1, 1.1, 1.5, "abc", None])
def test_invalid_threshold_raises(bad_threshold):
    with pytest.raises(ValueError):
        normalize_threshold(bad_threshold)


@pytest.mark.parametrize("good_threshold", [0.0, 0.25, 0.35, 0.5, 1.0])
def test_valid_threshold_accepted(good_threshold):
    assert normalize_threshold(good_threshold) == pytest.approx(good_threshold)


def test_custom_strict_threshold_rejects_weak_matches(faqs):
    strict_bot = FAQChatbot(faqs, threshold=1.0)
    weak = strict_bot.get_response("I forgot my password. How can I change it?")
    assert weak["matched"] is False


def test_custom_fallback_message_is_used(faqs):
    custom = "Custom fallback message."
    bot = FAQChatbot(faqs, fallback_message=custom)
    response = bot.get_response("What is the weather today?")
    assert response["message"] == custom


# --- Input validation ---


@pytest.mark.parametrize("empty_input", ["", "   ", "\t\n"])
def test_empty_input_raises(chatbot, empty_input):
    with pytest.raises(ValueError):
        chatbot.get_response(empty_input)


@pytest.mark.parametrize("bad_input", [None, 123, [], {}])
def test_invalid_input_type_raises(chatbot, bad_input):
    with pytest.raises(TypeError):
        chatbot.get_response(bad_input)


# --- Safety ---


def test_no_hallucination_on_unrelated_query(chatbot):
    response = chatbot.get_response("What is the weather today?")
    assert response["message"] not in FAQ_ANSWERS


def test_zero_score_query_returns_fallback(chatbot):
    response = chatbot.get_response("Tell me a joke.")
    assert response["matched"] is False
    assert response["faq"] is None
    assert response["score"] == 0.0


@pytest.mark.parametrize("short_query", ["certificate", "refund", "password", "code", "payment"])
def test_single_token_query_is_rejected(chatbot, short_query):
    response = chatbot.get_response(short_query)
    assert response["matched"] is False
    assert response["faq"] is None
    assert response["score"] == 0.0


def test_two_token_query_is_still_accepted(chatbot):
    response = chatbot.get_response("reset password")
    assert response["matched"] is True
    assert response["faq"]["id"] == 1


def test_fallback_message_is_centralized_and_non_empty():
    assert isinstance(DEFAULT_FALLBACK_MESSAGE, str)
    assert DEFAULT_FALLBACK_MESSAGE.strip()


# --- Evaluation over representative queries ---


RELATED_QUERIES = [
    ("I forgot my password. How can I change it?", 1),
    ("How do I change my password?", 1),
    ("How do I submit an assignment?", 25),
    ("Where can I download my certificate?", 22),
    ("My payment failed", 16),
    # Regression cases fixed by the Phase 9 normalizer.
    ("What payment options do you support?", 13),
    ("Where can I see my grades from exams?", 28),
    ("How do I check that a certificate is genuine?", 23),
    ("Can I retake an assignment after it is due?", 26),
    ("What do I get with the paid subscription?", 17),
    ("Do I need any background knowledge to enroll?", 12),
    ("Can I get a discount on a course?", 15),
    ("Can I pay for a course with a code?", 15),
]

UNRELATED_QUERIES = [
    "What is the weather today?",
    "Who won the football match?",
    "Tell me a joke.",
    "How do I bake a cake?",
]


@pytest.mark.parametrize("query,expected_id", RELATED_QUERIES)
def test_evaluation_related_queries_match(chatbot, query, expected_id):
    response = chatbot.get_response(query)
    assert response["matched"] is True, f"{query!r} should be accepted"
    assert response["faq"]["id"] == expected_id


@pytest.mark.parametrize("query", UNRELATED_QUERIES)
def test_evaluation_unrelated_queries_rejected(chatbot, query):
    response = chatbot.get_response(query)
    assert response["matched"] is False, f"{query!r} should be rejected"
    assert response["faq"] is None