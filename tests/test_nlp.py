"""Unit and integration tests for the NLP preprocessing pipeline."""

from pathlib import Path

import pytest

from nlp.text_processor import preprocess_text, preprocess_tokens

FAQ_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "faqs.json"


def test_lowercase_normalization():
    assert preprocess_text("HELLO WORLD") == "hello world"


def test_punctuation_removal():
    assert preprocess_text("Hello, World!") == "hello world"


def test_repeated_punctuation_is_removed():
    assert preprocess_text("How can I reset my password???") == "reset password"


def test_whitespace_normalization():
    assert preprocess_text("hello     world") == "hello world"


def test_tabs_and_newlines_are_normalized():
    assert preprocess_text("hello\tworld") == "hello world"
    assert preprocess_text("hello\n\nworld") == "hello world"
    assert preprocess_text("hello \t world\r\nchatbot") == "hello world chatbot"


def test_trailing_and_leading_whitespace():
    assert preprocess_text("   hello world   ") == "hello world"


def test_tokenization():
    assert preprocess_tokens("How can I reset my password?") == ["reset", "password"]


def test_stopword_removal():
    tokens = preprocess_tokens("The student is taking an important course")
    assert "student" in tokens
    assert "course" in tokens
    assert "the" not in tokens
    assert "is" not in tokens


def test_lemmatization():
    assert preprocess_text("courses students") == "course student"


@pytest.mark.parametrize(
    "source,expected",
    [
        ("buying", "buy"),
        ("failed", "fail"),
        ("forgot", "forget"),
        ("took", "take"),
        ("paid", "pay"),
        # WordNet has no verb paradigm for "resubmitted", so it stays as-is.
        ("resubmitted", "resubmitted"),
    ],
)
def test_verb_aware_lemmatization(source, expected):
    assert preprocess_text(source) == expected


def test_noun_pluralization_still_applies():
    assert preprocess_text("students courses") == "student course"


def test_domain_phrase_normalization_in_pipeline():
    assert preprocess_text("get my money back") == "get refund"


def test_domain_token_normalization_in_pipeline():
    assert preprocess_text("payment options") == "payment method"
    assert preprocess_text("is it genuine") == "authentic"


def test_code_expands_to_discount_code():
    assert preprocess_text("pay with a code") == "pay discount code"


def test_empty_input():
    assert preprocess_text("") == ""


def test_whitespace_only_input():
    assert preprocess_text("     ") == ""


def test_invalid_input_raises_type_error():
    for bad_input in (None, 123, [], {}):
        with pytest.raises(TypeError):
            preprocess_text(bad_input)


def test_deterministic_processing():
    text = "How can I reset my password?"
    assert preprocess_text(text) == preprocess_text(text)
    assert preprocess_tokens(text) == preprocess_tokens(text)


def test_real_faq_question_preserves_key_terms():
    result = preprocess_text("How can I reset my password?")
    assert isinstance(result, str)
    assert "reset" in result
    assert "password" in result


def test_real_faq_pipeline():
    import json

    with open(FAQ_DATA_PATH, "r", encoding="utf-8") as file:
        faqs = json.load(file)
    for faq in faqs:
        original = faq["question"]
        processed = preprocess_text(original)
        assert isinstance(processed, str)
        assert processed == processed.strip()