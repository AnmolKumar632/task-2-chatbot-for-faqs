"""Unit tests for the controlled domain normalizer (Phase 9)."""

import pytest

from nlp.query_normalizer import PHRASE_REPLACEMENTS, TOKEN_REPLACEMENTS, expand_phrases, normalize_token


def test_phrase_replacement_maps_to_lemmatizable_output():
    assert expand_phrases("do i get my money back after buying a course?") == (
        "do i get my refund after buying a course?"
    )


def test_phrase_replacement_is_context_free():
    assert expand_phrases("she put money back on the shelf") == (
        "she put refund on the shelf"
    )


def test_phrase_replacement_preserves_original_casing():
    assert expand_phrases("Do I get my money back?") == "Do I get my refund?"


def test_paid_subscription_phrase_replacement():
    assert expand_phrases("what do i get with the paid subscription?") == (
        "what do i get with the premium subscription?"
    )


def test_background_knowledge_phrase_replacement():
    assert expand_phrases("do i need any background knowledge to enroll?") == (
        "do i need any prerequisite to enroll?"
    )


def test_phrase_replacement_keeps_unrelated_text_unchanged():
    assert expand_phrases("how do i submit an assignment?") == "how do i submit an assignment?"


def test_token_replacement_is_identity_for_unknown_tokens():
    assert normalize_token("course") == "course"
    assert normalize_token("enroll") == "enroll"


@pytest.mark.parametrize(
    "source,expected",
    [
        ("option", "method"),
        ("grade", "result"),
        ("genuine", "authentic"),
        ("verify", "check"),
        ("retake", "resubmit"),
        ("code", "discount code"),
    ],
)
def test_known_token_replacements(source, expected):
    assert normalize_token(source) == expected


def test_all_token_keys_are_lowercase_lemma_forms():
    for key in TOKEN_REPLACEMENTS:
        assert key == key.lower()
        assert " " not in key


def test_all_phrase_components_are_lowercase():
    for source, target in PHRASE_REPLACEMENTS:
        assert source == source.lower()
        assert target == target.lower()


def test_normalizer_is_deterministic():
    text = "Can I pay for a course with a discount code?"
    assert expand_phrases(text) == expand_phrases(text)