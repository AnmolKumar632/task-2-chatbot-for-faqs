"""Dataset tests for the FAQ knowledge base."""

import json
from pathlib import Path

from services.chatbot_service import ALLOWED_CATEGORIES, load_faqs

FAQ_FILE = Path(__file__).resolve().parents[1] / "data" / "faqs.json"
REQUIRED_FIELDS = ("id", "question", "answer", "category")


def test_dataset_file_exists():
    assert FAQ_FILE.is_file()


def test_dataset_loads_without_errors():
    faqs = load_faqs(FAQ_FILE)
    assert isinstance(faqs, list)


def test_dataset_contains_enough_faqs():
    faqs = load_faqs(FAQ_FILE)
    assert 25 <= len(faqs) <= 30


def test_ids_are_unique():
    faqs = load_faqs(FAQ_FILE)
    ids = [faq["id"] for faq in faqs]
    assert len(ids) == len(set(ids))


def test_questions_are_unique():
    faqs = load_faqs(FAQ_FILE)
    questions = [" ".join(faq["question"].strip().lower().split()) for faq in faqs]
    assert len(questions) == len(set(questions))


def test_records_have_required_fields():
    faqs = load_faqs(FAQ_FILE)
    for faq in faqs:
        for field in REQUIRED_FIELDS:
            assert field in faq, f"FAQ {faq.get('id')} is missing '{field}'"


def test_text_fields_are_not_empty():
    faqs = load_faqs(FAQ_FILE)
    for faq in faqs:
        for field in ("question", "answer", "category"):
            assert str(faq[field]).strip(), f"FAQ {faq['id']} has an empty {field}"


def test_field_types_are_valid():
    faqs = load_faqs(FAQ_FILE)
    for faq in faqs:
        assert isinstance(faq["id"], int)
        assert isinstance(faq["question"], str)
        assert isinstance(faq["answer"], str)
        assert isinstance(faq["category"], str)


def test_categories_are_allowed():
    faqs = load_faqs(FAQ_FILE)
    for faq in faqs:
        assert faq["category"] in ALLOWED_CATEGORIES


def test_json_file_is_valid_json():
    with open(FAQ_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)
    assert isinstance(data, list)