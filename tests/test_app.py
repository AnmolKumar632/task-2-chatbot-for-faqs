"""Basic application and dataset tests for Phase 1."""

from pathlib import Path

import pytest

from services.chatbot_service import load_faqs, validate_faqs

FAQ_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "faqs.json"


def test_homepage_loads(client):
    response = client.get("/")
    assert response.status_code == 200


def test_health_endpoint_returns_ok(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "ok"
    assert data["message"] == "FAQ Chatbot API is running"


def test_faq_dataset_loads_and_is_valid():
    faqs = load_faqs(FAQ_DATA_PATH)
    assert isinstance(faqs, list)
    assert len(faqs) >= 20


def test_faq_ids_are_unique():
    faqs = load_faqs(FAQ_DATA_PATH)
    ids = [faq["id"] for faq in faqs]
    assert len(ids) == len(set(ids))


def test_validation_rejects_duplicate_ids():
    invalid_data = [
        {"id": 1, "question": "Question A", "answer": "Answer A", "category": "General"},
        {"id": 1, "question": "Question B", "answer": "Answer B", "category": "General"},
    ]
    with pytest.raises(ValueError):
        validate_faqs(invalid_data)


def test_validation_rejects_missing_question():
    invalid_data = [{"id": 1, "answer": "Answer A", "category": "General"}]
    with pytest.raises(ValueError):
        validate_faqs(invalid_data)