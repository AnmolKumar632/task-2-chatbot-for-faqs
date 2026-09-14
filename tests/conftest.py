"""Shared pytest fixtures for the FAQ chatbot test suite."""

from pathlib import Path

import pytest

from app import app as flask_app
from services.chatbot_service import FAQChatbot, load_faqs

BASE_DIR = Path(__file__).resolve().parents[1]
FAQ_DATA_PATH = BASE_DIR / "data" / "faqs.json"


@pytest.fixture()
def client():
    """Reusable in-process Flask test client."""
    flask_app.config["TESTING"] = True
    with flask_app.test_client() as test_client:
        yield test_client


@pytest.fixture(scope="module")
def faqs():
    """Loaded production FAQ dataset as a list of dictionaries."""
    return load_faqs(FAQ_DATA_PATH)


@pytest.fixture(scope="module")
def chatbot_service(faqs):
    """A real FAQChatbot instance built from the production dataset."""
    return FAQChatbot(faqs, threshold=0.35)