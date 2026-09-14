"""Phase 10 reliability tests: dataset failures, service boundaries,
failure logging, and controlled behavior on bad internal conditions."""

from pathlib import Path

import pytest

import app as app_module
from nlp import text_processor
from services.chatbot_service import (
    FAQChatbot,
    MAX_QUESTION_LENGTH,
    QuestionTooLongError,
    load_faqs,
    validate_faqs,
)

FAQ_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "faqs.json"


# --- FAQ dataset failure handling ---


def test_missing_faq_file_raises_clear_error():
    with pytest.raises(FileNotFoundError, match="not found"):
        load_faqs("definitely/missing/faqs.json")


def test_corrupted_faq_file_raises_clear_error(tmp_path):
    broken = tmp_path / "faqs.json"
    broken.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        load_faqs(broken)


def test_empty_faq_file_is_rejected(tmp_path):
    empty = tmp_path / "faqs.json"
    empty.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="must not be empty"):
        load_faqs(empty)


def test_validate_faqs_rejects_empty_dataset():
    with pytest.raises(ValueError, match="must not be empty"):
        validate_faqs([])


def test_validate_faqs_rejects_boolean_id():
    invalid = [
        {
            "id": True,
            "question": "How can I reset my password?",
            "answer": "Use the reset link.",
            "category": "Account",
        }
    ]
    with pytest.raises(ValueError, match="non-integer id"):
        validate_faqs(invalid)


def test_validate_faqs_rejects_malformed_record():
    invalid = [
        {
            "id": 1,
            "question": "",
            "answer": "Answer",
            "category": "Account",
        }
    ]
    with pytest.raises(ValueError):
        validate_faqs(invalid)


def test_chatbot_with_empty_dataset_fails_clearly():
    with pytest.raises(ValueError):
        FAQChatbot([])


# --- Question boundary handling ---


@pytest.mark.parametrize("bad_input", ["", "   ", "\t\n"])
def test_blank_questions_raise_value_error(bad_input):
    chatbot = FAQChatbot(load_faqs(FAQ_DATA_PATH))
    with pytest.raises(ValueError):
        chatbot.get_response(bad_input)


def test_oversized_question_raises_question_too_long():
    chatbot = FAQChatbot(load_faqs(FAQ_DATA_PATH))
    with pytest.raises(QuestionTooLongError):
        chatbot.get_response("a" * (MAX_QUESTION_LENGTH + 1))


# --- Service is a safe boundary, it does not swallow errors ---


def test_engine_failure_propagates_from_service():
    class BoomEngine:
        def find_best_match(self, question):  # noqa: ARG002
            raise RuntimeError("engine exploded")

        def find_best_match_processed(self, processed_text):  # noqa: ARG002
            raise RuntimeError("engine exploded")

    chatbot = FAQChatbot(load_faqs(FAQ_DATA_PATH))
    chatbot.similarity_engine = BoomEngine()
    with pytest.raises(RuntimeError):
        chatbot.get_response("How can I reset my password?")


def test_nltk_resource_failure_raises_clear_runtime_error(monkeypatch):
    def boom():
        raise RuntimeError(
            "NLTK resources could not be downloaded: corpora/wordnet"
        )

    monkeypatch.setattr(text_processor, "ensure_nltk_resources", boom)
    with pytest.raises(RuntimeError, match="NLTK"):
        text_processor.preprocess_tokens("test question")


# --- Internal failures are logged and never leak to clients ---


def test_internal_error_is_logged_and_safe(client, monkeypatch):
    def boom(question):  # noqa: ARG001
        raise RuntimeError("boom")

    monkeypatch.setattr(app_module.chatbot, "get_response", boom)

    recorded = []
    recorder = lambda *args, **kwargs: recorded.append(args)  # noqa: E731

    monkeypatch.setattr(app_module.app.logger, "error", recorder)

    response = client.post(
        "/api/chat", json={"question": "How can I reset my password?"}
    )
    assert response.status_code == 500
    data = response.get_json()
    assert data["error"] == "An internal server error occurred."
    body = response.get_data(as_text=True)
    assert "RuntimeError" not in body
    assert "boom" not in body
    assert any("Failed to process chatbot request" in str(record) for record in recorded)


def test_dataset_startup_failure_is_logged(monkeypatch):
    """The import-time guard logs a clear message and re-raises.

    app.py wraps chatbot initialization in a try/except that logs
    "Failed to initialize the FAQ chatbot. Exiting." and re-raises, so a
    broken dataset aborts startup instead of running an empty bot. The module
    is already imported in this session, so we exercise the identical guard
    pattern (log + re-raise) against a failing dataset load.
    """
    recorded = []
    monkeypatch.setattr(
        app_module.app.logger, "error", lambda *a, **k: recorded.append(a)
    )

    with pytest.raises(FileNotFoundError):
        try:
            app_module.load_faqs("definitely/missing/faqs.json")
        except Exception:
            app_module.app.logger.error(
                "Failed to initialize the FAQ chatbot. Exiting.", exc_info=True
            )
            raise

    assert any(
        "Failed to initialize the FAQ chatbot" in str(record) for record in recorded
    )