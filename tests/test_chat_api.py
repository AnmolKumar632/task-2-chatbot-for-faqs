"""API tests for the Phase 6 Flask chat endpoint."""

import app as app_module


def test_home_page_returns_200(client):
    response = client.get("/")
    assert response.status_code == 200


def test_health_returns_ok(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "ok"
    assert data["message"] == "FAQ Chatbot API is running"


def test_valid_question_returns_structured_answer(client):
    response = client.post(
        "/api/chat", json={"question": "How can I reset my password?"}
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["matched"] is True
    assert isinstance(data["faq_id"], int)
    assert data["answer"]
    assert isinstance(data["score"], float)
    assert data["category"]


def test_paraphrased_question_uses_real_pipeline(client):
    response = client.post(
        "/api/chat",
        json={"question": "I forgot my password. How can I change it?"},
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["matched"] is True
    assert data["faq_id"] == 1
    assert data["score"] >= 0.35


def test_unrelated_question_is_rejected_without_faq_answer(client):
    response = client.post(
        "/api/chat", json={"question": "What is the capital of France?"}
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["matched"] is False
    assert data["faq_id"] is None
    assert data["category"] is None
    assert isinstance(data["score"], float)


def test_unrelated_question_returns_fallback_not_faq(client):
    response = client.post(
        "/api/chat", json={"question": "What is the capital of France?"}
    )
    data = response.get_json()
    assert "couldn't find a relevant answer" in data["answer"]


def test_missing_question_returns_400(client):
    response = client.post("/api/chat", json={})
    assert response.status_code == 400
    assert response.get_json()["error"] == "Question is required."


def test_empty_question_returns_400(client):
    response = client.post("/api/chat", json={"question": ""})
    assert response.status_code == 400
    assert response.get_json()["error"] == "Question cannot be empty."


def test_whitespace_question_returns_400(client):
    response = client.post("/api/chat", json={"question": "     "})
    assert response.status_code == 400
    assert response.get_json()["error"] == "Question cannot be empty."


def test_non_string_question_returns_400(client):
    response = client.post("/api/chat", json={"question": 123})
    assert response.status_code == 400
    assert response.get_json()["error"] == "Question must be a string."


def test_invalid_json_returns_400(client):
    response = client.post(
        "/api/chat",
        data="{not valid json",
        content_type="application/json",
    )
    assert response.status_code == 400
    assert response.get_json()["error"] == "Request body must contain valid JSON."


def test_missing_body_returns_400(client):
    response = client.post("/api/chat")
    assert response.status_code == 400
    assert response.get_json()["error"] == "Request body must contain valid JSON."


def test_non_object_body_returns_400(client):
    response = client.post("/api/chat", json=["a", "b"])
    assert response.status_code == 400
    assert response.get_json()["error"] == "Request body must be a JSON object."


def test_wrong_content_type_with_json_body_returns_400(client):
    response = client.post(
        "/api/chat",
        data='{"question": "How can I reset my password?"}',
        content_type="text/plain",
    )
    assert response.status_code == 400
    assert response.get_json()["error"] == "Request body must contain valid JSON."


def test_missing_content_type_returns_400(client):
    response = client.post(
        "/api/chat",
        data='{"question": "How can I reset my password?"}',
    )
    assert response.status_code == 400
    assert response.get_json()["error"] == "Request body must contain valid JSON."


def test_repeated_requests_return_identical_results(client):
    question = {"question": "I forgot my password. How can I change it?"}
    first = client.post("/api/chat", json=question).get_json()
    second = client.post("/api/chat", json=question).get_json()
    assert first == second


def test_matched_response_schema(client):
    response = client.post(
        "/api/chat", json={"question": "How do I submit an assignment?"}
    )
    data = response.get_json()
    assert {"matched", "faq_id", "answer", "score", "category"} <= set(data)


def test_rejected_response_schema(client):
    response = client.post(
        "/api/chat", json={"question": "How far is the moon from Earth?"}
    )
    data = response.get_json()
    assert {"matched", "faq_id", "answer", "score", "category"} <= set(data)


def test_unexpected_fields_are_ignored(client):
    response = client.post(
        "/api/chat",
        json={"question": "How can I reset my password?", "random_field": "test"},
    )
    assert response.status_code == 200
    assert response.get_json()["matched"] is True


def test_internal_error_returns_safe_json(client, monkeypatch):
    def boom(question):
        raise RuntimeError("boom")

    monkeypatch.setattr(app_module.chatbot, "get_response", boom)

    response = client.post(
        "/api/chat", json={"question": "How can I reset my password?"}
    )
    assert response.status_code == 500
    data = response.get_json()
    assert data["error"] == "An internal server error occurred."
    assert "RuntimeError" not in response.get_data(as_text=True)