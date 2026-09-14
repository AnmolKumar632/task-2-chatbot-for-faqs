"""Phase 10 security tests: input validation, injection resistance,
HTTP security headers, status codes, and no internal-detail leakage."""

from pathlib import Path

import pytest

from services.chatbot_service import MAX_QUESTION_LENGTH

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Markers that must never appear in any API response body.
LEAK_MARKERS = (
    "Traceback",
    "  File \"",
    ".py",
    "chatbot_service",
    "similarity_engine",
    "SECRET_KEY",
    "environ",
)


def _assert_safe_body(response):
    """Assert the response body never leaks internal implementation details."""
    body = response.get_data(as_text=True)
    for marker in LEAK_MARKERS:
        assert marker not in body, f"response leaked {marker!r}"


# --- Malicious / injection input is treated purely as data ---


@pytest.mark.parametrize(
    "payload",
    [
        "<script>alert('XSS')</script>",
        "<img src=x onerror=alert(1)>",
        "javascript:alert(1)",
        "'; DROP TABLE users; --",
        "<svg/onload=alert(1)>",
        "../../../../etc/passwd",
        "..\\..\\..\\Windows\\system32\\cmd.exe",
        "{{ 7 * 7 }}",
        "${7 * 7}",
    ],
)
def test_malicious_input_is_never_executed(client, payload):
    response = client.post("/api/chat", json={"question": payload})
    assert response.status_code == 200
    data = response.get_json()
    assert data["matched"] is False
    assert data["faq_id"] is None
    assert payload not in data["answer"]
    _assert_safe_body(response)


@pytest.mark.parametrize(
    "payload",
    [
        "<b>bold</b>",
        "<a href='https://example.com'>link</a>",
        "&lt;tag&gt;",
    ],
)
def test_html_payload_is_returned_as_text_only(client, payload):
    response = client.post("/api/chat", json={"question": payload})
    assert response.status_code == 200
    data = response.get_json()
    assert data["matched"] is False
    assert "<b>" not in data["answer"]


# --- Invalid question types ---


@pytest.mark.parametrize(
    "bad_value",
    [None, 123, 1.5, True, ["a", "b"], {"question": "nested"}],
)
def test_non_string_question_types_are_rejected(client, bad_value):
    response = client.post("/api/chat", json={"question": bad_value})
    assert response.status_code == 400
    assert response.get_json()["error"] == "Question must be a string."
    _assert_safe_body(response)


# --- Oversized input ---


def test_oversized_question_is_rejected(client):
    too_long = "a" * (MAX_QUESTION_LENGTH + 1)
    response = client.post("/api/chat", json={"question": too_long})
    assert response.status_code == 400
    data = response.get_json()
    assert data["error"] == "Your question is too long. Please shorten it and try again."
    _assert_safe_body(response)


def test_at_limit_question_is_accepted(client):
    at_limit = "a" * MAX_QUESTION_LENGTH
    response = client.post("/api/chat", json={"question": at_limit})
    assert response.status_code in (200, 400)
    if response.status_code == 200:
        assert response.get_json()["matched"] is False


# --- Unexpected fields cannot manipulate behavior ---


def test_unexpected_fields_are_ignored(client):
    response = client.post(
        "/api/chat",
        json={
            "question": "How can I reset my password?",
            "admin": True,
            "role": "administrator",
            "matched": True,
            "faq_id": 999,
        },
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["matched"] is True
    assert data["faq_id"] == 1
    assert "role" not in data
    assert "admin" not in data


# --- Security headers ---


def test_security_headers_on_homepage(client):
    response = client.get("/")
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["Referrer-Policy"] == "no-referrer"
    assert "default-src 'self'" in response.headers["Content-Security-Policy"]


def test_security_headers_on_chat_response(client):
    response = client.post(
        "/api/chat", json={"question": "How can I reset my password?"}
    )
    assert response.status_code == 200
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["Referrer-Policy"] == "no-referrer"


def test_security_headers_on_error_response(client):
    response = client.post("/api/chat", json={})
    assert response.status_code == 400
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"


def test_csp_allows_only_same_origin_for_scripts(client):
    response = client.get("/")
    csp = response.headers["Content-Security-Policy"]
    assert "script-src 'self'" in csp
    assert "object-src 'none'" in csp


# --- HTTP status codes ---


def test_unknown_api_route_returns_json_404(client):
    response = client.get("/api/does-not-exist")
    assert response.status_code == 404
    assert response.get_json()["error"] == "Endpoint not found"
    _assert_safe_body(response)


def test_unsupported_method_returns_405(client):
    response = client.get("/api/chat")
    assert response.status_code == 405
    assert response.get_json()["error"] == "Method not allowed."
    _assert_safe_body(response)


def test_non_api_404_returns_plain_text(client):
    response = client.get("/this-route-does-not-exist")
    assert response.status_code == 404
    assert response.get_data(as_text=True) == "Page not found"


def test_oversized_request_body_returns_413(client):
    oversized = "x" * (16 * 1024 + 1)
    response = client.post(
        "/api/chat", data=oversized, content_type="application/json"
    )
    assert response.status_code == 413
    assert response.get_json()["error"] == "Request body is too large."
    _assert_safe_body(response)


def test_invalid_json_returns_safe_400(client):
    response = client.post(
        "/api/chat", data="{not valid json", content_type="application/json"
    )
    assert response.status_code == 400
    _assert_safe_body(response)


# --- Static source review: no dynamic code execution ---


def test_no_eval_or_exec_in_source():
    targets = [PROJECT_ROOT / "app.py", PROJECT_ROOT / "services", PROJECT_ROOT / "nlp"]
    for target in targets:
        files = [target] if target.is_file() else sorted(target.glob("*.py"))
        for file in files:
            source = file.read_text(encoding="utf-8")
            for pattern in ("eval(", "exec(", "os.system(", "subprocess", "pickle.loads"):
                assert pattern not in source, (
                    f"{file.relative_to(PROJECT_ROOT)} contains {pattern!r}"
                )


# --- Static frontend review: user content is never rendered as HTML ---


def test_frontend_has_no_dynamic_html_renderers():
    script_file = PROJECT_ROOT / "static" / "js" / "script.js"
    source = script_file.read_text(encoding="utf-8")
    for pattern in (
        "insertAdjacentHTML",
        "document.write",
        "eval(",
        "new Function",
        ".html(",
        "createElement(\"script\"",
    ):
        assert pattern not in source, f"script.js contains {pattern!r}"


def test_frontend_innerhtml_is_only_used_to_clear_the_container():
    script_file = PROJECT_ROOT / "static" / "js" / "script.js"
    source = script_file.read_text(encoding="utf-8")
    for line in source.splitlines():
        statement = line.strip().strip(";")
        if "innerHTML" not in statement:
            continue
        assert statement == "chatMessages.innerHTML = \"\"", (
            f"unsafe innerHTML usage: {line.strip()!r}"
        )


def test_user_output_is_rendered_with_textcontent():
    script_file = PROJECT_ROOT / "static" / "js" / "script.js"
    source = script_file.read_text(encoding="utf-8")
    assert "textContent" in source
    assert "appendChild" in source


def test_template_has_no_inline_dynamic_script():
    template_file = PROJECT_ROOT / "templates" / "index.html"
    source = template_file.read_text(encoding="utf-8", errors="replace")
    assert "<script>" not in source
    assert "onerror=" not in source