"""Flask application entrypoint for the FAQ Chatbot."""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

from services.chatbot_service import (
    FAQChatbot,
    MAX_QUESTION_LENGTH,
    QuestionTooLongError,
    load_faqs,
)

BASE_DIR = Path(__file__).resolve().parent
FAQ_DATA_PATH = BASE_DIR / "data" / "faqs.json"

load_dotenv(BASE_DIR / ".env")

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "change-this-value")

# Reject oversized request bodies up front (denial-of-service protection).
# A 500-character question plus the JSON envelope is tiny, so 16 KB is a
# generous but bounded ceiling for a single chat request.
app.config["MAX_CONTENT_LENGTH"] = int(
    os.environ.get("FAQ_MAX_CONTENT_LENGTH", str(16 * 1024))
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
app.logger.setLevel(logging.INFO)

# Content-Security-Policy for a frontend that only loads same-origin
# resources: the app's own stylesheet and script, plus same-origin API calls.
# Skipped in debug mode so the Werkzeug debugger console keeps working.
CSP_POLICY = (
    "default-src 'self'; "
    "script-src 'self'; style-src 'self'; img-src 'self' data:; "
    "connect-src 'self'; font-src 'self'; object-src 'none'; "
    "base-uri 'self'; form-action 'self'; frame-ancestors 'none'"
)

# The chatbot is built once at startup: FAQ data is loaded and the TF-IDF
# vectorizer is fitted against the corpus a single time, not per request.
# A broken dataset stops the application clearly instead of producing
# unpredictable chatbot responses.
try:
    chatbot = FAQChatbot(
        load_faqs(FAQ_DATA_PATH),
        threshold=os.environ.get("SIMILARITY_THRESHOLD"),
    )
except Exception:
    app.logger.error("Failed to initialize the FAQ chatbot. Exiting.", exc_info=True)
    raise

app.logger.info(
    "FAQ chatbot started with %d FAQs at threshold %.2f",
    len(chatbot.faq_data),
    chatbot.threshold,
)


def _reject(reason, message, status=400):
    """Log a rejected request and return a consistent JSON error response."""
    app.logger.warning("Rejected chat request: %s", reason)
    return jsonify(error=message), status


@app.route("/")
def index():
    """Render the chatbot landing page."""
    return render_template("index.html")


@app.route("/api/health")
def health():
    """Health check endpoint used for uptime verification."""
    return jsonify(status="ok", message="FAQ Chatbot API is running"), 200


@app.route("/api/chat", methods=["POST"])
def chat():
    """Answer a user question using the FAQ chatbot.

    Validates the request, delegates to the Phase 5 chatbot decision logic,
    and returns a structured JSON response. Additional fields in the request
    body are ignored. User input is always treated as data, never code.
    """
    data = request.get_json(silent=True)
    if data is None:
        return _reject("malformed or missing JSON body", "Request body must contain valid JSON.")
    if not isinstance(data, dict):
        return _reject("non-object JSON", "Request body must be a JSON object.")
    if "question" not in data:
        return _reject("missing question field", "Question is required.")

    question = data["question"]
    if not isinstance(question, str):
        return _reject("non-string question", "Question must be a string.")
    if not question.strip():
        return _reject("empty question", "Question cannot be empty.")
    if len(question) > MAX_QUESTION_LENGTH:
        return _reject(
            "question exceeds maximum length",
            "Your question is too long. Please shorten it and try again.",
        )

    try:
        response = chatbot.get_response(question)
    except QuestionTooLongError:
        return _reject(
            "question exceeds maximum length",
            "Your question is too long. Please shorten it and try again.",
        )
    except Exception:
        app.logger.error("Failed to process chatbot request", exc_info=True)
        return jsonify(error="An internal server error occurred."), 500

    faq = response["faq"]
    return jsonify(
        matched=response["matched"],
        faq_id=faq["id"] if faq is not None else None,
        answer=response["message"],
        score=response["score"],
        category=faq["category"] if faq is not None else None,
    ), 200


@app.after_request
def apply_security_headers(response):
    """Add lightweight HTTP security headers to every response."""
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    # The legacy XSS filter is deprecated; CSP is the effective protection.
    response.headers.setdefault("X-XSS-Protection", "0")
    if not app.debug:
        response.headers.setdefault("Content-Security-Policy", CSP_POLICY)
    return response


@app.errorhandler(400)
def bad_request(error):
    """Return a consistent JSON error for generic bad requests."""
    app.logger.warning("Bad request: %s", error)
    return jsonify(error="Bad request."), 400


@app.errorhandler(404)
def not_found(error):
    """Return a clear JSON error for unknown API routes."""
    if request.path.startswith("/api/"):
        return jsonify(error="Endpoint not found"), 404
    return "Page not found", 404


@app.errorhandler(405)
def method_not_allowed(error):
    """Return a clear JSON error for unsupported HTTP methods."""
    if request.path.startswith("/api/"):
        return jsonify(error="Method not allowed."), 405
    return "Method not allowed", 405


@app.errorhandler(413)
def request_too_large(error):
    """Return a safe JSON error when the request body exceeds the limit."""
    app.logger.warning("Request body too large")
    return jsonify(error="Request body is too large."), 413


@app.errorhandler(500)
def internal_error(error):
    """Log unexpected errors without exposing internal details to users."""
    app.logger.error("Unhandled exception: %s", error, exc_info=True)
    if request.path.startswith("/api/"):
        return jsonify(error="An internal server error occurred."), 500
    return "Internal server error", 500


def _env_flag(name):
    """Read a boolean-style environment variable (1/true/yes/on)."""
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


if __name__ == "__main__":
    app.run(debug=_env_flag("FLASK_DEBUG"))