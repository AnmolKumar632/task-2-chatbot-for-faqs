"""Service layer for the FAQ chatbot."""

import json
import os

from nlp.text_processor import preprocess_tokens
from nlp.similarity_engine import SimilarityEngine

REQUIRED_FIELDS = ("id", "question", "answer", "category")

DEFAULT_SIMILARITY_THRESHOLD = 0.35

# Maximum user-question length, in characters. Configurable through the
# environment so operators can tighten it without code changes; the value
# mirrors the frontend input's `maxlength` attribute.
MAX_QUESTION_LENGTH = int(os.environ.get("FAQ_MAX_QUESTION_LENGTH", "500"))

DEFAULT_FALLBACK_MESSAGE = (
    "I'm sorry, I couldn't find a relevant answer to your question. "
    "Please try rephrasing your question or ask about courses, enrollment, "
    "payments, certificates, assignments, or account-related issues."
)

ALLOWED_CATEGORIES = {
    "Account",
    "Courses",
    "Enrollment",
    "Payments",
    "Certificates",
    "Assignments",
    "Examinations",
    "Technical Support",
    "Progress",
    "Subscriptions",
    "Refunds",
    "Profile",
}


class QuestionTooLongError(ValueError):
    """Raised when a user question exceeds MAX_QUESTION_LENGTH."""


def load_faqs(filepath):
    """Load and validate the FAQ knowledge base from a JSON file.

    Raises a clear, actionable error when the file is missing or contains
    invalid JSON instead of letting low-level exceptions leak upward.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as file:
            faqs = json.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"FAQ data file not found: {filepath}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"FAQ data file is not valid JSON: {filepath}"
        ) from exc
    return validate_faqs(faqs)


def validate_faqs(faqs):
    """Verify the FAQ knowledge base is well formed; raise ValueError otherwise."""
    if not isinstance(faqs, list):
        raise ValueError("FAQ data must be a JSON array.")
    if not faqs:
        raise ValueError("FAQ data must not be empty.")

    seen_ids = set()
    seen_questions = set()
    for index, faq in enumerate(faqs, start=1):
        if not isinstance(faq, dict):
            raise ValueError(f"FAQ record {index} must be a JSON object.")

        missing = [field for field in REQUIRED_FIELDS if field not in faq]
        if missing:
            raise ValueError(
                f"FAQ record {index} is missing fields: {', '.join(missing)}"
            )

        faq_id = faq["id"]
        if isinstance(faq_id, bool) or not isinstance(faq_id, int):
            raise ValueError(f"FAQ record {index} has a non-integer id.")
        if faq_id in seen_ids:
            raise ValueError(f"Duplicate FAQ id: {faq_id}")
        seen_ids.add(faq_id)

        for field in ("question", "answer", "category"):
            if not isinstance(faq[field], str) or not faq[field].strip():
                raise ValueError(f"FAQ record {index} has an empty {field}.")

        if faq["category"] not in ALLOWED_CATEGORIES:
            raise ValueError(
                f"FAQ record {index} has an unknown category: {faq['category']}."
            )

        normalized_question = " ".join(faq["question"].strip().lower().split())
        if normalized_question in seen_questions:
            raise ValueError(
                f"FAQ record {index} duplicates an earlier question."
            )
        seen_questions.add(normalized_question)

    return faqs


def normalize_threshold(value):
    """Validate a similarity threshold and return it as a float in [0, 1]."""
    try:
        threshold = float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"Similarity threshold must be a number between 0 and 1, got {value!r}."
        ) from None
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            f"Similarity threshold must be between 0 and 1, got {threshold}."
        )
    return threshold


def is_confident_match(score, threshold):
    """Accept a match when its similarity score meets the threshold."""
    return score >= threshold


class FAQChatbot:
    """Answers questions by matching them against the FAQ knowledge base."""

    def __init__(self, faq_data, threshold=None, fallback_message=None):
        self.faq_data = faq_data
        self.threshold = normalize_threshold(
            threshold if threshold is not None else DEFAULT_SIMILARITY_THRESHOLD
        )
        self.fallback_message = fallback_message or DEFAULT_FALLBACK_MESSAGE
        self.similarity_engine = SimilarityEngine(faq_data)

    def get_response(self, user_question):
        """Return a structured answer: an accepted FAQ or the fallback.

        The confidence decision uses the similarity threshold; results below
        it never produce an FAQ answer.
        """
        if not isinstance(user_question, str):
            raise TypeError(
                f"User question must be a string, got {type(user_question).__name__}."
            )
        if not user_question.strip():
            raise ValueError("Please enter a question.")
        if len(user_question) > MAX_QUESTION_LENGTH:
            raise QuestionTooLongError(
                f"Question exceeds the maximum length of {MAX_QUESTION_LENGTH} "
                "characters."
            )

        # Single-token queries (e.g. "certificate", "refund", "password") are
        # too ambiguous to answer confidently from the FAQ knowledge base and
        # were measured to produce false positives, so reject them outright.
        # The query is preprocessed exactly once here: the normalized tokens
        # are reused for the ambiguity check AND for the similarity engine,
        # avoiding a redundant NLTK pass for accepted queries.
        tokens = preprocess_tokens(user_question)
        if len(tokens) < 2:
            return {
                "matched": False,
                "faq": None,
                "score": 0.0,
                "message": self.fallback_message,
            }

        best_match = self.similarity_engine.find_best_match_processed(
            " ".join(tokens)
        )
        score = best_match["score"]
        faq = best_match["faq"]

        if faq is not None and is_confident_match(score, self.threshold):
            return {
                "matched": True,
                "faq": faq,
                "score": round(score, 4),
                "message": faq["answer"],
            }

        return {
            "matched": False,
            "faq": None,
            "score": round(score, 4),
            "message": self.fallback_message,
        }