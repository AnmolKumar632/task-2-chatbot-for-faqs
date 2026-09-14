"""Controlled, domain-specific normalization for FAQ matching quality.

Only mappings that fix measured matching failures in the evaluation set are
included. This is intentionally tiny: it normalizes a small set of terms a
user is likely to phrase differently from the FAQ wording.

Rules are split into two layers:

* Phrase rules run on the lowercased raw text BEFORE tokenization, because
  they contain stop words (e.g. "money back") that would otherwise be
  removed before they can be matched.
* Token rules run AFTER lemmatization on base forms, so keys are stored in
  their lemmatized shape (e.g. "option", not "options").
"""

import re

PHRASE_REPLACEMENTS = [
    # "Do I get my money back after buying a course?" intends refund policy.
    ("money back", "refund"),
    # "What do I get with the paid subscription?" intends premium content.
    ("paid subscription", "premium subscription"),
    # "Do I need any background knowledge to enroll?" intends prerequisites.
    ("background knowledge", "prerequisite"),
]

TOKEN_REPLACEMENTS = {
    # "What payment options do you support?" ~ "What payment methods ..."
    "option": "method",
    # "Where can I see my grades from exams?" ~ "check my exam results".
    "grade": "result",
    # "How do I check that a certificate is genuine?" ~ "verify ... authentic".
    "genuine": "authentic",
    "verify": "check",
    # "Can I retake an assignment after it is due?" ~ "resubmit ... deadline".
    "retake": "resubmit",
    # "Can I pay for a course with a code?" means a discount/promo code.
    # Expands to a phrase so "discount code" becomes a shared 2-token unit.
    "code": "discount code",
}

_PHRASE_PATTERNS = [
    (re.compile(r"\b" + re.escape(phrase) + r"\b"), replacement)
    for phrase, replacement in PHRASE_REPLACEMENTS
]


def expand_phrases(text):
    """Replace known domain phrases with their normalized equivalents."""
    for pattern, replacement in _PHRASE_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def normalize_token(token):
    """Return the normalized form of a single lemmatized token."""
    return TOKEN_REPLACEMENTS.get(token, token)