"""NLP preprocessing pipeline for the FAQ chatbot.

The pipeline normalizes raw text so that FAQ questions and future user
questions can be compared consistently by the similarity engine built in a
later phase:

    Raw text
     -> Lowercase
     -> Contraction expansion
     -> Domain phrase replacement
     -> Special-character cleaning
     -> Tokenization (NLTK)
     -> Stop-word removal
     -> Lemmatization (verb-aware, with noun fallback)
     -> Domain token normalization
     -> Normalized text
"""

import re
import os

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from nlp.query_normalizer import expand_phrases, normalize_token

REQUIRED_NLTK_RESOURCES = (
    "tokenizers/punkt_tab",
    "corpora/stopwords",
    "corpora/wordnet",
    "corpora/omw-1.4",
)

_CONTRACTION_MAP = {
    "can't": "cannot",
    "can\u2019t": "cannot",
    "don't": "do not",
    "don\u2019t": "do not",
    "won't": "will not",
    "won\u2019t": "will not",
    "doesn't": "does not",
    "doesn\u2019t": "does not",
    "isn't": "is not",
    "isn\u2019t": "is not",
    "didn't": "did not",
    "didn\u2019t": "did not",
    "i'm": "i am",
    "i\u2019m": "i am",
    "i've": "i have",
    "i\u2019ve": "i have",
    "you're": "you are",
    "you\u2019re": "you are",
    "there's": "there is",
    "there\u2019s": "there is",
}

_CLEANING_PATTERN = re.compile(r"[^a-z0-9]+")

_STOP_WORDS = None
_LEMMATIZER = None
_resources_ready = False
_PORTABLE_NLP = os.environ.get("FAQ_PORTABLE_NLP", "").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)

_PORTABLE_STOP_WORDS = {
    "a", "am", "an", "and", "are", "as", "at", "be", "can", "could",
    "do", "does", "for", "from", "how", "i", "if", "in", "is", "it",
    "me", "my", "of", "on", "or", "should", "that", "the", "their",
    "there", "this", "to", "was", "what", "when", "where", "which",
    "who", "will", "with", "you", "your",
}

_PORTABLE_IRREGULARS = {
    "bought": "buy",
    "buying": "buy",
    "failed": "fail",
    "forgot": "forget",
    "paid": "pay",
    "took": "take",
}


def _initialize_components():
    """Load stop words and the lemmatizer after the resources are available."""
    global _STOP_WORDS, _LEMMATIZER
    if _STOP_WORDS is None:
        _STOP_WORDS = set(stopwords.words("english"))
    if _LEMMATIZER is None:
        _LEMMATIZER = WordNetLemmatizer()


def ensure_nltk_resources():
    """Download any missing NLTK resources once and reuse them afterwards."""
    global _resources_ready
    if _resources_ready:
        return

    if _PORTABLE_NLP:
        _resources_ready = True
        return

    missing = []
    for resource in REQUIRED_NLTK_RESOURCES:
        try:
            nltk.data.find(resource)
        except LookupError:
            missing.append(resource)

    unavailable = []
    for resource in missing:
        package_name = resource.rsplit("/", 1)[-1]
        try:
            downloaded = nltk.download(package_name, quiet=True)
        except Exception:  # noqa: BLE001 - download failures vary by environment
            downloaded = False
        if not downloaded:
            unavailable.append(resource)

    if unavailable:
        raise RuntimeError(
            f"NLTK resources could not be downloaded: {', '.join(unavailable)}. "
            "Check your internet connection. If you are behind a proxy, install "
            "the resources manually with the NLTK Downloader before starting."
        )

    _initialize_components()
    _resources_ready = True


def _validate_text(text):
    """Return text as a string or raise a clear TypeError."""
    if not isinstance(text, str):
        raise TypeError(f"preprocess_text expects a string, got {type(text).__name__}.")
    return text


def _normalize_case(text):
    return text.lower()


def _expand_contractions(text):
    for contraction, expanded in _CONTRACTION_MAP.items():
        text = text.replace(contraction, expanded)
    return text


def _clean_text(text):
    return _CLEANING_PATTERN.sub(" ", text)


def _portable_lemma(token):
    """Apply small deterministic rules when NLTK data is unavailable."""
    if token in _PORTABLE_IRREGULARS:
        return _PORTABLE_IRREGULARS[token]
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 5 and token.endswith("ing"):
        stem = token[:-3]
        return stem[:-1] if len(stem) > 3 and stem[-1] == stem[-2] else stem
    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _lemmatize_token(token):
    """Lemmatize with verb-aware inflection handling and noun fallback.

    WordNetLemmatizer defaults to the noun sense, which leaves verb
    inflections untouched (e.g. "buying", "failed"). If the verb-sense
    lemma differs from the raw token, prefer it; otherwise keep the noun
    result (which also handles plurals).
    """
    noun = _LEMMATIZER.lemmatize(token)
    if noun == token:
        verb = _LEMMATIZER.lemmatize(token, pos="v")
        if verb != token:
            return verb
    return noun


def preprocess_tokens(text):
    """Normalize text and return the processed tokens (token list)."""
    text = _validate_text(text).strip()
    if not text:
        return []

    ensure_nltk_resources()

    normalized = _normalize_case(text)
    normalized = _expand_contractions(normalized)
    normalized = expand_phrases(normalized)
    normalized = _clean_text(normalized)
    tokens = (
        normalized.split()
        if _PORTABLE_NLP
        else word_tokenize(normalized)
    )

    meaningful = []
    for token in tokens:
        if not token.isalnum():
            continue
        if token in (_PORTABLE_STOP_WORDS if _PORTABLE_NLP else _STOP_WORDS):
            continue
        lemma = _portable_lemma(token) if _PORTABLE_NLP else _lemmatize_token(token)
        meaningful.append(normalize_token(lemma))

    return meaningful


def preprocess_text(text):
    """Normalize text into a cleaned, whitespace-normalized string."""
    tokens = preprocess_tokens(text)
    return " ".join(tokens)