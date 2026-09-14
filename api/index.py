"""Vercel serverless entrypoint for the Flask application."""

import os

# Vercel functions do not guarantee writable, persistent NLTK data or outbound
# downloads during a cold start. The NLP module uses its deterministic portable
# tokenizer when this deployment flag is enabled.
os.environ.setdefault("FAQ_PORTABLE_NLP", "1")

from app import app

__all__ = ["app"]