"""Direction 1 serving package (FastAPI app + session store + ingest worker).

Importing this package is cheap: the heavy modules are only pulled in when you import
``tmm.serve.app`` (which builds the FAISS index) or ``tmm.serve.consumer``.
"""

__all__ = ["app", "consumer", "events", "obs", "ratelimit", "resilience", "session",
           "settings"]
