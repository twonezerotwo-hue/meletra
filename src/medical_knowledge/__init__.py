"""v4 Medical Intelligence: an offline medical terminology + concept knowledge base (SQLite) and the retrieval / scoring engine on top of it.

Nothing in the v3.1.0 pipeline imports this package unless the feature flag ``MLT_V4_SHADOW`` / the "Medical Intelligence" setting is on.
"""

from .database.schema import KNOWLEDGE_NAME, SCHEMA_VERSION  # noqa: F401

__all__ = ["KNOWLEDGE_NAME", "SCHEMA_VERSION"]
