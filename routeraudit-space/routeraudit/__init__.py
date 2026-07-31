"""RouterAudit: offline audit planning for quality-constrained LLM model-switch decisions.

The package never calls a model API, never modifies routing, and never claims a saving that its
own intervals do not support.
"""
__version__ = "0.1.0"

from .api import AuditConfig, AuditResult, analyze_trace  # noqa: E402,F401
from .validation import SchemaError  # noqa: E402,F401

__all__ = ["AuditConfig", "AuditResult", "analyze_trace", "SchemaError", "__version__"]
