"""Model definitions for RNA classification."""

from .gru_classifier import GRUClassifier

__all__ = ["GRUClassifier"]

try:
    from .transformer_classifier import TransformerClassifier
except ImportError:
    TransformerClassifier = None
else:
    __all__.append("TransformerClassifier")
