"""
Language-specific evaluation modules.
"""

from .bhojpuri import evaluate_bhojpuri
from .bengali import evaluate_bengali
from .chhattisgarhi import evaluate_chhattisgarhi
from .hindi import evaluate_hindi
from .kannada import evaluate_kannada
from .magahi import evaluate_magahi
from .marathi import evaluate_marathi
from .marathi_konkani import evaluate_marathi_konkani
from .maithili import evaluate_maithili
from .telugu import evaluate_telugu

__all__ = [
    "evaluate_bhojpuri",
    "evaluate_bengali",
    "evaluate_chhattisgarhi",
    "evaluate_hindi",
    "evaluate_kannada",
    "evaluate_magahi",
    "evaluate_marathi",
    "evaluate_marathi_konkani",
    "evaluate_maithili",
    "evaluate_telugu",
]
