"""
Dialect Normalisation & Evaluation Package for Indic Languages.
"""

__version__ = "0.1.0"

from .audio import load_and_preprocess_audio
from .metrics import normalize_text, compute_metrics_for_set
from .model import load_indic_conformer_model
from .reporter import generate_yaml_reports
from .engine import evaluate_language

__all__ = [
    "load_and_preprocess_audio",
    "normalize_text",
    "compute_metrics_for_set",
    "load_indic_conformer_model",
    "generate_yaml_reports",
    "evaluate_language",
]
