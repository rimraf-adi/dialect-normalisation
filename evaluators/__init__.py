"""
Language evaluation scripts for IndicConformer.
"""

from src.dialect_norm.evaluators.bhojpuri import evaluate_bhojpuri
from src.dialect_norm.evaluators.bengali import evaluate_bengali
from src.dialect_norm.evaluators.chhattisgarhi import evaluate_chhattisgarhi
from src.dialect_norm.evaluators.hindi import evaluate_hindi
from src.dialect_norm.evaluators.kannada import evaluate_kannada
from src.dialect_norm.evaluators.magahi import evaluate_magahi
from src.dialect_norm.evaluators.marathi import evaluate_marathi
from src.dialect_norm.evaluators.marathi_konkani import evaluate_marathi_konkani
from src.dialect_norm.evaluators.maithili import evaluate_maithili
from src.dialect_norm.evaluators.telugu import evaluate_telugu

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
