"""
Legacy entry point for eval_engine. Re-exports functions from src.dialect_norm.
"""

from src.dialect_norm import (
    load_and_preprocess_audio,
    normalize_text,
    compute_metrics_for_set,
    load_indic_conformer_model,
    generate_yaml_reports,
    evaluate_language as evaluate_dataset,
)

__all__ = [
    "load_and_preprocess_audio",
    "normalize_text",
    "compute_metrics_for_set",
    "load_indic_conformer_model",
    "generate_yaml_reports",
    "evaluate_dataset",
]
