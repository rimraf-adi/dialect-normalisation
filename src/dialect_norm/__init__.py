"""
Dialect Normalisation Framework Package.
"""

from .distortion import compute_dialect_distortion_score
from .sampler import load_and_sample_dialect_dataset
from .gemma_pipeline import execute_generation_pipeline

# Optional PyTorch/ASR dependencies
try:
    from .audio import load_and_preprocess_audio
    from .metrics import compute_metrics_for_set, normalize_text
    from .model import load_indic_conformer_model, run_decoder_inference
    from .reporter import generate_yaml_reports
    from .engine import evaluate_language
except ImportError:
    pass

__all__ = [
    "compute_dialect_distortion_score",
    "load_and_sample_dialect_dataset",
    "execute_generation_pipeline",
]
