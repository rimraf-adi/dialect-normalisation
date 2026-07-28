"""
Dialectal Distortion & Divergence Scoring Engine for Marathi Sub-Dialects (D1, D2, D4).
"""

import re
from typing import Dict

# Dialect-specific non-standard lexical markers and suffixes
DIALECT_MARKERS: Dict[str, list] = {
    "D1": [  # South Konkan (Sindhudurg / Ratnagiri)
        r"\bहा\b", r"\bअसात\b", r"\bनाहींत\b", r"जानवरांका", r"\bतुका\b",
        r"ओढतत", r"केली जाता", r"\bअसां\b", r"करतां", r"\bआसा\b", r"झालां",
        r"गेलो\b", r"ऐकया\b", r"सांगा\b", r"माका\b", r"चेडवा\b", r"घोवा\b"
    ],
    "D2": [  # North Konkan (Palghar / Thane)
        r"\bसंगी\b", r"\bनयी\b", r"शकतस", r"\bमा\b", r"\bहोयस\b",
        r"\bयेस\b", r"ठेवलं जातं", r"करणार", r"\bपोरा\b", r"\bखास\b",
        r"जास\b", r"करस\b", r"बघस\b", r"\bआसवे\b", r"\bतुले\b"
    ],
    "D4": [  # Varhadi (Vidarbha / Amravati)
        r"\bमले\b", r"\bतुले\b", r"\bभेटन\b", r"\bरायते\b", r"\bपाजी\b",
        r"\bआयत\b", r"\bकोनचे\b", r"\bलागन\b", r"\bहोयी\b", r"\bनाय\b",
        r"पाहिजेत\b", r"होऊन राहिला", r"\bकायले\b", r"\bकशाले\b", r"पाहिजेना\b"
    ]
}

# Generic non-standard regional verb endings & suffixes across Marathi dialects
NON_STANDARD_SUFFIX_PATTERNS = [
    r"[क-ह]तत\b",       # e.g., ओढतत, करतातत
    r"[क-ह]नस\b",       # e.g., देऊ शकतस
    r"[क-ह]तन\b",       # e.g., भेटन, जायन
    r"[क-ह]ास\b",       # e.g., जास, करास
    r"[क-ह]ांका\b",     # e.g., जानवरांका, पोरांका
    r"[क-ह]ाले\b",      # e.g., कायले, कशाले
]


def compute_dialect_distortion_score(text: str, dialect_code: str = None) -> float:
    """
    Computes a numerical dialectal distortion score for a given Marathi sentence.
    Higher score indicates higher divergence from Standard Pune Marathi (D3).
    """
    if not text or not isinstance(text, str):
        return 0.0

    text_clean = text.strip()
    words = text_clean.split()
    if not words:
        return 0.0

    score = 0.0

    # 1. Target Dialect Marker Match (High Weight)
    if dialect_code in DIALECT_MARKERS:
        for pattern in DIALECT_MARKERS[dialect_code]:
            matches = len(re.findall(pattern, text_clean))
            score += matches * 3.0

    # 2. Cross-Dialect Marker Match (Medium Weight)
    for d_code, markers in DIALECT_MARKERS.items():
        if d_code != dialect_code:
            for pattern in markers:
                matches = len(re.findall(pattern, text_clean))
                score += matches * 1.5

    # 3. Non-Standard Suffix Pattern Match
    for pattern in NON_STANDARD_SUFFIX_PATTERNS:
        matches = len(re.findall(pattern, text_clean))
        score += matches * 2.0

    # 4. Length and Lexical Complexity Normalization
    unique_word_ratio = len(set(words)) / len(words)
    score += unique_word_ratio * 1.0

    # 5. Punctuation & Non-standard token density
    non_std_chars = len(re.findall(r"[^\w\s\.\,\?\!]", text_clean))
    score += non_std_chars * 0.5

    return round(score, 3)
