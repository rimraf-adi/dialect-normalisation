# Dialect Normalisation & Indic ASR Benchmarking Suite

An automated evaluation framework for benchmarking **IndicConformer** (`ai4bharat/indic-conformer-600m-multilingual`) across Indian languages and dialect variations using the **IISc RESPIN** datasets.

---

## 🌟 Supported Languages & Datasets

| Language | Dataset Code | Model Language Code | RESPIN Dataset Folder | Utterances |
| :--- | :--- | :--- | :--- | :--- |
| **Bhojpuri** | `bh` | `hi` | `IISc_RESPIN_test_bh` | 2,220 |
| **Bengali** | `bn` | `bn` | `IISc_RESPIN_test_bn` | 2,174 |
| **Chhattisgarhi** | `ch` | `hi` | `IISc_RESPIN_test_ch` | 2,234 |
| **Hindi** | `hi` | `hi` | `IISc_RESPIN_test_hi` | 2,288 |
| **Kannada** | `kn` | `kn` | `IISc_RESPIN_test_kn` | 2,161 |
| **Magahi** | `mg` | `hi` | `IISc_RESPIN_test_mg` | 2,193 |
| **Marathi** | `mr` | `mr` | `IISc_RESPIN_test_mr` | 2,170 |
| **Maithili** | `mt` | `mai` | `IISc_RESPIN_test_mt` | 2,172 |
| **Telugu** | `te` | `te` | `IISc_RESPIN_test_te` | 2,226 |

---

## 📁 Repository Structure

```
dialect-norm/
├── pyproject.toml                         # Package configuration & dependencies
├── README.md                              # Framework documentation
├── evaluate.py                            # Unified CLI entry point
├── baseline-indic-conformer/              # Output directory for YAML benchmarks
│   ├── indic_conformer_marathi_detailed.yaml
│   └── indic_conformer_marathi_summary.yaml
├── src/                                   # Core dialect_norm Python Package
│   └── dialect_norm/
│       ├── __init__.py                    # Package initialization & exports
│       ├── audio.py                       # Audio loader, mono conversion, 16kHz resampling
│       ├── metrics.py                     # Indic text normalization, WER/CER/SER, breakdowns
│       ├── model.py                       # Model loader, Hugging Face auth token, decoders
│       ├── reporter.py                    # Dual YAML report generator (Detailed & Summary)
│       └── engine.py                      # Core evaluation engine
└── evaluators/                            # Clean language-specific evaluation scripts
    ├── __init__.py
    ├── bhojpuri.py
    ├── bengali.py
    ├── chhattisgarhi.py
    ├── hindi.py
    ├── kannada.py
    ├── magahi.py
    ├── marathi.py
    ├── maithili.py
    ├── telugu.py
    └── run_all.py                         # Batch evaluation runner across all languages
```

---

## ⚙️ Installation & Setup

1. **Clone Repository & Install Dependencies:**
   ```bash
   git clone https://github.com/rimraf-adi/dialect-normalisation.git
   cd dialect-normalisation
   uv sync
   ```

2. **Hugging Face Authentication (Required for gated model access):**
   Set your Hugging Face Access Token as an environment variable or pass `--token`:
   ```bash
   # Windows PowerShell
   $env:HF_TOKEN="YOUR_HF_ACCESS_TOKEN"

   # Linux/macOS
   export HF_TOKEN="YOUR_HF_ACCESS_TOKEN"
   ```

---

## 🚀 Quick Start & Usage

### 1. Using the Unified CLI (`evaluate.py`)

Run evaluation for any language using the main entry point:

```bash
# Evaluate Marathi (Default)
uv run python evaluate.py --lang marathi

# Evaluate Hindi on 50 samples using RNNT decoding
uv run python evaluate.py --lang hindi --decoder rnnt --max-samples 50

# Evaluate Bengali using CUDA device and explicit token
uv run python evaluate.py --lang bengali --device cuda --token YOUR_HF_TOKEN

# Evaluate ALL 9 available languages
uv run python evaluate.py --lang all
```

---

### 2. Using Language-Specific Evaluators (`evaluators/`)

Each language has an independent evaluator script inside `evaluators/`:

```bash
uv run python evaluators/marathi.py --max-samples 20
uv run python evaluators/hindi.py
uv run python evaluators/telugu.py
uv run python evaluators/run_all.py --max-samples 10
```

---

### 3. Programmatic Python API

Import `dialect_norm` directly in Python:

```python
import dialect_norm
from dialect_norm.evaluators.marathi import evaluate_marathi
from dialect_norm.evaluators.hindi import evaluate_hindi

# Run Marathi Evaluation
evaluate_marathi(max_samples=50, decoder="both")

# Run Hindi Evaluation
evaluate_hindi(max_samples=50, decoder="both")
```

---

## 📊 Benchmark Output Reports (`baseline-indic-conformer/`)

For each language evaluation, **two YAML report files** are generated:

1. **`indic_conformer_<language>_detailed.yaml`**:
   - Full sample-wise itemization (utterance ID, reference vs. hypothesis, exact match boolean, raw/normalized WER & CER).
   - Complete sub-group breakdowns (Dialects D1-D5, Domains Agriculture/Banking, Gender, Age Group, Slab).

2. **`indic_conformer_<language>_summary.yaml`**:
   - Concise executive summary containing key metrics (overall RAW & NORMALIZED WER/CER/SER, Exact Match Accuracy %).
   - High-level dialect-wise and domain-wise normalized WER summaries.

---

## 🛠️ Metrics & Normalization

- **Text Normalization**: Removes Devanagari and general punctuation (`।`, `॥`, `?`, `!`, `,`, `.`, etc.) and collapses multi-space whitespace.
- **Evaluated Metrics**:
  - **WER** (Word Error Rate - Raw & Normalized)
  - **CER** (Character Error Rate - Raw & Normalized)
  - **SER** (Sentence Error Rate)
  - **Exact Match Accuracy %**
  - **Edit Distance Breakdown** (Substitutions, Deletions, Insertions, Hits)
