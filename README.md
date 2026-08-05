# Dialect Normalisation & Indic ASR Benchmarking Suite

An automated evaluation and fine-tuning framework for benchmarking:
1. **IndicConformer** (`ai4bharat/indic-conformer-600m-multilingual`) ASR across Indian languages and dialect variations using **IISc RESPIN** datasets.
2. **Seq2Seq Dialect Normalization** (`google/mt5-small` & `ai4bharat/IndicBART`) fine-tuned across 32,335 clean parallel pairs for non-standard Marathi dialects (**D1 Malvani**, **D2 Ahirani**, **D4 Varhadi**) into Standard Marathi.

---

## 🏆 Key Benchmark Highlights (Seq2Seq Dialect Normalization)

| Model Architecture | Combined 16k BLEU | Combined 32k BLEU | Combined 32k WER (%) | **Relative WER Reduction (%)** | **Varhadi D4 Peak BLEU** | **Malvani D1 Peak BLEU** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `ai4bharat/IndicBART` (244M) | 48.17 | 63.59 | 27.16% | -31.07% | 79.51 | 46.12 |
| **`google/mt5-small` (300M)** | **62.68** 🚀 | **69.05** 🚀 | **21.43%** 🔥 | **-45.62%** 🔥 | **79.83** 🔥 | **54.84** 🚀 |

*Detailed comparative analysis is documented in [docs/results.md](file:///d:/dialect-norm/docs/results.md).*

---

## 🌟 Supported Languages & Datasets (ASR)

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
├── pyproject.toml                         # Package configuration & CLI script entry points
├── README.md                              # Main framework documentation
├── docs/                                  # Benchmark documentation & logs
│   ├── results.md                         # Full Comparative Benchmark Report
│   └── logs.tex                           # Implementation log file
├── evaluate.py                            # Unified ASR CLI entry point
├── src/                                   # Core dialect_norm Python Package
│   └── dialect_norm/
│       ├── training/                      # Seq2Seq Fine-tuning Engines & Runner Suites
│       │   ├── mt5_trainer.py             # google/mt5-small fine-tuning engine
│       │   ├── indicbart_trainer.py       # IndicBART fine-tuning engine
│       │   ├── indictrans2_trainer.py     # IndicTrans2 fine-tuning engine
│       │   ├── run_sequential_mt5_suite.py
│       │   └── run_sequential_indictrans2_suite.py
│       ├── audio.py                       # Audio loader & 16kHz resampling
│       ├── metrics.py                     # Indic text normalization & WER/CER
│       └── model.py                       # Model loader
└── models/                                # Fine-tuned model checkpoints & cv_summary.yaml
```

---

## ⚙️ Installation & Setup

1. **Clone Repository & Install Dependencies:**
   ```bash
   git clone https://github.com/rimraf-adi/dialect-normalisation.git
   cd dialect-normalisation
   uv sync
   ```

2. **Hugging Face Authentication (Required for gated models):**
   ```bash
   # Windows PowerShell
   $env:HF_TOKEN="YOUR_HF_ACCESS_TOKEN"

   # Linux/macOS
   export HF_TOKEN="YOUR_HF_ACCESS_TOKEN"
   ```

---

## 🚀 Quick Start & Usage

### 1. Sequential Fine-Tuning Suites (Seq2Seq Normalization)

Run sequential fine-tuning across all 8 dataset variants (16k and 32k for D1, D2, D4, D124):

```bash
# Run mT5-small Sequential Suite (300M)
uv run train-mt5-suite

# Run IndicTrans2 Sequential Suite (dist-320M)
uv run train-indictrans2-suite
```

Individual variant entry points:
```bash
uv run train-mt5-all-32k
uv run train-mt5-d1-32k
uv run train-mt5-d2-32k
uv run train-mt5-d4-32k
```

---

### 2. Using the Unified ASR CLI (`evaluate.py`)

Run ASR evaluation for any language:

```bash
# Evaluate Marathi (Default)
uv run python evaluate.py --lang marathi

# Evaluate Hindi on 50 samples using RNNT decoding
uv run python evaluate.py --lang hindi --decoder rnnt --max-samples 50

# Evaluate ALL 9 available languages
uv run python evaluate.py --lang all
```

---

## 📊 Benchmark Output Reports (`baseline-indic-conformer/` & `models/`)

For each fine-tuned model and ASR evaluation, **structured YAML summary reports** are generated:

1. **`models/<model_variant>/cv_summary.yaml`**:
   - 5-Fold Cross-Validation metrics (Val/Test Loss, BLEU score, chrF++).
   - Per-dialect test breakdown (D1 Malvani, D2 Ahirani, D4 Varhadi).

2. **`baseline-indic-conformer/indic_conformer_<lang>_summary.yaml`**:
   - Concise ASR summary containing overall RAW & NORMALIZED WER/CER/SER metrics.

---

## 🛠️ Metrics & Normalization

- **Text Normalization**: Removes Devanagari and general punctuation (`।`, `॥`, `?`, `!`, `,`, `.`, etc.) and collapses multi-space whitespace.
- **Evaluated Metrics**:
  - **BLEU & chrF++** (SacredBLEU & chrF++ for Seq2Seq Dialect Normalization)
  - **WER / CER / SER** (Word, Character, and Sentence Error Rate for ASR)
  - **Exact Match Accuracy %**
