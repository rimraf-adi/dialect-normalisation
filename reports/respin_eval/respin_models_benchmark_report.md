# Official IISc RESPIN Held-Out Spoken Test Set Benchmark Report

**Test Dataset**: `IISc_RESPIN_test_mr` (2,170 Total Spoken Utterances)
**Evaluation Metrics**: BLEU (sacrebleu), chrF++ (word_order=2), Word Error Rate (WER %), Character Error Rate (CER %), Exact Match Accuracy (%)

---

## 1. Overall Combined Test Performance (2,170 Utterances)

| Model Variant | Training Scale / Quality | Checkpoint Path | BLEU Score | chrF++ Score | Word Error Rate (WER %) | Char Error Rate (CER %) | Exact Match Acc (%) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **IndicBART Combined 16k** | `indicbart_combined_16k` | `models\indicbart_combined\fold_1\best_model` | **61.93** | **75.37** | **31.17%** | 26.34% | 53.87% |
| **IndicBART Combined 32k** | `indicbart_combined_32k` | `models\indicbart_combined_32k\fold_1\best_model` | **65.14** | **79.15** | **26.83%** | 19.16% | 50.0% |
| **IndicBART Raw Unverified 32k** | `indicbart_raw_unverified_32k` | `models\indicbart_raw_unverified_32k\fold_1\best_model` | **61.67** | **75.26** | **31.46%** | 26.38% | 52.76% |
| **mT5-Small Combined 16k** | `mt5_combined_16k` | `models\mt5_combined_16k\best_model` | **73.02** | **87.9** | **16.83%** | 5.76% | 52.35% |
| **mT5-Small Combined 32k** | `mt5_combined_32k` | `models\mt5_combined_32k\best_model` | **74.44** | **88.24** | **15.96%** | 5.66% | 53.96% |
| **mT5-Small Raw Unverified 32k** | `mt5_raw_unverified_32k` | `models\mt5_raw_unverified_32k\best_model` | **74.39** | **88.19** | **16.23%** | 5.68% | 53.5% |

---

## 2. Granular Per-Dialect Breakdown across all Models

| Model Variant | D1 Malvani (559 utts)<br>BLEU / WER / chrF++ | D2 Ahirani (540 utts)<br>BLEU / WER / chrF++ | D3 Standard (555 utts)<br>BLEU / WER / chrF++ | D4 Varhadi (516 utts)<br>BLEU / WER / chrF++ |
| :--- | :---: | :---: | :---: | :---: |
| **IndicBART Combined 16k** | 20.19 / 63.83% / 46.49 | 69.58 / 25.24% / 79.89 | 94.47 / 5.04% / 96.58 | 64.4 / 29.08% / 77.95 |
| **IndicBART Combined 32k** | 39.04 / 44.07% / 65.17 | 69.25 / 24.08% / 80.54 | 87.1 / 11.45% / 91.85 | 65.82 / 26.88% / 79.77 |
| **IndicBART Raw Unverified 32k** | 21.93 / 62.8% / 47.74 | 69.28 / 25.56% / 79.64 | 92.17 / 6.99% / 95.24 | 64.28 / 29.08% / 78.01 |
| **mT5-Small Combined 16k** | 42.22 / 34.83% / 74.6 | 78.19 / 14.15% / 89.69 | 96.98 / 1.72% / 99.13 | 74.65 / 15.75% / 89.12 |
| **mT5-Small Combined 32k** | 44.72 / 34.61% / 74.8 | 79.15 / 12.66% / 90.2 | 97.74 / 1.07% / 99.22 | 76.45 / 14.64% / 89.68 |
| **mT5-Small Raw Unverified 32k** | 44.64 / 34.59% / 75.07 | 79.13 / 13.27% / 90.14 | 97.97 / 0.98% / 99.34 | 75.97 / 15.21% / 89.15 |

---

