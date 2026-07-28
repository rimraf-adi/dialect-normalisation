# Text-to-Text Dialect Normalisation for Marathi

## Feasibility Study: Can a Text-to-Text Model Translate Marathi Dialects to Standard Marathi?

---

## 1. Problem Statement

**Core Question:** Can a text-to-text sequence-to-sequence model translate non-standard Marathi dialect text (D1 South Konkan, D2 North Konkan, D4 Varhadi) into Standard Pune Marathi (D3)?

This is a **pure text-level** problem. No speech or ASR is involved. Given a sentence written in a regional Marathi dialect, the model must produce the equivalent sentence in standard formal Marathi.

### Why This Matters
- Rural voice interfaces (agriculture helplines, banking bots) receive dialect text from upstream systems
- Downstream NLU (intent classifiers, slot-fillers, translation engines) expect standard Marathi
- Re-training every upstream model per dialect is prohibitively expensive
- A lightweight, modular text-to-text normaliser can be dropped into any pipeline

---

## 2. Literature Survey

### 2.1 Language-Agnostic Dialect Normalisation

| Domain | Approach | Key Finding | Reference |
| :--- | :--- | :--- | :--- |
| **Swiss German → Standard German** | Character-level Transformer (byT5, sliding window) | Character-level Transformers outperform word-level models for normalisation; treats it as seq2seq transduction without changing word order | GSWNORM Shared Task, SwissText 2024; ACL Anthology |
| **Arabic Dialects → Modern Standard Arabic** | AraT5-MSAizer (Transformer encoder-decoder) | Fine-tuned T5-style model on 5 regional Arabic dialects; achieved BLEU ~21.79%; joint multi-dialect training outperforms single-dialect training | OSACT 2024, ACL Anthology |
| **Arabic Dialects → MSA (LLM-based)** | Few-shot prompting (GPT-4, Jais, AceGPT) | Arabic-centric LLMs (Jais) outperform generic LLMs (LLaMA) for dialect-to-standard translation; few-shot > zero-shot > chain-of-thought | SpringerNature 2024, arXiv |
| **Hindi/Odia Dialects** | INDIC-DIALECT benchmark (13K parallel pairs, 11 dialects) | Hybrid AI model achieved BLEU 61.32 for dialect→standard vs. baseline 23.36; fine-tuned transformers improved F1 from 19.6% → 89.8% for dialect classification | arXiv 2026 |
| **General Low-Resource** | Context-Aware Prompting (CAP) | Embedding linguistic rulebooks + dialect dictionaries into LLM prompts improves dialect-specific accuracy over vanilla prompting | ACL Anthology 2024 |
| **General Low-Resource** | TopXGen (topic-diverse synthetic data) | LLM-generated topic-diverse parallel data + back-translation creates effective training corpora for low-resource MT | arXiv 2024 |
| **Parameter-Efficient** | LoRA fine-tuning of LLMs | LoRA adapters on multilingual models achieve high dialect translation quality with minimal labelled data | ACL Anthology 2024 |

### 2.2 Marathi-Specific NLP Resources

| Resource | Description | Relevance |
| :--- | :--- | :--- |
| **MahaBERT / MahaBERT-v2** (L3Cube Pune) | BERT-Base fine-tuned on Marathi monolingual corpus; encoder-only | Useful for dialect classification but NOT for generation/translation |
| **mahaNLP** library | Python library for Marathi NLP (POS, NER, sentiment) | Foundation tooling; no dialect translation capability |
| **IndicBART** (AI4Bharat) | Seq2seq model pre-trained on 11 Indic languages | Strong candidate for fine-tuning on dialect→standard pairs |
| **IndicTrans2** (AI4Bharat) | Open-source NMT for 22 scheduled Indian languages | Inter-language translation; not designed for intra-language dialect normalisation |
| **mT5** (Google) | Massively multilingual text-to-text Transformer | General-purpose; can be fine-tuned for Marathi dialect normalisation |

### 2.3 Gap Analysis — What Has NOT Been Done

> **No published work exists that performs text-to-text dialect normalisation specifically for Marathi sub-dialects (D1–D4).**

- INDIC-DIALECT (2026) covers Hindi and Odia dialects but **not Marathi**
- MahaBERT/mahaNLP are encoder-only models (classification, not generation)
- IndicBART/mT5 have not been fine-tuned for intra-Marathi dialect translation
- IISc RESPIN provides dialectal speech data but **no parallel text pairs across dialects**

**This is the research gap our work fills.**

---

## 3. Dataset Reality Check

### 3.1 IISc RESPIN Marathi Test Corpus

| Dialect | Region | Utterances | Unique Texts | Avg Words/Sentence |
| :--- | :--- | :--- | :--- | :--- |
| **D1** | South Konkan (Sindhudurg/Ratnagiri) | 559 | 212 | 8.7 |
| **D2** | North Konkan (Palghar/Thane) | 540 | 152 | 8.8 |
| **D3** | Standard Pune (Target) | 555 | 149 | 8.4 |
| **D4** | Varhadi (Vidarbha/Amravati) | 516 | 198 | 8.9 |

### 3.2 Critical Constraint: No Parallel Data

- **Zero** shared text strings across dialects
- **Zero** shared text IDs across dialects
- Each dialect has completely independent sentences on Agriculture and Banking topics
- **We cannot directly pair D1 sentences with D3 equivalents**

### 3.3 Implication: We Must Generate Synthetic Parallel Data

Since no natural parallel corpus exists, we follow the established approach from Arabic (AraT5-MSAizer) and Swiss German (GSWNORM) literature: **use an LLM to generate synthetic parallel pairs**.

---

## 4. Proposed Methodology

### 4.1 High-Level Pipeline

```
┌─────────────────────────────────────────────────────┐
│                SYNTHETIC DATA GENERATION             │
│                                                     │
│  D1/D2/D4 Reference Texts                           │
│         │                                           │
│         ▼                                           │
│  [gemma4:12b via Ollama]                            │
│  Prompt: "Translate this Marathi dialect sentence    │
│           to Standard Pune Marathi"                  │
│         │                                           │
│         ▼                                           │
│  Synthetic Parallel Pairs:                           │
│    (dialect_text, standard_text)                     │
│                                                     │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│              MODEL FINE-TUNING                       │
│                                                     │
│  Candidate Models:                                   │
│    • mT5-small / mT5-base                           │
│    • IndicBART / IndicBARTSS                        │
│    • ByT5-small (character-level)                    │
│                                                     │
│  Training: dialect_text → standard_text              │
│  Validation: held-out split                          │
│                                                     │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│                  EVALUATION                          │
│                                                     │
│  Metrics: BLEU, chrF++, BERTScore, Exact Match      │
│  Baselines:                                         │
│    • Identity (no change)                           │
│    • Direct LLM prompting (gemma4:12b zero-shot)    │
│    • Rule-based substitution dictionary              │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### 4.2 Phase 1: Synthetic Parallel Data Generation

**Tool:** `gemma4:12b` via local Ollama server (verified active, 50.72 tok/s)

**Strategy:**
1. Take all unique dialect reference texts from D1 (212), D2 (152), D4 (198) = **562 unique source sentences**
2. Prompt gemma4:12b to translate each into Standard Pune Marathi (D3-style)
3. Apply quality filtering:
   - Remove pairs where output is identical to input (no normalization happened)
   - Remove pairs where output is empty or garbled
   - Remove pairs where output language is not Marathi (script check)
4. Generate **reverse pairs** from D3 texts: prompt LLM to generate dialectal variants → more training data
5. Target: **~1,500–3,000 parallel pairs** after filtering

**Prompt Template (Context-Aware Prompting):**
```
You are a Marathi language expert. The following sentence is written in
{dialect_name} dialect of Marathi (spoken in {region}).

Translate it into Standard Pune Marathi (शुद्ध पुणेरी मराठी) while preserving
the original meaning. Only change dialectal words and grammar to their
standard equivalents. Do not add or remove information.

Dialect sentence: {dialect_text}

Standard Marathi translation:
```

### 4.3 Phase 2: Model Fine-Tuning

**Candidate Models (ranked by suitability):**

| Model | Parameters | Why Consider | Why Hesitate |
| :--- | :--- | :--- | :--- |
| **IndicBART** | 244M | Pre-trained on 11 Indic languages including Marathi; seq2seq | Older architecture; limited dialect exposure |
| **mT5-small** | 300M | Massively multilingual; proven on similar tasks (AraT5-MSAizer) | Not Indic-specialized |
| **mT5-base** | 580M | Larger capacity; better for nuanced linguistic shifts | Higher compute cost |
| **ByT5-small** | 300M | Character-level; handles orthographic variation natively | Slower inference; less semantic awareness |

**Training Configuration:**
- Input format: `normalize dialect: {dialect_text}`
- Output format: `{standard_text}`
- Optimizer: AdamW, lr=3e-4 with linear warmup
- Batch size: 8–16
- Epochs: 10–30 (small dataset, risk of overfitting → use early stopping)
- Validation: 15% held-out split

### 4.4 Phase 3: Evaluation

**Automatic Metrics:**
- **BLEU** (SacreBLEU): Standard MT metric; measures n-gram overlap with reference
- **chrF++**: Character-level F-score; better for morphologically rich languages like Marathi
- **BERTScore**: Semantic similarity using multilingual BERT embeddings
- **Exact Match %**: Fraction of outputs identical to reference
- **Word Error Rate (WER)**: Edit distance ratio (reuse our existing `dialect_norm.metrics`)

**Baselines to Compare Against:**
1. **Identity Baseline**: Output = Input (no normalization). Measures how different dialect text already is from standard.
2. **Zero-shot LLM**: Direct gemma4:12b prompting without fine-tuning. Measures raw LLM capability.
3. **Rule-based Dictionary**: Hand-crafted word substitution table for common dialectal words.
4. **Fine-tuned mT5/IndicBART**: Our proposed model.

**Evaluation Sets:**
- Held-out 15% of synthetic parallel data
- Manual human evaluation on 50–100 samples (fluency + adequacy scores 1–5)

---

## 5. Experimental Design

### Experiment 1: Zero-Shot LLM Dialect Translation
- **Goal:** Establish how well gemma4:12b handles Marathi dialect normalization out-of-the-box
- **Method:** Prompt all 562 unique dialect texts through gemma4:12b, evaluate against human-corrected references
- **Expected Outcome:** Moderate quality; LLM may struggle with rare dialectal vocabulary

### Experiment 2: Fine-Tuned Seq2Seq Model
- **Goal:** Train a lightweight, deployable model on synthetic parallel data
- **Method:** Fine-tune mT5-small and IndicBART on generated pairs
- **Expected Outcome:** Should outperform zero-shot LLM on in-domain dialect text

### Experiment 3: Dialect-Specific vs. Multi-Dialect Training
- **Goal:** Test whether a single model trained on all dialects (D1+D2+D4) outperforms dialect-specific models
- **Method:** Compare:
  - Model A: Trained only on D1→Standard pairs
  - Model B: Trained only on D4→Standard pairs
  - Model C: Trained on all D1+D2+D4→Standard pairs jointly
- **Expected Outcome:** Joint training should win (consistent with Arabic dialect literature finding)

### Experiment 4: Character-Level vs. Subword-Level
- **Goal:** Compare ByT5 (character-level) against mT5 (subword-level) for dialect normalization
- **Method:** Same training data, same evaluation
- **Expected Outcome:** ByT5 may handle orthographic variation better; mT5 may handle semantic shifts better

---

## 6. Implementation Plan

### Step 1: Synthetic Data Generation Script
- `src/dialect_norm/data_gen.py`: Script to prompt gemma4:12b via Ollama API and generate parallel pairs
- Output: `data/synthetic_parallel/d1_to_standard.jsonl`, `d2_to_standard.jsonl`, `d4_to_standard.jsonl`

### Step 2: Data Quality Filtering
- `src/dialect_norm/data_filter.py`: Script to filter low-quality pairs (empty, non-Marathi, identity copies)
- Output: `data/synthetic_parallel/filtered_all.jsonl`

### Step 3: Model Training Scripts
- `src/dialect_norm/train_t2t.py`: HuggingFace Trainer script for mT5/IndicBART fine-tuning
- `configs/`: Training hyperparameter YAML configs

### Step 4: Evaluation Scripts
- `src/dialect_norm/eval_t2t.py`: Evaluate trained model using BLEU, chrF++, BERTScore, WER
- Output: `results/` directory with evaluation YAML reports

### Step 5: Analysis & Reporting
- Per-dialect breakdown tables
- Error analysis (what types of dialectal constructions the model handles well vs. poorly)
- Comparison table against baselines

---

## 7. Expected Contributions

1. **First text-to-text dialect normalisation system for Marathi sub-dialects** (D1 South Konkan, D2 North Konkan, D4 Varhadi → D3 Standard Pune)
2. **Synthetic parallel corpus** for Marathi dialect normalisation (generated via LLM, quality-filtered)
3. **Empirical comparison** of fine-tuned seq2seq models (mT5, IndicBART, ByT5) vs. LLM zero-shot prompting for intra-language dialect translation
4. **Feasibility validation** answering: *Does text-to-text dialect normalisation work for Marathi?*

---

## 8. Risks & Mitigations

| Risk | Severity | Mitigation |
| :--- | :--- | :--- |
| Synthetic data quality is poor (LLM hallucinations) | High | Quality filtering pipeline + manual spot-checking of 100 samples |
| Fine-tuned model overfits on small dataset | Medium | Early stopping, dropout, data augmentation via back-translation |
| gemma4:12b doesn't understand Marathi dialects well enough | Medium | Try multiple prompt templates; fall back to IndicTrans2 or GPT-4 API if needed |
| No ground-truth parallel data for proper evaluation | High | Human evaluation on subset; cross-validate with multiple LLM-generated references |
| Character-level model too slow for practical deployment | Low | Acceptable for feasibility study; optimize later |

---

## 9. References

1. GSWNORM Shared Task, "Swiss German Dialect Normalization," SwissText/KONVENS, 2024.
2. AraT5-MSAizer, "Dialectal Arabic to MSA Translation," OSACT Workshop, ACL, 2024.
3. INDIC-DIALECT, "A Multi-Task Benchmark for Indian Language Dialects," arXiv, 2026.
4. L3Cube-MahaBERT, "Marathi Monolingual Corpus, Marathi BERT Language Models," WILDRE-6, LREC, 2022.
5. AI4Bharat, "IndicBART: Pre-trained Seq2Seq Model for Indic Languages," ACL Findings, 2022.
6. Xue et al., "mT5: A Massively Multilingual Pre-trained Text-to-Text Transformer," NAACL, 2021.
7. Context-Aware Prompting for Dialect Translation, ACL Anthology, 2024.
8. TopXGen: Topic-Diverse Synthetic Data for Low-Resource MT, arXiv, 2024.
9. IISc RESPIN Consortium, "Resilient Speech Recognition in Indic Languages Benchmark," 2023.
10. AI4Bharat, "IndicTrans2: Towards High-Quality and Accessible MT for Indian Languages," TMLR, 2024.
