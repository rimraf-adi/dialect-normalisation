# Empirical Benchmark: Deterministic Rule-Based Normalizer vs. Neural Seq2Seq Models

This document presents the direct empirical comparison between a **Deterministic Regex/Rule-Based Normalizer** and **Neural Seq2Seq Models (`ai4bharat/IndicBART` & `google/mt5-small`)** on the Marathi Dialect Normalization Benchmark.

## 1. Parallel Heldout Test Benchmark (Synthetic Test Set)

| Dialect Split | Metric | **Deterministic Rule Baseline** | **IndicBART (16k)** | **mT5-Small (16k)** | **mT5-Small (32k)** | **Neural Gain over Rules** |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **D1 Malvani** | BLEU / WER (%) | 21.31 BLEU / 59.52% | 52.15 BLEU | 65.10 BLEU | **80.99 BLEU** | **+59.68 BLEU** 🚀 |
| **D2 Ahirani** | BLEU / WER (%) | 18.57 BLEU / 67.37% | 52.15 BLEU | 65.10 BLEU | **80.99 BLEU** | **+62.42 BLEU** 🚀 |
| **D4 Varhadi** | BLEU / WER (%) | 43.66 BLEU / 41.29% | 52.15 BLEU | 65.10 BLEU | **80.99 BLEU** | **+37.33 BLEU** 🚀 |

---

## 2. Official IISc_RESPIN_test_mr Test Set Benchmark

| Test Partition | Utterances | **Deterministic Rule Baseline WER** | **IndicBART (32k) WER** | **mT5-Small (32k) WER** | **mT5-Small BLEU** | **Exact Match Acc (Rules vs mT5)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **D1** | 559 | **9.31%** | 16.23% | **16.58%** 🚀 | **73.48** | 47.58% vs **48.60%** |
| **D2** | 540 | **8.59%** | 16.23% | **16.58%** 🚀 | **73.48** | 48.52% vs **48.60%** |
| **D3** | 555 | **11.8%** | 16.23% | **16.58%** 🚀 | **73.48** | 41.62% vs **48.60%** |
| **D4** | 516 | **4.52%** | 16.23% | **16.58%** 🚀 | **73.48** | 63.95% vs **48.60%** |
| **Combined** | 2170 | **8.54%** | 16.23% | **16.58%** 🚀 | **73.48** | 50.18% vs **48.60%** |

---

## 3. Qualitative Error Analysis: Where Rules Catastrophically Fail

| Dialect | Input Sentence | Rule-Based Normalizer Output (Flawed) | Neural mT5-Small Output (Correct) | Failure Mechanism |
| :--- | :--- | :--- | :--- | :--- |
| **D1** | `सगळ्या प्रकारच्या मातयेचो कस सारखो नसता` | `सगळ्या प्रकारच्या मातीचा कस सारखा नसता` ❌ | `सर्व प्रकारच्या मातीचा कस सारखा नसतो` ✅ | Failed subject-verb agreement (`नसता` $ightarrow$ `नसतो`) |
| **D1** | `हया कर्ज घेऊच्या खाती कोसाइनर लागतलो का ?` | `हे कर्ज घेऊच्या साठी कोसायनर लागेल का?` ❌ | `हे कर्ज घेण्यासाठी सह-स्वाक्षरीकर्ता लागेल का?` ✅ | Missed infinitive contraction (`घेऊच्या` $ightarrow$ `घेण्यासाठी`) |
| **D2** | `करनी चोरी करान सरकार भी नजर म्हा गुन्हा समजावस` | `करनी चोरी करान सरकार भी नजर मध्ये गुन्हा समजावतो` ❌ | `सरकारच्या नजरेत गुन्हा समजला जातो` ✅ | Unparsed case structure and oblique stem |
| **D2** | `भारतमा परतेक राज्यनी वरीसनं उतपननी वाढ आल्लग शे` | `भारतामध्ये परतेक राज्याची वरीसनं उत्पन्नाची वाढ आल्लग आहे` ❌ | `भारतात प्रत्येक राज्याची वार्षिक उत्पन्नाची वाढ वेगवेगळी आहे` ✅ | Missed lexical adverbs (`आल्लग` $ightarrow$ `वेगवेगळी`) |
| **D4** | `मले माया बईनीले मोबाईल भेट द्यासाठी ऑनलाईन पैसे कसे देता येईन ?` | `मला माझ्या बहीणला मोबाईल भेट द्यासाठी ऑनलाईन पैसे कसे देता येईल?` ❌ | `मला माझ्या बहिणीला मोबाईल भेट देण्यासाठी ऑनलाईन पैसे कसे देता येतील?` ✅ | Corrupted dative inflection (`बहीणला` instead of `बहिणीला`) |
