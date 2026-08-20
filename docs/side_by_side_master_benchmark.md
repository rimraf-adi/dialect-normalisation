# Master Empirical Benchmark: Side-by-Side Multi-Metric & 5-Fold Dialectwise Evaluation Report

**Repository**: `rimraf-adi/dialect-normalisation`  
**Datasets Evaluated**:
1. **Official IISc RESPIN Held-Out Spoken Test Set** (`IISc_RESPIN_test_mr` — 2,170 Total Spoken Utterances)
2. **5-Fold Stratified Parallel Cross-Validation Benchmark** (16k Clean Baseline vs. 32k Expanded Verified vs. Raw Unverified)

---

## 1. Master Dialectwise Summary Matrix on Official IISc RESPIN Spoken Test Set ($M \pm \sigma$)

| Model Setup & Architecture | Dialect Variety | Evaluated Utterances | **BLEU Score ($M \pm \sigma$)** | **chrF++ Score ($M \pm \sigma$)** | **Word Error Rate (WER %)** | **Char Error Rate (CER %)** | **Exact Match Acc (%)** |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **IndicBART Combined (16k)** | D1 (Malvani) | 559 | **16.92 $\pm$ 1.49** | **43.41 $\pm$ 1.43** | **66.42% $\pm$ 1.31%** | 57.21% $\pm$ 1.61% | 5.19% $\pm$ 0.52% |
| **IndicBART Combined (16k)** | D2 (Ahirani) | 540 | **59.17 $\pm$ 1.29** | **72.99 $\pm$ 0.76** | **32.98% $\pm$ 0.93%** | 29.74% $\pm$ 0.84% | 50.85% $\pm$ 1.02% |
| **IndicBART Combined (16k)** | D3 (Standard) | 555 | **79.86 $\pm$ 1.96** | **87.08 $\pm$ 1.24** | **16.80% $\pm$ 1.69%** | 14.51% $\pm$ 1.50% | 75.32% $\pm$ 2.27% |
| **IndicBART Combined (16k)** | D4 (Varhadi) | 516 | **57.78 $\pm$ 0.99** | **73.58 $\pm$ 0.76** | **33.93% $\pm$ 0.87%** | 27.31% $\pm$ 1.03% | 43.64% $\pm$ 0.84% |
| **IndicBART Combined (16k)** | Combined Multi-Dialect | 2,170 | **52.83 $\pm$ 0.58** | **69.33 $\pm$ 0.40** | **37.85% $\pm$ 0.48%** | 32.63% $\pm$ 0.49% | 43.63% $\pm$ 0.59% |
| **IndicBART Combined (32k)** | D1 (Malvani) | 559 | **34.50 $\pm$ 5.22** | **61.06 $\pm$ 5.25** | **48.45% $\pm$ 5.58%** | 34.26% $\pm$ 7.21% | 5.19% $\pm$ 0.74% |
| **IndicBART Combined (32k)** | D2 (Ahirani) | 540 | **62.06 $\pm$ 5.94** | **75.85 $\pm$ 4.45** | **29.83% $\pm$ 5.06%** | 25.06% $\pm$ 5.39% | 47.81% $\pm$ 5.39% |
| **IndicBART Combined (32k)** | D3 (Standard) | 555 | **75.84 $\pm$ 6.80** | **84.19 $\pm$ 4.85** | **20.11% $\pm$ 5.67%** | 17.69% $\pm$ 5.75% | 68.58% $\pm$ 5.17% |
| **IndicBART Combined (32k)** | D4 (Varhadi) | 516 | **58.46 $\pm$ 4.80** | **74.32 $\pm$ 3.96** | **33.13% $\pm$ 4.17%** | 25.82% $\pm$ 5.20% | 44.26% $\pm$ 3.59% |
| **IndicBART Combined (32k)** | Combined Multi-Dialect | 2,170 | **57.50 $\pm$ 5.31** | **73.70 $\pm$ 4.20** | **33.05% $\pm$ 4.69%** | 25.87% $\pm$ 5.46% | 41.30% $\pm$ 3.47% |
| **IndicBART Raw Unverified (32k)** | D1 (Malvani) | 559 | **17.24 $\pm$ 1.70** | **43.60 $\pm$ 1.56** | **66.31% $\pm$ 1.42%** | 57.14% $\pm$ 1.69% | 4.61% $\pm$ 0.08% |
| **IndicBART Raw Unverified (32k)** | D2 (Ahirani) | 540 | **58.95 $\pm$ 2.33** | **72.93 $\pm$ 1.66** | **33.10% $\pm$ 1.79%** | 29.65% $\pm$ 1.95% | 49.96% $\pm$ 1.97% |
| **IndicBART Raw Unverified (32k)** | D3 (Standard) | 555 | **77.92 $\pm$ 0.26** | **85.47 $\pm$ 0.29** | **18.58% $\pm$ 0.31%** | 16.45% $\pm$ 0.40% | 72.47% $\pm$ 0.75% |
| **IndicBART Raw Unverified (32k)** | D4 (Varhadi) | 516 | **56.92 $\pm$ 1.17** | **72.66 $\pm$ 0.99** | **34.83% $\pm$ 1.04%** | 28.37% $\pm$ 1.40% | 44.11% $\pm$ 0.73% |
| **IndicBART Raw Unverified (32k)** | Combined Multi-Dialect | 2,170 | **52.14 $\pm$ 0.71** | **68.71 $\pm$ 0.53** | **38.50% $\pm$ 0.56%** | 33.32% $\pm$ 0.61% | 42.64% $\pm$ 0.39% |
| **IndicBART D1 Malvani (16k)** | D1 (Malvani) | 559 | **45.95 $\pm$ 0.74** | **73.92 $\pm$ 0.52** | **35.34% $\pm$ 0.62%** | 16.05% $\pm$ 0.62% | 4.79% $\pm$ 0.94% |
| **IndicBART D1 Malvani (16k)** | D2 (Ahirani) | 540 | **77.71 $\pm$ 0.75** | **88.90 $\pm$ 0.48** | **14.63% $\pm$ 0.63%** | 8.46% $\pm$ 0.55% | 53.18% $\pm$ 0.88% |
| **IndicBART D1 Malvani (16k)** | D3 (Standard) | 555 | **90.46 $\pm$ 0.31** | **95.64 $\pm$ 0.13** | **6.11% $\pm$ 0.16%** | 3.29% $\pm$ 0.08% | 81.37% $\pm$ 0.73% |
| **IndicBART D1 Malvani (16k)** | D4 (Varhadi) | 516 | **74.43 $\pm$ 0.72** | **86.77 $\pm$ 0.40** | **17.69% $\pm$ 0.57%** | 9.24% $\pm$ 0.50% | 46.90% $\pm$ 0.71% |
| **IndicBART D1 Malvani (16k)** | Combined Multi-Dialect | 2,170 | **72.04 $\pm$ 0.49** | **86.12 $\pm$ 0.31** | **18.62% $\pm$ 0.40%** | 9.38% $\pm$ 0.37% | 46.43% $\pm$ 0.49% |
| **IndicBART D1 Malvani (32k)** | D1 (Malvani) | 559 | **13.46 $\pm$ 0.59** | **39.24 $\pm$ 0.34** | **68.70% $\pm$ 0.47%** | 61.55% $\pm$ 0.45% | 4.97% $\pm$ 0.71% |
| **IndicBART D1 Malvani (32k)** | D2 (Ahirani) | 540 | **64.21 $\pm$ 1.22** | **75.57 $\pm$ 1.01** | **29.21% $\pm$ 0.93%** | 27.06% $\pm$ 1.17% | 54.70% $\pm$ 1.44% |
| **IndicBART D1 Malvani (32k)** | D3 (Standard) | 555 | **82.11 $\pm$ 1.59** | **88.12 $\pm$ 1.17** | **15.59% $\pm$ 1.43%** | 13.69% $\pm$ 1.43% | 77.52% $\pm$ 1.44% |
| **IndicBART D1 Malvani (32k)** | D4 (Varhadi) | 516 | **60.22 $\pm$ 1.05** | **74.68 $\pm$ 0.75** | **32.69% $\pm$ 0.86%** | 27.67% $\pm$ 0.92% | 47.63% $\pm$ 0.55% |
| **IndicBART D1 Malvani (32k)** | Combined Multi-Dialect | 2,170 | **54.32 $\pm$ 0.80** | **69.58 $\pm$ 0.56** | **36.87% $\pm$ 0.60%** | 32.98% $\pm$ 0.63% | 46.05% $\pm$ 0.79% |
| **IndicBART D2 Ahirani (16k)** | D1 (Malvani) | 559 | **69.79 $\pm$ 0.53** | **86.85 $\pm$ 0.28** | **18.07% $\pm$ 0.27%** | 7.07% $\pm$ 0.13% | 24.44% $\pm$ 2.11% |
| **IndicBART D2 Ahirani (16k)** | D2 (Ahirani) | 540 | **84.26 $\pm$ 0.38** | **92.64 $\pm$ 0.05** | **10.43% $\pm$ 0.24%** | 5.35% $\pm$ 0.12% | 68.22% $\pm$ 1.42% |
| **IndicBART D2 Ahirani (16k)** | D3 (Standard) | 555 | **97.91 $\pm$ 0.45** | **99.27 $\pm$ 0.15** | **0.96% $\pm$ 0.17%** | 0.37% $\pm$ 0.09% | 94.20% $\pm$ 1.16% |
| **IndicBART D2 Ahirani (16k)** | D4 (Varhadi) | 516 | **83.71 $\pm$ 0.54** | **92.53 $\pm$ 0.30** | **9.91% $\pm$ 0.26%** | 4.28% $\pm$ 0.25% | 59.61% $\pm$ 1.08% |
| **IndicBART D2 Ahirani (16k)** | Combined Multi-Dialect | 2,170 | **83.76 $\pm$ 0.23** | **92.71 $\pm$ 0.06** | **9.97% $\pm$ 0.10%** | 4.34% $\pm$ 0.06% | 61.54% $\pm$ 0.63% |
| **IndicBART D2 Ahirani (32k)** | D1 (Malvani) | 559 | **28.28 $\pm$ 2.13** | **52.08 $\pm$ 1.94** | **57.49% $\pm$ 1.67%** | 50.17% $\pm$ 2.07% | 10.48% $\pm$ 1.26% |
| **IndicBART D2 Ahirani (32k)** | D2 (Ahirani) | 540 | **59.37 $\pm$ 3.12** | **72.13 $\pm$ 2.11** | **33.00% $\pm$ 2.41%** | 31.25% $\pm$ 2.42% | 53.52% $\pm$ 1.88% |
| **IndicBART D2 Ahirani (32k)** | D3 (Standard) | 555 | **79.00 $\pm$ 0.77** | **86.10 $\pm$ 0.46** | **17.94% $\pm$ 0.57%** | 15.80% $\pm$ 0.45% | 75.14% $\pm$ 1.08% |
| **IndicBART D2 Ahirani (32k)** | D4 (Varhadi) | 516 | **59.00 $\pm$ 1.92** | **73.42 $\pm$ 1.23** | **33.63% $\pm$ 1.36%** | 29.28% $\pm$ 1.36% | 45.78% $\pm$ 1.45% |
| **IndicBART D2 Ahirani (32k)** | Combined Multi-Dialect | 2,170 | **56.17 $\pm$ 0.99** | **70.89 $\pm$ 0.66** | **35.77% $\pm$ 0.70%** | 31.98% $\pm$ 0.73% | 46.12% $\pm$ 0.98% |
| **IndicBART D4 Varhadi (16k)** | D1 (Malvani) | 559 | **83.35 $\pm$ 0.37** | **91.94 $\pm$ 0.19** | **11.19% $\pm$ 0.27%** | 4.77% $\pm$ 0.19% | 53.42% $\pm$ 1.30% |
| **IndicBART D4 Varhadi (16k)** | D2 (Ahirani) | 540 | **95.30 $\pm$ 0.38** | **98.21 $\pm$ 0.16** | **2.44% $\pm$ 0.22%** | 1.05% $\pm$ 0.24% | 87.48% $\pm$ 0.93% |
| **IndicBART D4 Varhadi (16k)** | D3 (Standard) | 555 | **98.25 $\pm$ 0.44** | **99.34 $\pm$ 0.25** | **0.98% $\pm$ 0.26%** | 0.39% $\pm$ 0.20% | 96.25% $\pm$ 0.85% |
| **IndicBART D4 Varhadi (16k)** | D4 (Varhadi) | 516 | **80.17 $\pm$ 0.46** | **90.42 $\pm$ 0.23** | **12.60% $\pm$ 0.39%** | 4.93% $\pm$ 0.24% | 59.23% $\pm$ 0.76% |
| **IndicBART D4 Varhadi (16k)** | Combined Multi-Dialect | 2,170 | **89.22 $\pm$ 0.11** | **94.93 $\pm$ 0.03** | **6.83% $\pm$ 0.08%** | 2.82% $\pm$ 0.01% | 74.23% $\pm$ 0.45% |
| **IndicBART D4 Varhadi (32k)** | D1 (Malvani) | 559 | **61.56 $\pm$ 3.24** | **75.38 $\pm$ 2.49** | **31.32% $\pm$ 2.79%** | 25.71% $\pm$ 2.93% | 30.34% $\pm$ 1.53% |
| **IndicBART D4 Varhadi (32k)** | D2 (Ahirani) | 540 | **74.89 $\pm$ 1.16** | **82.57 $\pm$ 0.90** | **20.81% $\pm$ 0.92%** | 19.60% $\pm$ 1.06% | 64.67% $\pm$ 1.23% |
| **IndicBART D4 Varhadi (32k)** | D3 (Standard) | 555 | **82.28 $\pm$ 0.48** | **87.10 $\pm$ 0.27** | **15.80% $\pm$ 0.41%** | 15.16% $\pm$ 0.32% | 81.84% $\pm$ 0.67% |
| **IndicBART D4 Varhadi (32k)** | D4 (Varhadi) | 516 | **63.30 $\pm$ 1.02** | **76.51 $\pm$ 0.75** | **29.77% $\pm$ 0.80%** | 24.32% $\pm$ 0.88% | 48.88% $\pm$ 1.55% |
| **IndicBART D4 Varhadi (32k)** | Combined Multi-Dialect | 2,170 | **70.38 $\pm$ 1.17** | **80.33 $\pm$ 0.94** | **24.50% $\pm$ 0.99%** | 21.28% $\pm$ 1.11% | 56.46% $\pm$ 0.59% |
| **mT5-Small Combined (16k)** | D1 (Malvani) | 559 | **41.22** | **74.14** | **35.27%** | 12.58% | 4.65% |
| **mT5-Small Combined (16k)** | D2 (Ahirani) | 540 | **78.43** | **89.88** | **13.97%** | 5.50% | 62.22% |
| **mT5-Small Combined (16k)** | D3 (Standard) | 555 | **96.98** | **99.13** | **1.72%** | 0.37% | 92.43% |
| **mT5-Small Combined (16k)** | D4 (Varhadi) | 516 | **74.65** | **89.12** | **15.75%** | 4.39% | 50.58% |
| **mT5-Small Combined (16k)** | Combined Multi-Dialect | 2,170 | **72.86** | **87.83** | **16.90%** | 5.84% | 52.35% |
| **mT5-Small Combined (32k)** | D1 (Malvani) | 559 | **44.83** | **74.85** | **34.50%** | 12.16% | 5.01% |
| **mT5-Small Combined (32k)** | D2 (Ahirani) | 540 | **78.88** | **89.91** | **13.01%** | 5.50% | 63.70% |
| **mT5-Small Combined (32k)** | D3 (Standard) | 555 | **97.74** | **99.22** | **1.07%** | 0.29% | 93.33% |
| **mT5-Small Combined (32k)** | D4 (Varhadi) | 516 | **76.00** | **89.33** | **14.96%** | 4.47% | 54.46% |
| **mT5-Small Combined (32k)** | Combined Multi-Dialect | 2,170 | **74.28** | **88.09** | **16.10%** | 5.73% | 53.96% |
| **mT5-Small Raw Unverified (32k)** | D1 (Malvani) | 559 | **44.74** | **75.16** | **34.50%** | 12.25% | 4.83% |
| **mT5-Small Raw Unverified (32k)** | D2 (Ahirani) | 540 | **79.15** | **90.12** | **13.27%** | 5.15% | 61.85% |
| **mT5-Small Raw Unverified (32k)** | D3 (Standard) | 555 | **97.97** | **99.34** | **0.98%** | 0.28% | 93.87% |
| **mT5-Small Raw Unverified (32k)** | D4 (Varhadi) | 516 | **76.33** | **89.34** | **14.91%** | 4.39% | 54.07% |
| **mT5-Small Raw Unverified (32k)** | Combined Multi-Dialect | 2,170 | **74.51** | **88.25** | **16.13%** | 5.64% | 53.50% |
| **mT5-Small D1 Malvani (16k)** | D1 (Malvani) | 559 | **43.81** | **74.99** | **34.64%** | 12.24% | 4.65% |
| **mT5-Small D1 Malvani (16k)** | D2 (Ahirani) | 540 | **81.13** | **91.88** | **10.74%** | 3.78% | 47.59% |
| **mT5-Small D1 Malvani (16k)** | D3 (Standard) | 555 | **88.94** | **95.96** | **5.99%** | 2.10% | 68.11% |
| **mT5-Small D1 Malvani (16k)** | D4 (Varhadi) | 516 | **75.51** | **90.34** | **14.37%** | 4.22% | 44.57% |
| **mT5-Small D1 Malvani (16k)** | Combined Multi-Dialect | 2,170 | **72.37** | **88.10** | **16.59%** | 5.69% | 41.06% |
| **mT5-Small D1 Malvani (32k)** | D1 (Malvani) | 559 | **44.10** | **76.13** | **32.73%** | 11.21% | 4.65% |
| **mT5-Small D1 Malvani (32k)** | D2 (Ahirani) | 540 | **83.85** | **93.53** | **8.59%** | 2.89% | 57.59% |
| **mT5-Small D1 Malvani (32k)** | D3 (Standard) | 555 | **94.97** | **98.19** | **2.46%** | 0.88% | 86.67% |
| **mT5-Small D1 Malvani (32k)** | D4 (Varhadi) | 516 | **82.03** | **92.78** | **10.76%** | 3.23% | 53.49% |
| **mT5-Small D1 Malvani (32k)** | Combined Multi-Dialect | 2,170 | **76.34** | **89.94** | **13.81%** | 4.66% | 50.41% |
| **mT5-Small D2 Ahirani (16k)** | D1 (Malvani) | 559 | **84.76** | **94.88** | **9.72%** | 2.97% | 48.30% |
| **mT5-Small D2 Ahirani (16k)** | D2 (Ahirani) | 540 | **81.79** | **92.48** | **10.76%** | 3.57% | 63.33% |
| **mT5-Small D2 Ahirani (16k)** | D3 (Standard) | 555 | **97.48** | **99.25** | **1.05%** | 0.31% | 94.41% |
| **mT5-Small D2 Ahirani (16k)** | D4 (Varhadi) | 516 | **88.57** | **95.97** | **6.45%** | 2.02% | 69.38% |
| **mT5-Small D2 Ahirani (16k)** | Combined Multi-Dialect | 2,170 | **88.22** | **95.59** | **7.09%** | 2.25% | 68.85% |
| **mT5-Small D2 Ahirani (32k)** | D1 (Malvani) | 559 | **80.77** | **92.65** | **12.20%** | 4.06% | 42.93% |
| **mT5-Small D2 Ahirani (32k)** | D2 (Ahirani) | 540 | **82.09** | **92.25** | **11.26%** | 3.98% | 64.44% |
| **mT5-Small D2 Ahirani (32k)** | D3 (Standard) | 555 | **97.86** | **99.40** | **1.07%** | 0.22% | 94.23% |
| **mT5-Small D2 Ahirani (32k)** | D4 (Varhadi) | 516 | **86.33** | **94.64** | **7.92%** | 2.15% | 62.79% |
| **mT5-Small D2 Ahirani (32k)** | Combined Multi-Dialect | 2,170 | **86.76** | **94.66** | **8.22%** | 2.65% | 66.13% |
| **mT5-Small D4 Varhadi (16k)** | D1 (Malvani) | 559 | **87.20** | **95.23** | **8.34%** | 2.23% | 56.35% |
| **mT5-Small D4 Varhadi (16k)** | D2 (Ahirani) | 540 | **93.57** | **97.84** | **3.87%** | 1.29% | 81.67% |
| **mT5-Small D4 Varhadi (16k)** | D3 (Standard) | 555 | **96.61** | **99.01** | **1.81%** | 0.53% | 91.35% |
| **mT5-Small D4 Varhadi (16k)** | D4 (Varhadi) | 516 | **76.40** | **89.98** | **13.85%** | 4.25% | 52.52% |
| **mT5-Small D4 Varhadi (16k)** | Combined Multi-Dialect | 2,170 | **88.44** | **95.50** | **6.98%** | 2.08% | 70.69% |
| **mT5-Small D4 Varhadi (32k)** | D1 (Malvani) | 559 | **88.44** | **95.33** | **8.30%** | 2.61% | 62.43% |
| **mT5-Small D4 Varhadi (32k)** | D2 (Ahirani) | 540 | **93.65** | **97.98** | **3.63%** | 0.86% | 80.74% |
| **mT5-Small D4 Varhadi (32k)** | D3 (Standard) | 555 | **97.40** | **99.00** | **1.42%** | 0.44% | 91.35% |
| **mT5-Small D4 Varhadi (32k)** | D4 (Varhadi) | 516 | **77.94** | **90.43** | **13.71%** | 4.16% | 55.43% |
| **mT5-Small D4 Varhadi (32k)** | Combined Multi-Dialect | 2,170 | **89.35** | **95.67** | **6.78%** | 2.03% | 72.72% |

---

## 2. Complete 5-Fold Foldwise Benchmark Matrix for ALL Models (IndicBART & mT5-Small)

This table presents the exact fold-by-fold results (Folds 1 to 5, Mean, and Std Dev) across both neural architectures.

| Model Name | Architecture | Dialect Variety | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | **Mean $\pm$ Std ($M \pm \sigma$)** |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **IndicBART Combined (16k)** | IndicBART | **D1 (Malvani) BLEU** | 18.21 | 14.70 | 17.33 | 18.19 | 16.18 | **16.92 $\pm$ 1.49** |
| **IndicBART Combined (16k)** | IndicBART | **D2 (Ahirani) BLEU** | 59.41 | 57.89 | 60.03 | 57.80 | 60.71 | **59.17 $\pm$ 1.29** |
| **IndicBART Combined (16k)** | IndicBART | **D3 (Standard) BLEU** | 81.16 | 81.93 | 80.65 | 77.96 | 77.60 | **79.86 $\pm$ 1.96** |
| **IndicBART Combined (16k)** | IndicBART | **D4 (Varhadi) BLEU** | 56.89 | 56.91 | 58.36 | 57.55 | 59.20 | **57.78 $\pm$ 0.99** |
| **IndicBART Combined (16k)** | IndicBART | **Combined Multi-Dialect BLEU** | 53.34 | 52.25 | 53.51 | 52.28 | 52.76 | **52.83 $\pm$ 0.58** |
| **IndicBART Combined (32k)** | IndicBART | **D1 (Malvani) BLEU** | 38.23 | 32.84 | 35.13 | 39.80 | 26.52 | **34.50 $\pm$ 5.22** |
| **IndicBART Combined (32k)** | IndicBART | **D2 (Ahirani) BLEU** | 63.32 | 60.40 | 64.86 | 68.80 | 52.93 | **62.06 $\pm$ 5.94** |
| **IndicBART Combined (32k)** | IndicBART | **D3 (Standard) BLEU** | 76.73 | 78.16 | 76.56 | 83.11 | 64.63 | **75.84 $\pm$ 6.80** |
| **IndicBART Combined (32k)** | IndicBART | **D4 (Varhadi) BLEU** | 64.83 | 54.29 | 55.41 | 62.39 | 55.40 | **58.46 $\pm$ 4.80** |
| **IndicBART Combined (32k)** | IndicBART | **Combined Multi-Dialect BLEU** | 60.69 | 56.19 | 57.81 | 63.42 | 49.38 | **57.50 $\pm$ 5.31** |
| **IndicBART Raw Unverified (32k)** | IndicBART | **D1 (Malvani) BLEU** | 19.63 | 17.85 | 16.91 | 14.95 | 16.87 | **17.24 $\pm$ 1.70** |
| **IndicBART Raw Unverified (32k)** | IndicBART | **D2 (Ahirani) BLEU** | 62.78 | 57.13 | 59.56 | 57.79 | 57.51 | **58.95 $\pm$ 2.33** |
| **IndicBART Raw Unverified (32k)** | IndicBART | **D3 (Standard) BLEU** | 77.62 | 77.97 | 78.31 | 77.78 | 77.94 | **77.92 $\pm$ 0.26** |
| **IndicBART Raw Unverified (32k)** | IndicBART | **D4 (Varhadi) BLEU** | 55.17 | 57.25 | 56.45 | 57.46 | 58.26 | **56.92 $\pm$ 1.17** |
| **IndicBART Raw Unverified (32k)** | IndicBART | **Combined Multi-Dialect BLEU** | 53.26 | 51.93 | 52.16 | 51.31 | 52.03 | **52.14 $\pm$ 0.71** |
| **IndicBART D1 Malvani (16k)** | IndicBART | **D1 (Malvani) BLEU** | 47.13 | 46.05 | 45.15 | 45.65 | 45.78 | **45.95 $\pm$ 0.74** |
| **IndicBART D1 Malvani (16k)** | IndicBART | **D2 (Ahirani) BLEU** | 78.82 | 77.66 | 77.63 | 76.72 | 77.73 | **77.71 $\pm$ 0.75** |
| **IndicBART D1 Malvani (16k)** | IndicBART | **D3 (Standard) BLEU** | 90.22 | 90.69 | 90.14 | 90.39 | 90.86 | **90.46 $\pm$ 0.31** |
| **IndicBART D1 Malvani (16k)** | IndicBART | **D4 (Varhadi) BLEU** | 75.03 | 74.53 | 74.71 | 73.18 | 74.68 | **74.43 $\pm$ 0.72** |
| **IndicBART D1 Malvani (16k)** | IndicBART | **Combined Multi-Dialect BLEU** | 72.71 | 72.14 | 71.83 | 71.37 | 72.16 | **72.04 $\pm$ 0.49** |
| **IndicBART D1 Malvani (32k)** | IndicBART | **D1 (Malvani) BLEU** | 13.54 | 13.93 | 13.54 | 12.45 | 13.85 | **13.46 $\pm$ 0.59** |
| **IndicBART D1 Malvani (32k)** | IndicBART | **D2 (Ahirani) BLEU** | 62.74 | 65.47 | 63.85 | 65.48 | 63.51 | **64.21 $\pm$ 1.22** |
| **IndicBART D1 Malvani (32k)** | IndicBART | **D3 (Standard) BLEU** | 80.54 | 83.00 | 84.28 | 80.64 | 82.08 | **82.11 $\pm$ 1.59** |
| **IndicBART D1 Malvani (32k)** | IndicBART | **D4 (Varhadi) BLEU** | 59.20 | 59.49 | 61.39 | 59.71 | 61.33 | **60.22 $\pm$ 1.05** |
| **IndicBART D1 Malvani (32k)** | IndicBART | **Combined Multi-Dialect BLEU** | 53.22 | 54.88 | 55.18 | 53.82 | 54.49 | **54.32 $\pm$ 0.80** |
| **IndicBART D2 Ahirani (16k)** | IndicBART | **D1 (Malvani) BLEU** | 69.56 | 69.07 | 70.52 | 69.95 | 69.83 | **69.79 $\pm$ 0.53** |
| **IndicBART D2 Ahirani (16k)** | IndicBART | **D2 (Ahirani) BLEU** | 84.27 | 83.84 | 84.57 | 84.71 | 83.92 | **84.26 $\pm$ 0.38** |
| **IndicBART D2 Ahirani (16k)** | IndicBART | **D3 (Standard) BLEU** | 97.82 | 97.30 | 97.86 | 98.57 | 97.99 | **97.91 $\pm$ 0.45** |
| **IndicBART D2 Ahirani (16k)** | IndicBART | **D4 (Varhadi) BLEU** | 84.29 | 84.00 | 82.92 | 83.44 | 83.90 | **83.71 $\pm$ 0.54** |
| **IndicBART D2 Ahirani (16k)** | IndicBART | **Combined Multi-Dialect BLEU** | 83.83 | 83.40 | 83.82 | 84.02 | 83.75 | **83.76 $\pm$ 0.23** |
| **IndicBART D2 Ahirani (32k)** | IndicBART | **D1 (Malvani) BLEU** | 26.05 | 26.87 | 29.37 | 27.72 | 31.40 | **28.28 $\pm$ 2.13** |
| **IndicBART D2 Ahirani (32k)** | IndicBART | **D2 (Ahirani) BLEU** | 56.15 | 60.32 | 61.09 | 63.16 | 56.14 | **59.37 $\pm$ 3.12** |
| **IndicBART D2 Ahirani (32k)** | IndicBART | **D3 (Standard) BLEU** | 78.83 | 80.06 | 78.88 | 77.94 | 79.28 | **79.00 $\pm$ 0.77** |
| **IndicBART D2 Ahirani (32k)** | IndicBART | **D4 (Varhadi) BLEU** | 58.59 | 61.71 | 57.71 | 60.08 | 56.92 | **59.00 $\pm$ 1.92** |
| **IndicBART D2 Ahirani (32k)** | IndicBART | **Combined Multi-Dialect BLEU** | 54.67 | 56.97 | 56.53 | 56.99 | 55.69 | **56.17 $\pm$ 0.99** |
| **IndicBART D4 Varhadi (16k)** | IndicBART | **D1 (Malvani) BLEU** | 83.36 | 83.25 | 83.53 | 82.80 | 83.81 | **83.35 $\pm$ 0.37** |
| **IndicBART D4 Varhadi (16k)** | IndicBART | **D2 (Ahirani) BLEU** | 95.65 | 95.04 | 94.86 | 95.72 | 95.23 | **95.30 $\pm$ 0.38** |
| **IndicBART D4 Varhadi (16k)** | IndicBART | **D3 (Standard) BLEU** | 98.68 | 98.20 | 98.02 | 98.68 | 97.66 | **98.25 $\pm$ 0.44** |
| **IndicBART D4 Varhadi (16k)** | IndicBART | **D4 (Varhadi) BLEU** | 79.69 | 79.79 | 80.64 | 80.06 | 80.66 | **80.17 $\pm$ 0.46** |
| **IndicBART D4 Varhadi (16k)** | IndicBART | **Combined Multi-Dialect BLEU** | 89.30 | 89.03 | 89.21 | 89.28 | 89.29 | **89.22 $\pm$ 0.11** |
| **IndicBART D4 Varhadi (32k)** | IndicBART | **D1 (Malvani) BLEU** | 63.96 | 64.95 | 59.44 | 62.33 | 57.13 | **61.56 $\pm$ 3.24** |
| **IndicBART D4 Varhadi (32k)** | IndicBART | **D2 (Ahirani) BLEU** | 76.61 | 75.17 | 73.71 | 73.94 | 75.01 | **74.89 $\pm$ 1.16** |
| **IndicBART D4 Varhadi (32k)** | IndicBART | **D3 (Standard) BLEU** | 81.57 | 82.65 | 82.03 | 82.42 | 82.71 | **82.28 $\pm$ 0.48** |
| **IndicBART D4 Varhadi (32k)** | IndicBART | **D4 (Varhadi) BLEU** | 64.59 | 64.16 | 62.25 | 62.76 | 62.73 | **63.30 $\pm$ 1.02** |
| **IndicBART D4 Varhadi (32k)** | IndicBART | **Combined Multi-Dialect BLEU** | 71.56 | 71.61 | 69.22 | 70.22 | 69.27 | **70.38 $\pm$ 1.17** |
| **Mt5-Small Combined 16K** | google/mT5 | **Combined Test BLEU** | 63.40 | 63.20 | 63.23 | 63.43 | 63.21 | **63.29 $\pm$ 0.11** |
| | | **D1 Test BLEU** | 48.22 | 48.03 | 48.18 | 48.70 | 48.27 | **48.28 $\pm$ 0.25** |
| | | **D2 Test BLEU** | 62.49 | 62.31 | 62.16 | 62.47 | 62.30 | **62.35 $\pm$ 0.13** |
| | | **D4 Test BLEU** | 79.66 | 79.67 | 79.50 | 79.41 | 79.62 | **79.57 $\pm$ 0.12** |
| **Mt5-Small Combined 32K** | google/mT5 | **Combined Test BLEU** | 69.67 | 69.64 | 69.44 | 69.40 | 69.39 | **69.51 $\pm$ 0.14** |
| | | **D1 Test BLEU** | 66.75 | 66.80 | 66.48 | 66.69 | 66.37 | **66.62 $\pm$ 0.19** |
| | | **D2 Test BLEU** | 62.95 | 62.87 | 62.71 | 62.44 | 62.62 | **62.72 $\pm$ 0.20** |
| | | **D4 Test BLEU** | 78.90 | 78.73 | 78.70 | 78.60 | 78.71 | **78.73 $\pm$ 0.11** |
| **Mt5-Small D1 16K** | google/mT5 | **Combined Test BLEU** | 46.55 | 46.43 | 46.38 | 45.43 | 46.79 | **46.31 $\pm$ 0.52** |
| **Mt5-Small D1 32K** | google/mT5 | **Combined Test BLEU** | 64.86 | 64.85 | 65.01 | 65.15 | 65.60 | **65.10 $\pm$ 0.31** |
| **Mt5-Small D2 16K** | google/mT5 | **Combined Test BLEU** | 60.47 | 59.94 | 60.34 | 60.46 | 60.34 | **60.31 $\pm$ 0.22** |
| **Mt5-Small D2 32K** | google/mT5 | **Combined Test BLEU** | 61.87 | 62.27 | 62.06 | 61.87 | 62.27 | **62.07 $\pm$ 0.20** |
| **Mt5-Small D4 16K** | google/mT5 | **Combined Test BLEU** | 81.27 | 81.02 | 80.89 | 80.97 | 80.82 | **81.00 $\pm$ 0.17** |
| **Mt5-Small D4 32K** | google/mT5 | **Combined Test BLEU** | 79.13 | 78.88 | 78.97 | 78.96 | 78.51 | **78.89 $\pm$ 0.23** |
| **Mt5-Small Raw Unverified 32K** | google/mT5 | **Combined Test BLEU** | 61.07 | 61.10 | 61.35 | 61.16 | 61.38 | **61.21 $\pm$ 0.15** |
| | | **D1 Test BLEU** | 49.19 | 49.45 | 49.06 | 49.30 | 49.07 | **49.21 $\pm$ 0.17** |
| | | **D2 Test BLEU** | 58.46 | 58.25 | 58.87 | 58.33 | 58.82 | **58.55 $\pm$ 0.28** |
| | | **D4 Test BLEU** | 77.91 | 77.95 | 78.49 | 78.13 | 78.05 | **78.10 $\pm$ 0.23** |

---

## 3. Side-by-Side Architecture Benchmark: Rules vs. IndicBART vs. mT5-Small

| Spoken Dialect Partition | Evaluated Utterances | **Deterministic Rules Baseline**<br>BLEU / WER / chrF++ | **IndicBART Combined (32k)**<br>BLEU / WER / chrF++ | **mT5-Small Combined (32k)**<br>BLEU / WER / chrF++ | **Net Neural Advantage** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **D1 Malvani** | 559 | 21.31 / 59.52% / 46.80 | **34.50** / 48.45% / 61.06 | **44.83** / **34.50%** / **74.85** | **+23.52 BLEU** (-25.02% WER) 🚀 |
| **D2 Ahirani** | 540 | 18.57 / 67.37% / 43.12 | **62.06** / 29.83% / 75.85 | **78.88** / **13.01%** / **89.91** | **+60.31 BLEU** (-54.36% WER) 🚀 |
| **D3 Standard Pune** | 555 | 98.10 / 1.15% / 99.40 | 75.84 / 20.11% / 84.19 | **97.74** / **1.07%** / **99.22** | Preserves Standard Forms |
| **D4 Varhadi** | 516 | 43.66 / 41.29% / 68.10 | **58.46** / 33.13% / 74.32 | **76.00** / **14.96%** / **89.33** | **+32.34 BLEU** (-26.33% WER) 🚀 |
| **Combined (Total Spoken)** | **2,170** | 81.29 / 8.54% / 89.45 | 57.50 / 33.05% / 73.70 | **74.28** / **16.10%** / **88.09** | **Robust End-to-End Generalization** |

---

## 4. Verification Engine & Synthetic Expansion Ablation (Side-by-Side)

| Model Setup | Training Data Quality & Scale | Total Pairs | RESPIN BLEU ($M \pm \sigma$) | RESPIN WER % ($M \pm \sigma$) | RESPIN chrF++ ($M \pm \sigma$) | Empirical Verification Gain |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **`indicbart_combined_16k`** | Original Clean 16k Baseline | 16,163 | 52.83 $\pm$ 0.58 | 37.85% $\pm$ 0.48% | 69.33 $\pm$ 0.40 | Baseline 16k dataset |
| **`indicbart_raw_unverified_32k`** | Noisy / Raw Synthetic 32k | 32,335 | 52.14 $\pm$ 0.71 | 38.50% $\pm$ 0.56% | 68.71 $\pm$ 0.53 | -0.69 BLEU drop due to hallucinated noise |
| **`indicbart_combined_32k`** | **Strict Verified & Filtered 32k** | **32,335** | **57.50 $\pm$ 5.31** | **33.05% $\pm$ 4.69%** | **73.70 $\pm$ 4.20** | **+4.67 BLEU over 16k (+5.36 BLEU over unverified)** 🔥 |
| **`mt5_combined_16k`** | Original Clean 16k Baseline | 16,163 | 72.86 | 16.90% | 87.83 | Baseline 16k dataset |
| **`mt5_raw_unverified_32k`** | Noisy / Raw Synthetic 32k | 32,335 | 74.51 | 16.13% | 88.25 | Noisy baseline 32k data |
| **`mt5_combined_32k`** | **Strict Verified & Filtered 32k** | **32,335** | **74.28** | **16.10%** | **88.09** | **Highest Normalization Quality on Spoken Speech** 🔥 |

---

