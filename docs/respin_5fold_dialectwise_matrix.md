# Master Empirical Benchmark: Dialectwise & Foldwise Evaluation on IISc RESPIN Test Set

**Official Held-Out Spoken Test Set**: `IISc_RESPIN_test_mr` (2,170 Utterances: D1=559, D2=540, D3=555, D4=516)

---

## 1. Dialectwise & Foldwise BLEU Score Matrix

| Model Name | Dialect Variety | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | **Mean BLEU ($\pm \sigma$)** |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **IndicBART Combined (16k)** | **D1 (Malvani)** | 18.21 | 14.70 | 17.33 | 18.19 | 16.18 | **16.92 $\pm$ 1.49** |
| **IndicBART Combined (16k)** | **D2 (Ahirani)** | 59.41 | 57.89 | 60.03 | 57.80 | 60.71 | **59.17 $\pm$ 1.29** |
| **IndicBART Combined (16k)** | **D3 (Standard)** | 81.16 | 81.93 | 80.65 | 77.96 | 77.60 | **79.86 $\pm$ 1.96** |
| **IndicBART Combined (16k)** | **D4 (Varhadi)** | 56.89 | 56.91 | 58.36 | 57.55 | 59.20 | **57.78 $\pm$ 0.99** |
| **IndicBART Combined (16k)** | **Combined** | 53.34 | 52.25 | 53.51 | 52.28 | 52.76 | **52.83 $\pm$ 0.58** |
| **IndicBART Combined (32k)** | **D1 (Malvani)** | 38.23 | 32.84 | 35.13 | 39.80 | 26.52 | **34.50 $\pm$ 5.22** |
| **IndicBART Combined (32k)** | **D2 (Ahirani)** | 63.32 | 60.40 | 64.86 | 68.80 | 52.93 | **62.06 $\pm$ 5.94** |
| **IndicBART Combined (32k)** | **D3 (Standard)** | 76.73 | 78.16 | 76.56 | 83.11 | 64.63 | **75.84 $\pm$ 6.80** |
| **IndicBART Combined (32k)** | **D4 (Varhadi)** | 64.83 | 54.29 | 55.41 | 62.39 | 55.40 | **58.46 $\pm$ 4.80** |
| **IndicBART Combined (32k)** | **Combined** | 60.69 | 56.19 | 57.81 | 63.42 | 49.38 | **57.50 $\pm$ 5.31** |
| **IndicBART Raw Unverified (32k)** | **D1 (Malvani)** | 19.63 | 17.85 | 16.91 | 14.95 | 16.87 | **17.24 $\pm$ 1.70** |
| **IndicBART Raw Unverified (32k)** | **D2 (Ahirani)** | 62.78 | 57.13 | 59.56 | 57.79 | 57.51 | **58.95 $\pm$ 2.33** |
| **IndicBART Raw Unverified (32k)** | **D3 (Standard)** | 77.62 | 77.97 | 78.31 | 77.78 | 77.94 | **77.92 $\pm$ 0.26** |
| **IndicBART Raw Unverified (32k)** | **D4 (Varhadi)** | 55.17 | 57.25 | 56.45 | 57.46 | 58.26 | **56.92 $\pm$ 1.17** |
| **IndicBART Raw Unverified (32k)** | **Combined** | 53.26 | 51.93 | 52.16 | 51.31 | 52.03 | **52.14 $\pm$ 0.71** |
| **IndicBART D1 Malvani (16k)** | **D1 (Malvani)** | 47.13 | 46.05 | 45.15 | 45.65 | 45.78 | **45.95 $\pm$ 0.74** |
| **IndicBART D1 Malvani (16k)** | **D2 (Ahirani)** | 78.82 | 77.66 | 77.63 | 76.72 | 77.73 | **77.71 $\pm$ 0.75** |
| **IndicBART D1 Malvani (16k)** | **D3 (Standard)** | 90.22 | 90.69 | 90.14 | 90.39 | 90.86 | **90.46 $\pm$ 0.31** |
| **IndicBART D1 Malvani (16k)** | **D4 (Varhadi)** | 75.03 | 74.53 | 74.71 | 73.18 | 74.68 | **74.43 $\pm$ 0.72** |
| **IndicBART D1 Malvani (16k)** | **Combined** | 72.71 | 72.14 | 71.83 | 71.37 | 72.16 | **72.04 $\pm$ 0.49** |
| **IndicBART D1 Malvani (32k)** | **D1 (Malvani)** | 13.54 | 13.93 | 13.54 | 12.45 | 13.85 | **13.46 $\pm$ 0.59** |
| **IndicBART D1 Malvani (32k)** | **D2 (Ahirani)** | 62.74 | 65.47 | 63.85 | 65.48 | 63.51 | **64.21 $\pm$ 1.22** |
| **IndicBART D1 Malvani (32k)** | **D3 (Standard)** | 80.54 | 83.00 | 84.28 | 80.64 | 82.08 | **82.11 $\pm$ 1.59** |
| **IndicBART D1 Malvani (32k)** | **D4 (Varhadi)** | 59.20 | 59.49 | 61.39 | 59.71 | 61.33 | **60.22 $\pm$ 1.05** |
| **IndicBART D1 Malvani (32k)** | **Combined** | 53.22 | 54.88 | 55.18 | 53.82 | 54.49 | **54.32 $\pm$ 0.80** |
| **IndicBART D2 Ahirani (16k)** | **D1 (Malvani)** | 69.56 | 69.07 | 70.52 | 69.95 | 69.83 | **69.79 $\pm$ 0.53** |
| **IndicBART D2 Ahirani (16k)** | **D2 (Ahirani)** | 84.27 | 83.84 | 84.57 | 84.71 | 83.92 | **84.26 $\pm$ 0.38** |
| **IndicBART D2 Ahirani (16k)** | **D3 (Standard)** | 97.82 | 97.30 | 97.86 | 98.57 | 97.99 | **97.91 $\pm$ 0.45** |
| **IndicBART D2 Ahirani (16k)** | **D4 (Varhadi)** | 84.29 | 84.00 | 82.92 | 83.44 | 83.90 | **83.71 $\pm$ 0.54** |
| **IndicBART D2 Ahirani (16k)** | **Combined** | 83.83 | 83.40 | 83.82 | 84.02 | 83.75 | **83.76 $\pm$ 0.23** |
| **IndicBART D2 Ahirani (32k)** | **D1 (Malvani)** | 26.05 | 26.87 | 29.37 | 27.72 | 31.40 | **28.28 $\pm$ 2.13** |
| **IndicBART D2 Ahirani (32k)** | **D2 (Ahirani)** | 56.15 | 60.32 | 61.09 | 63.16 | 56.14 | **59.37 $\pm$ 3.12** |
| **IndicBART D2 Ahirani (32k)** | **D3 (Standard)** | 78.83 | 80.06 | 78.88 | 77.94 | 79.28 | **79.00 $\pm$ 0.77** |
| **IndicBART D2 Ahirani (32k)** | **D4 (Varhadi)** | 58.59 | 61.71 | 57.71 | 60.08 | 56.92 | **59.00 $\pm$ 1.92** |
| **IndicBART D2 Ahirani (32k)** | **Combined** | 54.67 | 56.97 | 56.53 | 56.99 | 55.69 | **56.17 $\pm$ 0.99** |
| **IndicBART D4 Varhadi (16k)** | **D1 (Malvani)** | 83.36 | 83.25 | 83.53 | 82.80 | 83.81 | **83.35 $\pm$ 0.37** |
| **IndicBART D4 Varhadi (16k)** | **D2 (Ahirani)** | 95.65 | 95.04 | 94.86 | 95.72 | 95.23 | **95.30 $\pm$ 0.38** |
| **IndicBART D4 Varhadi (16k)** | **D3 (Standard)** | 98.68 | 98.20 | 98.02 | 98.68 | 97.66 | **98.25 $\pm$ 0.44** |
| **IndicBART D4 Varhadi (16k)** | **D4 (Varhadi)** | 79.69 | 79.79 | 80.64 | 80.06 | 80.66 | **80.17 $\pm$ 0.46** |
| **IndicBART D4 Varhadi (16k)** | **Combined** | 89.30 | 89.03 | 89.21 | 89.28 | 89.29 | **89.22 $\pm$ 0.11** |
| **IndicBART D4 Varhadi (32k)** | **D1 (Malvani)** | 63.96 | 64.95 | 59.44 | 62.33 | 57.13 | **61.56 $\pm$ 3.24** |
| **IndicBART D4 Varhadi (32k)** | **D2 (Ahirani)** | 76.61 | 75.17 | 73.71 | 73.94 | 75.01 | **74.89 $\pm$ 1.16** |
| **IndicBART D4 Varhadi (32k)** | **D3 (Standard)** | 81.57 | 82.65 | 82.03 | 82.42 | 82.71 | **82.28 $\pm$ 0.48** |
| **IndicBART D4 Varhadi (32k)** | **D4 (Varhadi)** | 64.59 | 64.16 | 62.25 | 62.76 | 62.73 | **63.30 $\pm$ 1.02** |
| **IndicBART D4 Varhadi (32k)** | **Combined** | 71.56 | 71.61 | 69.22 | 70.22 | 69.27 | **70.38 $\pm$ 1.17** |
| **mT5-Small Combined (16k)** | **D1 (Malvani)** | 41.22 | - | - | - | - | **41.22** |
| **mT5-Small Combined (16k)** | **D2 (Ahirani)** | 78.43 | - | - | - | - | **78.43** |
| **mT5-Small Combined (16k)** | **D3 (Standard)** | 96.98 | - | - | - | - | **96.98** |
| **mT5-Small Combined (16k)** | **D4 (Varhadi)** | 74.65 | - | - | - | - | **74.65** |
| **mT5-Small Combined (16k)** | **Combined** | 72.86 | - | - | - | - | **72.86** |
| **mT5-Small Combined (32k)** | **D1 (Malvani)** | 44.83 | - | - | - | - | **44.83** |
| **mT5-Small Combined (32k)** | **D2 (Ahirani)** | 78.88 | - | - | - | - | **78.88** |
| **mT5-Small Combined (32k)** | **D3 (Standard)** | 97.74 | - | - | - | - | **97.74** |
| **mT5-Small Combined (32k)** | **D4 (Varhadi)** | 76.00 | - | - | - | - | **76.00** |
| **mT5-Small Combined (32k)** | **Combined** | 74.28 | - | - | - | - | **74.28** |
| **mT5-Small Raw Unverified (32k)** | **D1 (Malvani)** | 44.74 | - | - | - | - | **44.74** |
| **mT5-Small Raw Unverified (32k)** | **D2 (Ahirani)** | 79.15 | - | - | - | - | **79.15** |
| **mT5-Small Raw Unverified (32k)** | **D3 (Standard)** | 97.97 | - | - | - | - | **97.97** |
| **mT5-Small Raw Unverified (32k)** | **D4 (Varhadi)** | 76.33 | - | - | - | - | **76.33** |
| **mT5-Small Raw Unverified (32k)** | **Combined** | 74.51 | - | - | - | - | **74.51** |
| **mT5-Small D1 Malvani (16k)** | **D1 (Malvani)** | 43.81 | - | - | - | - | **43.81** |
| **mT5-Small D1 Malvani (16k)** | **D2 (Ahirani)** | 81.13 | - | - | - | - | **81.13** |
| **mT5-Small D1 Malvani (16k)** | **D3 (Standard)** | 88.94 | - | - | - | - | **88.94** |
| **mT5-Small D1 Malvani (16k)** | **D4 (Varhadi)** | 75.51 | - | - | - | - | **75.51** |
| **mT5-Small D1 Malvani (16k)** | **Combined** | 72.37 | - | - | - | - | **72.37** |
| **mT5-Small D1 Malvani (32k)** | **D1 (Malvani)** | 44.10 | - | - | - | - | **44.10** |
| **mT5-Small D1 Malvani (32k)** | **D2 (Ahirani)** | 83.85 | - | - | - | - | **83.85** |
| **mT5-Small D1 Malvani (32k)** | **D3 (Standard)** | 94.97 | - | - | - | - | **94.97** |
| **mT5-Small D1 Malvani (32k)** | **D4 (Varhadi)** | 82.03 | - | - | - | - | **82.03** |
| **mT5-Small D1 Malvani (32k)** | **Combined** | 76.34 | - | - | - | - | **76.34** |
| **mT5-Small D2 Ahirani (16k)** | **D1 (Malvani)** | 84.76 | - | - | - | - | **84.76** |
| **mT5-Small D2 Ahirani (16k)** | **D2 (Ahirani)** | 81.79 | - | - | - | - | **81.79** |
| **mT5-Small D2 Ahirani (16k)** | **D3 (Standard)** | 97.48 | - | - | - | - | **97.48** |
| **mT5-Small D2 Ahirani (16k)** | **D4 (Varhadi)** | 88.57 | - | - | - | - | **88.57** |
| **mT5-Small D2 Ahirani (16k)** | **Combined** | 88.22 | - | - | - | - | **88.22** |
| **mT5-Small D2 Ahirani (32k)** | **D1 (Malvani)** | 80.77 | - | - | - | - | **80.77** |
| **mT5-Small D2 Ahirani (32k)** | **D2 (Ahirani)** | 82.09 | - | - | - | - | **82.09** |
| **mT5-Small D2 Ahirani (32k)** | **D3 (Standard)** | 97.86 | - | - | - | - | **97.86** |
| **mT5-Small D2 Ahirani (32k)** | **D4 (Varhadi)** | 86.33 | - | - | - | - | **86.33** |
| **mT5-Small D2 Ahirani (32k)** | **Combined** | 86.76 | - | - | - | - | **86.76** |
| **mT5-Small D4 Varhadi (16k)** | **D1 (Malvani)** | 87.20 | - | - | - | - | **87.20** |
| **mT5-Small D4 Varhadi (16k)** | **D2 (Ahirani)** | 93.57 | - | - | - | - | **93.57** |
| **mT5-Small D4 Varhadi (16k)** | **D3 (Standard)** | 96.61 | - | - | - | - | **96.61** |
| **mT5-Small D4 Varhadi (16k)** | **D4 (Varhadi)** | 76.40 | - | - | - | - | **76.40** |
| **mT5-Small D4 Varhadi (16k)** | **Combined** | 88.44 | - | - | - | - | **88.44** |
| **mT5-Small D4 Varhadi (32k)** | **D1 (Malvani)** | 88.44 | - | - | - | - | **88.44** |
| **mT5-Small D4 Varhadi (32k)** | **D2 (Ahirani)** | 93.65 | - | - | - | - | **93.65** |
| **mT5-Small D4 Varhadi (32k)** | **D3 (Standard)** | 97.40 | - | - | - | - | **97.40** |
| **mT5-Small D4 Varhadi (32k)** | **D4 (Varhadi)** | 77.94 | - | - | - | - | **77.94** |
| **mT5-Small D4 Varhadi (32k)** | **Combined** | 89.35 | - | - | - | - | **89.35** |

---

## 2. Dialectwise & Foldwise Word Error Rate (WER %) Matrix

| Model Name | Dialect Variety | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | **Mean WER ($\pm \sigma$)** |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **IndicBART Combined (16k)** | **D1 (Malvani)** | 65.19% | 68.42% | 66.13% | 65.43% | 66.94% | **66.42% $\pm$ 1.31%** |
| **IndicBART Combined (16k)** | **D2 (Ahirani)** | 32.65% | 33.87% | 32.45% | 34.03% | 31.90% | **32.98% $\pm$ 0.93%** |
| **IndicBART Combined (16k)** | **D3 (Standard)** | 15.68% | 15.03% | 16.08% | 18.38% | 18.82% | **16.80% $\pm$ 1.69%** |
| **IndicBART Combined (16k)** | **D4 (Varhadi)** | 34.71% | 34.73% | 33.48% | 34.05% | 32.69% | **33.93% $\pm$ 0.87%** |
| **IndicBART Combined (16k)** | **Combined** | 37.38% | 38.36% | 37.36% | 38.28% | 37.89% | **37.85% $\pm$ 0.48%** |
| **IndicBART Combined (32k)** | **D1 (Malvani)** | 44.34% | 50.37% | 47.96% | 42.71% | 56.87% | **48.45% $\pm$ 5.58%** |
| **IndicBART Combined (32k)** | **D2 (Ahirani)** | 29.08% | 31.60% | 27.55% | 23.64% | 37.26% | **29.83% $\pm$ 5.06%** |
| **IndicBART Combined (32k)** | **D3 (Standard)** | 19.82% | 17.94% | 19.33% | 14.06% | 29.41% | **20.11% $\pm$ 5.67%** |
| **IndicBART Combined (32k)** | **D4 (Varhadi)** | 27.90% | 36.73% | 35.60% | 29.33% | 36.10% | **33.13% $\pm$ 4.17%** |
| **IndicBART Combined (32k)** | **Combined** | 30.45% | 34.36% | 32.77% | 27.60% | 40.09% | **33.05% $\pm$ 4.69%** |
| **IndicBART Raw Unverified (32k)** | **D1 (Malvani)** | 64.35% | 65.78% | 66.55% | 68.26% | 66.59% | **66.31% $\pm$ 1.42%** |
| **IndicBART Raw Unverified (32k)** | **D2 (Ahirani)** | 30.16% | 34.53% | 32.65% | 34.18% | 34.00% | **33.10% $\pm$ 1.79%** |
| **IndicBART Raw Unverified (32k)** | **D3 (Standard)** | 18.84% | 18.56% | 18.08% | 18.82% | 18.61% | **18.58% $\pm$ 0.31%** |
| **IndicBART Raw Unverified (32k)** | **D4 (Varhadi)** | 36.14% | 34.51% | 35.62% | 34.32% | 33.55% | **34.83% $\pm$ 1.04%** |
| **IndicBART Raw Unverified (32k)** | **Combined** | 37.64% | 38.65% | 38.52% | 39.21% | 38.50% | **38.50% $\pm$ 0.56%** |
| **IndicBART D1 Malvani (16k)** | **D1 (Malvani)** | 34.42% | 35.16% | 35.38% | 35.69% | 36.06% | **35.34% $\pm$ 0.62%** |
| **IndicBART D1 Malvani (16k)** | **D2 (Ahirani)** | 13.62% | 14.69% | 14.91% | 15.33% | 14.61% | **14.63% $\pm$ 0.63%** |
| **IndicBART D1 Malvani (16k)** | **D3 (Standard)** | 6.18% | 5.95% | 6.34% | 6.13% | 5.97% | **6.11% $\pm$ 0.16%** |
| **IndicBART D1 Malvani (16k)** | **D4 (Varhadi)** | 17.19% | 17.57% | 17.93% | 18.55% | 17.21% | **17.69% $\pm$ 0.57%** |
| **IndicBART D1 Malvani (16k)** | **Combined** | 18.02% | 18.52% | 18.82% | 19.11% | 18.65% | **18.62% $\pm$ 0.40%** |
| **IndicBART D1 Malvani (32k)** | **D1 (Malvani)** | 68.44% | 68.44% | 68.48% | 69.53% | 68.63% | **68.70% $\pm$ 0.47%** |
| **IndicBART D1 Malvani (32k)** | **D2 (Ahirani)** | 30.40% | 28.27% | 29.54% | 28.25% | 29.61% | **29.21% $\pm$ 0.93%** |
| **IndicBART D1 Malvani (32k)** | **D3 (Standard)** | 17.15% | 14.92% | 13.59% | 16.73% | 15.57% | **15.59% $\pm$ 1.43%** |
| **IndicBART D1 Malvani (32k)** | **D4 (Varhadi)** | 33.55% | 33.14% | 31.71% | 33.23% | 31.83% | **32.69% $\pm$ 0.86%** |
| **IndicBART D1 Malvani (32k)** | **Combined** | 37.69% | 36.51% | 36.18% | 37.25% | 36.74% | **36.87% $\pm$ 0.60%** |
| **IndicBART D2 Ahirani (16k)** | **D1 (Malvani)** | 18.24% | 18.28% | 17.65% | 17.95% | 18.22% | **18.07% $\pm$ 0.27%** |
| **IndicBART D2 Ahirani (16k)** | **D2 (Ahirani)** | 10.41% | 10.45% | 10.28% | 10.19% | 10.82% | **10.43% $\pm$ 0.24%** |
| **IndicBART D2 Ahirani (16k)** | **D3 (Standard)** | 1.00% | 1.18% | 0.81% | 0.77% | 1.02% | **0.96% $\pm$ 0.17%** |
| **IndicBART D2 Ahirani (16k)** | **D4 (Varhadi)** | 9.83% | 9.56% | 10.17% | 10.15% | 9.83% | **9.91% $\pm$ 0.26%** |
| **IndicBART D2 Ahirani (16k)** | **Combined** | 10.00% | 10.00% | 9.85% | 9.89% | 10.10% | **9.97% $\pm$ 0.10%** |
| **IndicBART D2 Ahirani (32k)** | **D1 (Malvani)** | 59.13% | 58.65% | 56.76% | 57.97% | 54.96% | **57.49% $\pm$ 1.67%** |
| **IndicBART D2 Ahirani (32k)** | **D2 (Ahirani)** | 35.47% | 32.36% | 31.69% | 30.00% | 35.47% | **33.00% $\pm$ 2.41%** |
| **IndicBART D2 Ahirani (32k)** | **D3 (Standard)** | 17.94% | 17.24% | 18.24% | 18.70% | 17.57% | **17.94% $\pm$ 0.57%** |
| **IndicBART D2 Ahirani (32k)** | **D4 (Varhadi)** | 33.76% | 31.74% | 34.60% | 32.89% | 35.16% | **33.63% $\pm$ 1.36%** |
| **IndicBART D2 Ahirani (32k)** | **Combined** | 36.86% | 35.27% | 35.56% | 35.13% | 36.05% | **35.77% $\pm$ 0.70%** |
| **IndicBART D4 Varhadi (16k)** | **D1 (Malvani)** | 11.22% | 11.50% | 11.11% | 11.35% | 10.78% | **11.19% $\pm$ 0.27%** |
| **IndicBART D4 Varhadi (16k)** | **D2 (Ahirani)** | 2.19% | 2.58% | 2.41% | 2.27% | 2.73% | **2.44% $\pm$ 0.22%** |
| **IndicBART D4 Varhadi (16k)** | **D3 (Standard)** | 0.70% | 1.07% | 1.25% | 0.70% | 1.16% | **0.98% $\pm$ 0.26%** |
| **IndicBART D4 Varhadi (16k)** | **D4 (Varhadi)** | 13.08% | 12.58% | 12.30% | 12.89% | 12.15% | **12.60% $\pm$ 0.39%** |
| **IndicBART D4 Varhadi (16k)** | **Combined** | 6.82% | 6.96% | 6.79% | 6.83% | 6.73% | **6.83% $\pm$ 0.08%** |
| **IndicBART D4 Varhadi (32k)** | **D1 (Malvani)** | 29.32% | 28.12% | 33.12% | 31.04% | 34.99% | **31.32% $\pm$ 2.79%** |
| **IndicBART D4 Varhadi (32k)** | **D2 (Ahirani)** | 19.62% | 20.49% | 21.93% | 21.50% | 20.49% | **20.81% $\pm$ 0.92%** |
| **IndicBART D4 Varhadi (32k)** | **D3 (Standard)** | 16.40% | 15.47% | 16.01% | 15.68% | 15.43% | **15.80% $\pm$ 0.41%** |
| **IndicBART D4 Varhadi (32k)** | **D4 (Varhadi)** | 28.63% | 29.33% | 30.69% | 29.94% | 30.24% | **29.77% $\pm$ 0.80%** |
| **IndicBART D4 Varhadi (32k)** | **Combined** | 23.55% | 23.41% | 25.52% | 24.62% | 25.38% | **24.50% $\pm$ 0.99%** |
| **mT5-Small Combined (16k)** | **D1 (Malvani)** | 35.27% | - | - | - | - | **35.27%** |
| **mT5-Small Combined (16k)** | **D2 (Ahirani)** | 13.97% | - | - | - | - | **13.97%** |
| **mT5-Small Combined (16k)** | **D3 (Standard)** | 1.72% | - | - | - | - | **1.72%** |
| **mT5-Small Combined (16k)** | **D4 (Varhadi)** | 15.75% | - | - | - | - | **15.75%** |
| **mT5-Small Combined (16k)** | **Combined** | 16.90% | - | - | - | - | **16.90%** |
| **mT5-Small Combined (32k)** | **D1 (Malvani)** | 34.50% | - | - | - | - | **34.50%** |
| **mT5-Small Combined (32k)** | **D2 (Ahirani)** | 13.01% | - | - | - | - | **13.01%** |
| **mT5-Small Combined (32k)** | **D3 (Standard)** | 1.07% | - | - | - | - | **1.07%** |
| **mT5-Small Combined (32k)** | **D4 (Varhadi)** | 14.96% | - | - | - | - | **14.96%** |
| **mT5-Small Combined (32k)** | **Combined** | 16.10% | - | - | - | - | **16.10%** |
| **mT5-Small Raw Unverified (32k)** | **D1 (Malvani)** | 34.50% | - | - | - | - | **34.50%** |
| **mT5-Small Raw Unverified (32k)** | **D2 (Ahirani)** | 13.27% | - | - | - | - | **13.27%** |
| **mT5-Small Raw Unverified (32k)** | **D3 (Standard)** | 0.98% | - | - | - | - | **0.98%** |
| **mT5-Small Raw Unverified (32k)** | **D4 (Varhadi)** | 14.91% | - | - | - | - | **14.91%** |
| **mT5-Small Raw Unverified (32k)** | **Combined** | 16.13% | - | - | - | - | **16.13%** |
| **mT5-Small D1 Malvani (16k)** | **D1 (Malvani)** | 34.64% | - | - | - | - | **34.64%** |
| **mT5-Small D1 Malvani (16k)** | **D2 (Ahirani)** | 10.74% | - | - | - | - | **10.74%** |
| **mT5-Small D1 Malvani (16k)** | **D3 (Standard)** | 5.99% | - | - | - | - | **5.99%** |
| **mT5-Small D1 Malvani (16k)** | **D4 (Varhadi)** | 14.37% | - | - | - | - | **14.37%** |
| **mT5-Small D1 Malvani (16k)** | **Combined** | 16.59% | - | - | - | - | **16.59%** |
| **mT5-Small D1 Malvani (32k)** | **D1 (Malvani)** | 32.73% | - | - | - | - | **32.73%** |
| **mT5-Small D1 Malvani (32k)** | **D2 (Ahirani)** | 8.59% | - | - | - | - | **8.59%** |
| **mT5-Small D1 Malvani (32k)** | **D3 (Standard)** | 2.46% | - | - | - | - | **2.46%** |
| **mT5-Small D1 Malvani (32k)** | **D4 (Varhadi)** | 10.76% | - | - | - | - | **10.76%** |
| **mT5-Small D1 Malvani (32k)** | **Combined** | 13.81% | - | - | - | - | **13.81%** |
| **mT5-Small D2 Ahirani (16k)** | **D1 (Malvani)** | 9.72% | - | - | - | - | **9.72%** |
| **mT5-Small D2 Ahirani (16k)** | **D2 (Ahirani)** | 10.76% | - | - | - | - | **10.76%** |
| **mT5-Small D2 Ahirani (16k)** | **D3 (Standard)** | 1.05% | - | - | - | - | **1.05%** |
| **mT5-Small D2 Ahirani (16k)** | **D4 (Varhadi)** | 6.45% | - | - | - | - | **6.45%** |
| **mT5-Small D2 Ahirani (16k)** | **Combined** | 7.09% | - | - | - | - | **7.09%** |
| **mT5-Small D2 Ahirani (32k)** | **D1 (Malvani)** | 12.20% | - | - | - | - | **12.20%** |
| **mT5-Small D2 Ahirani (32k)** | **D2 (Ahirani)** | 11.26% | - | - | - | - | **11.26%** |
| **mT5-Small D2 Ahirani (32k)** | **D3 (Standard)** | 1.07% | - | - | - | - | **1.07%** |
| **mT5-Small D2 Ahirani (32k)** | **D4 (Varhadi)** | 7.92% | - | - | - | - | **7.92%** |
| **mT5-Small D2 Ahirani (32k)** | **Combined** | 8.22% | - | - | - | - | **8.22%** |
| **mT5-Small D4 Varhadi (16k)** | **D1 (Malvani)** | 8.34% | - | - | - | - | **8.34%** |
| **mT5-Small D4 Varhadi (16k)** | **D2 (Ahirani)** | 3.87% | - | - | - | - | **3.87%** |
| **mT5-Small D4 Varhadi (16k)** | **D3 (Standard)** | 1.81% | - | - | - | - | **1.81%** |
| **mT5-Small D4 Varhadi (16k)** | **D4 (Varhadi)** | 13.85% | - | - | - | - | **13.85%** |
| **mT5-Small D4 Varhadi (16k)** | **Combined** | 6.98% | - | - | - | - | **6.98%** |
| **mT5-Small D4 Varhadi (32k)** | **D1 (Malvani)** | 8.30% | - | - | - | - | **8.30%** |
| **mT5-Small D4 Varhadi (32k)** | **D2 (Ahirani)** | 3.63% | - | - | - | - | **3.63%** |
| **mT5-Small D4 Varhadi (32k)** | **D3 (Standard)** | 1.42% | - | - | - | - | **1.42%** |
| **mT5-Small D4 Varhadi (32k)** | **D4 (Varhadi)** | 13.71% | - | - | - | - | **13.71%** |
| **mT5-Small D4 Varhadi (32k)** | **Combined** | 6.78% | - | - | - | - | **6.78%** |

---

## 3. Dialectwise & Foldwise chrF++ Score Matrix

| Model Name | Dialect Variety | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | **Mean chrF++ ($\pm \sigma$)** |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **IndicBART Combined (16k)** | **D1 (Malvani)** | 44.70 | 41.30 | 43.83 | 44.55 | 42.68 | **43.41 $\pm$ 1.43** |
| **IndicBART Combined (16k)** | **D2 (Ahirani)** | 73.28 | 72.29 | 73.23 | 72.17 | 73.99 | **72.99 $\pm$ 0.76** |
| **IndicBART Combined (16k)** | **D3 (Standard)** | 88.00 | 88.34 | 87.54 | 85.96 | 85.57 | **87.08 $\pm$ 1.24** |
| **IndicBART Combined (16k)** | **D4 (Varhadi)** | 72.79 | 73.03 | 74.13 | 73.37 | 74.60 | **73.58 $\pm$ 0.76** |
| **IndicBART Combined (16k)** | **Combined** | 69.74 | 68.86 | 69.74 | 69.04 | 69.28 | **69.33 $\pm$ 0.40** |
| **IndicBART Combined (32k)** | **D1 (Malvani)** | 64.73 | 59.29 | 61.79 | 66.44 | 53.04 | **61.06 $\pm$ 5.25** |
| **IndicBART Combined (32k)** | **D2 (Ahirani)** | 76.39 | 74.74 | 77.80 | 81.19 | 69.11 | **75.85 $\pm$ 4.45** |
| **IndicBART Combined (32k)** | **D3 (Standard)** | 85.20 | 86.19 | 84.28 | 89.13 | 76.15 | **84.19 $\pm$ 4.85** |
| **IndicBART Combined (32k)** | **D4 (Varhadi)** | 79.22 | 70.94 | 71.60 | 78.00 | 71.82 | **74.32 $\pm$ 3.96** |
| **IndicBART Combined (32k)** | **Combined** | 76.21 | 72.66 | 73.73 | 78.52 | 67.39 | **73.70 $\pm$ 4.20** |
| **IndicBART Raw Unverified (32k)** | **D1 (Malvani)** | 45.83 | 44.11 | 43.35 | 41.53 | 43.17 | **43.60 $\pm$ 1.56** |
| **IndicBART Raw Unverified (32k)** | **D2 (Ahirani)** | 75.72 | 71.80 | 73.08 | 71.70 | 72.33 | **72.93 $\pm$ 1.66** |
| **IndicBART Raw Unverified (32k)** | **D3 (Standard)** | 85.06 | 85.39 | 85.86 | 85.42 | 85.60 | **85.47 $\pm$ 0.29** |
| **IndicBART Raw Unverified (32k)** | **D4 (Varhadi)** | 71.39 | 72.99 | 71.90 | 73.20 | 73.81 | **72.66 $\pm$ 0.99** |
| **IndicBART Raw Unverified (32k)** | **Combined** | 69.52 | 68.60 | 68.59 | 68.05 | 68.78 | **68.71 $\pm$ 0.53** |
| **IndicBART D1 Malvani (16k)** | **D1 (Malvani)** | 74.76 | 74.03 | 73.65 | 73.76 | 73.42 | **73.92 $\pm$ 0.52** |
| **IndicBART D1 Malvani (16k)** | **D2 (Ahirani)** | 89.54 | 88.93 | 88.66 | 88.26 | 89.11 | **88.90 $\pm$ 0.48** |
| **IndicBART D1 Malvani (16k)** | **D3 (Standard)** | 95.61 | 95.77 | 95.44 | 95.63 | 95.76 | **95.64 $\pm$ 0.13** |
| **IndicBART D1 Malvani (16k)** | **D4 (Varhadi)** | 87.14 | 86.85 | 86.75 | 86.10 | 87.01 | **86.77 $\pm$ 0.40** |
| **IndicBART D1 Malvani (16k)** | **Combined** | 86.58 | 86.20 | 85.94 | 85.75 | 86.13 | **86.12 $\pm$ 0.31** |
| **IndicBART D1 Malvani (32k)** | **D1 (Malvani)** | 39.39 | 39.37 | 39.20 | 38.68 | 39.57 | **39.24 $\pm$ 0.34** |
| **IndicBART D1 Malvani (32k)** | **D2 (Ahirani)** | 74.16 | 76.44 | 75.22 | 76.67 | 75.37 | **75.57 $\pm$ 1.01** |
| **IndicBART D1 Malvani (32k)** | **D3 (Standard)** | 86.89 | 88.74 | 89.75 | 87.14 | 88.06 | **88.12 $\pm$ 1.17** |
| **IndicBART D1 Malvani (32k)** | **D4 (Varhadi)** | 73.73 | 74.28 | 75.54 | 74.53 | 75.32 | **74.68 $\pm$ 0.75** |
| **IndicBART D1 Malvani (32k)** | **Combined** | 68.68 | 69.90 | 70.13 | 69.45 | 69.74 | **69.58 $\pm$ 0.56** |
| **IndicBART D2 Ahirani (16k)** | **D1 (Malvani)** | 86.78 | 86.57 | 87.26 | 87.00 | 86.65 | **86.85 $\pm$ 0.28** |
| **IndicBART D2 Ahirani (16k)** | **D2 (Ahirani)** | 92.67 | 92.57 | 92.67 | 92.67 | 92.60 | **92.64 $\pm$ 0.05** |
| **IndicBART D2 Ahirani (16k)** | **D3 (Standard)** | 99.24 | 99.06 | 99.27 | 99.48 | 99.29 | **99.27 $\pm$ 0.15** |
| **IndicBART D2 Ahirani (16k)** | **D4 (Varhadi)** | 92.70 | 92.77 | 92.08 | 92.37 | 92.72 | **92.53 $\pm$ 0.30** |
| **IndicBART D2 Ahirani (16k)** | **Combined** | 92.73 | 92.62 | 92.71 | 92.77 | 92.70 | **92.71 $\pm$ 0.06** |
| **IndicBART D2 Ahirani (32k)** | **D1 (Malvani)** | 50.07 | 50.80 | 52.95 | 51.61 | 54.98 | **52.08 $\pm$ 1.94** |
| **IndicBART D2 Ahirani (32k)** | **D2 (Ahirani)** | 70.16 | 72.73 | 73.14 | 74.81 | 69.81 | **72.13 $\pm$ 2.11** |
| **IndicBART D2 Ahirani (32k)** | **D3 (Standard)** | 85.96 | 86.83 | 86.14 | 85.56 | 86.01 | **86.10 $\pm$ 0.46** |
| **IndicBART D2 Ahirani (32k)** | **D4 (Varhadi)** | 73.35 | 74.98 | 72.45 | 74.28 | 72.02 | **73.42 $\pm$ 1.23** |
| **IndicBART D2 Ahirani (32k)** | **Combined** | 69.86 | 71.30 | 71.12 | 71.53 | 70.63 | **70.89 $\pm$ 0.66** |
| **IndicBART D4 Varhadi (16k)** | **D1 (Malvani)** | 91.86 | 91.91 | 92.11 | 91.68 | 92.14 | **91.94 $\pm$ 0.19** |
| **IndicBART D4 Varhadi (16k)** | **D2 (Ahirani)** | 98.25 | 98.24 | 98.09 | 98.44 | 98.04 | **98.21 $\pm$ 0.16** |
| **IndicBART D4 Varhadi (16k)** | **D3 (Standard)** | 99.60 | 99.25 | 99.10 | 99.60 | 99.13 | **99.34 $\pm$ 0.25** |
| **IndicBART D4 Varhadi (16k)** | **D4 (Varhadi)** | 90.23 | 90.32 | 90.62 | 90.22 | 90.71 | **90.42 $\pm$ 0.23** |
| **IndicBART D4 Varhadi (16k)** | **Combined** | 94.93 | 94.88 | 94.93 | 94.93 | 94.96 | **94.93 $\pm$ 0.03** |
| **IndicBART D4 Varhadi (32k)** | **D1 (Malvani)** | 77.48 | 78.01 | 74.02 | 75.44 | 71.96 | **75.38 $\pm$ 2.49** |
| **IndicBART D4 Varhadi (32k)** | **D2 (Ahirani)** | 83.64 | 83.22 | 81.53 | 81.81 | 82.67 | **82.57 $\pm$ 0.90** |
| **IndicBART D4 Varhadi (32k)** | **D3 (Standard)** | 86.71 | 87.39 | 86.98 | 87.09 | 87.32 | **87.10 $\pm$ 0.27** |
| **IndicBART D4 Varhadi (32k)** | **D4 (Varhadi)** | 77.60 | 76.92 | 75.71 | 76.21 | 76.11 | **76.51 $\pm$ 0.75** |
| **IndicBART D4 Varhadi (32k)** | **Combined** | 81.30 | 81.33 | 79.49 | 80.07 | 79.44 | **80.33 $\pm$ 0.94** |
| **mT5-Small Combined (16k)** | **D1 (Malvani)** | 74.14 | - | - | - | - | **74.14** |
| **mT5-Small Combined (16k)** | **D2 (Ahirani)** | 89.88 | - | - | - | - | **89.88** |
| **mT5-Small Combined (16k)** | **D3 (Standard)** | 99.13 | - | - | - | - | **99.13** |
| **mT5-Small Combined (16k)** | **D4 (Varhadi)** | 89.12 | - | - | - | - | **89.12** |
| **mT5-Small Combined (16k)** | **Combined** | 87.83 | - | - | - | - | **87.83** |
| **mT5-Small Combined (32k)** | **D1 (Malvani)** | 74.85 | - | - | - | - | **74.85** |
| **mT5-Small Combined (32k)** | **D2 (Ahirani)** | 89.91 | - | - | - | - | **89.91** |
| **mT5-Small Combined (32k)** | **D3 (Standard)** | 99.22 | - | - | - | - | **99.22** |
| **mT5-Small Combined (32k)** | **D4 (Varhadi)** | 89.33 | - | - | - | - | **89.33** |
| **mT5-Small Combined (32k)** | **Combined** | 88.09 | - | - | - | - | **88.09** |
| **mT5-Small Raw Unverified (32k)** | **D1 (Malvani)** | 75.16 | - | - | - | - | **75.16** |
| **mT5-Small Raw Unverified (32k)** | **D2 (Ahirani)** | 90.12 | - | - | - | - | **90.12** |
| **mT5-Small Raw Unverified (32k)** | **D3 (Standard)** | 99.34 | - | - | - | - | **99.34** |
| **mT5-Small Raw Unverified (32k)** | **D4 (Varhadi)** | 89.34 | - | - | - | - | **89.34** |
| **mT5-Small Raw Unverified (32k)** | **Combined** | 88.25 | - | - | - | - | **88.25** |
| **mT5-Small D1 Malvani (16k)** | **D1 (Malvani)** | 74.99 | - | - | - | - | **74.99** |
| **mT5-Small D1 Malvani (16k)** | **D2 (Ahirani)** | 91.88 | - | - | - | - | **91.88** |
| **mT5-Small D1 Malvani (16k)** | **D3 (Standard)** | 95.96 | - | - | - | - | **95.96** |
| **mT5-Small D1 Malvani (16k)** | **D4 (Varhadi)** | 90.34 | - | - | - | - | **90.34** |
| **mT5-Small D1 Malvani (16k)** | **Combined** | 88.10 | - | - | - | - | **88.10** |
| **mT5-Small D1 Malvani (32k)** | **D1 (Malvani)** | 76.13 | - | - | - | - | **76.13** |
| **mT5-Small D1 Malvani (32k)** | **D2 (Ahirani)** | 93.53 | - | - | - | - | **93.53** |
| **mT5-Small D1 Malvani (32k)** | **D3 (Standard)** | 98.19 | - | - | - | - | **98.19** |
| **mT5-Small D1 Malvani (32k)** | **D4 (Varhadi)** | 92.78 | - | - | - | - | **92.78** |
| **mT5-Small D1 Malvani (32k)** | **Combined** | 89.94 | - | - | - | - | **89.94** |
| **mT5-Small D2 Ahirani (16k)** | **D1 (Malvani)** | 94.88 | - | - | - | - | **94.88** |
| **mT5-Small D2 Ahirani (16k)** | **D2 (Ahirani)** | 92.48 | - | - | - | - | **92.48** |
| **mT5-Small D2 Ahirani (16k)** | **D3 (Standard)** | 99.25 | - | - | - | - | **99.25** |
| **mT5-Small D2 Ahirani (16k)** | **D4 (Varhadi)** | 95.97 | - | - | - | - | **95.97** |
| **mT5-Small D2 Ahirani (16k)** | **Combined** | 95.59 | - | - | - | - | **95.59** |
| **mT5-Small D2 Ahirani (32k)** | **D1 (Malvani)** | 92.65 | - | - | - | - | **92.65** |
| **mT5-Small D2 Ahirani (32k)** | **D2 (Ahirani)** | 92.25 | - | - | - | - | **92.25** |
| **mT5-Small D2 Ahirani (32k)** | **D3 (Standard)** | 99.40 | - | - | - | - | **99.40** |
| **mT5-Small D2 Ahirani (32k)** | **D4 (Varhadi)** | 94.64 | - | - | - | - | **94.64** |
| **mT5-Small D2 Ahirani (32k)** | **Combined** | 94.66 | - | - | - | - | **94.66** |
| **mT5-Small D4 Varhadi (16k)** | **D1 (Malvani)** | 95.23 | - | - | - | - | **95.23** |
| **mT5-Small D4 Varhadi (16k)** | **D2 (Ahirani)** | 97.84 | - | - | - | - | **97.84** |
| **mT5-Small D4 Varhadi (16k)** | **D3 (Standard)** | 99.01 | - | - | - | - | **99.01** |
| **mT5-Small D4 Varhadi (16k)** | **D4 (Varhadi)** | 89.98 | - | - | - | - | **89.98** |
| **mT5-Small D4 Varhadi (16k)** | **Combined** | 95.50 | - | - | - | - | **95.50** |
| **mT5-Small D4 Varhadi (32k)** | **D1 (Malvani)** | 95.33 | - | - | - | - | **95.33** |
| **mT5-Small D4 Varhadi (32k)** | **D2 (Ahirani)** | 97.98 | - | - | - | - | **97.98** |
| **mT5-Small D4 Varhadi (32k)** | **D3 (Standard)** | 99.00 | - | - | - | - | **99.00** |
| **mT5-Small D4 Varhadi (32k)** | **D4 (Varhadi)** | 90.43 | - | - | - | - | **90.43** |
| **mT5-Small D4 Varhadi (32k)** | **Combined** | 95.67 | - | - | - | - | **95.67** |

---

