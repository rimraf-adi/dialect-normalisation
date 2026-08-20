# Comprehensive Linguistic Analysis: Divergent Spoken Marathi Dialect Utterances (D1, D2, D4)

**Benchmark Dataset**: IISc RESPIN Spoken Dialect Corpus (`IISc_RESPIN_test_mr`)  
**Methodology**: Unsupervised Dialect Divergence Ranking via Fine-Tuned mT5 Normalizer & Standard Marathi (D3) Lexical Out-of-Vocabulary (OOV) Scoring  
**Target Dialects**:
1. **D1 Malvani** (South Konkan: Ratnagiri / Sindhudurg)
2. **D2 Ahirani** (Khandesh & North Konkan: Palghar / Thane / Jalgaon)
3. **D4 Varhadi** (Vidarbha: Amravati / Akola / Yavatmal)

---

## 1. Methodology: Measuring Dialect Divergence on Unpaired Spoken Audio

In natural speech corpora like IISc RESPIN, speech recordings are **unpaired** (i.e. speakers spoke natural domain sentences without pre-aligned standard text). 

To objectively identify the utterances that deviate most from Pune Standard Marathi, each original spoken transcript $T_{\text{orig}}$ is processed through our computational pipeline:

1. **Machine Normalization Transformation**:
   $T_{\text{orig}}$ is passed through our fine-tuned `mT5-Small` normalizer to generate the predicted Standard Pune Marathi target $T_{\text{std}}$.
2. **Bounded Levenshtein Edit Distance**:
   $$\text{Word Edit Distance (WED \%)} = \frac{S + D + I}{\max(|T_{\text{orig}}|, |T_{\text{std}}|)} \times 100\% \quad \in [0, 100\%]$$
   $$\text{Character Edit Distance (CED \%)} = \frac{\text{CharEdits}}{\max(\text{len}(T_{\text{orig}}), \text{len}(T_{\text{std}}))} \times 100\% \quad \in [0, 100\%]$$
3. **Standard Marathi Lexical OOV Ratio**:
   $$\text{OOV Ratio (\%)} = \frac{\text{Count of tokens not in Standard D3 Vocab}}{|T_{\text{orig}}|} \times 100\%$$
4. **Composite Divergence Metric**:
   $$\text{Divergence Score} = 0.45 \times \text{WED} + 0.30 \times \text{CED} + 0.25 \times \text{OOV}$$

---

## 2. Dialectwise Extraction Summary

| Dialect Code | Region / Variety | Unique Spoken Transcripts | Top Extracted Chunk | Avg Divergence Score | Avg Word Edit Distance (%) | Standard OOV Ratio (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **D1** | **Malvani** *(South Konkan)* | 212 | **100** | **46.21** | **46.94%** | **78.34%** |
| **D2** | **Ahirani** *(Khandesh / North Konkan)* | 152 | **100** | **31.58** | **22.49%** | **74.05%** |
| **D4** | **Varhadi** *(Vidarbha)* | 198 | **100** | **33.37** | **26.54%** | **75.33%** |

---

## 3. Detailed Linguistic Breakdown & Step-by-Step Sample Calculations

---

### 3.1 📍 D1 Malvani (South Konkan: Ratnagiri / Sindhudurg)

Malvani is characterized by nominal and verbal inflections ending in `-ो` / `-ूचे`, dative/purpose marker `खाती` (`खातिर`), and first-person pronoun `माका`.

| Rank | Score | WED (%) | CED (%) | OOV (%) | Original Spoken RESPIN Transcript | Normalized Standard Pune Marathi | Step-by-Step Mathematical Calculation |
| :---: | :---: | :---: | :---: | :---: | :--- | :--- | :--- |
| **1** | **77.17** | 83.33% | 48.89% | 100.0% | `टँकरात भरलल्या दुधाक पायपातसून सगळीकडे पोचयतत` | `टँकरात भरलेल्या दुधाकडे पाहिजे` | **WED**: $(3S + 0D + 2I)/6 = 83.33\%$<br>**CED**: $22/45 = 48.89\%$<br>**OOV**: $6/6 = 100.0\%$<br>$\rightarrow 0.45(83.33) + 0.30(48.89) + 0.25(100.0) = \mathbf{77.17}$ |
| **2** | **71.11** | 85.71% | 39.02% | 83.33% | `ऊसाच्या पिकाक खयच्या सिंचनाचो उपयोग करतत ?` | `ऊसाच्या पिकांवर सिंचनाचा उपयोग करत आहे का?` | **WED**: $(3S + 2D + 1I)/7 = 85.71\%$<br>**CED**: $16/41 = 39.02\%$<br>**OOV**: $5/6 = 83.33\%$<br>$\rightarrow 0.45(85.71) + 0.30(39.02) + 0.25(83.33) = \mathbf{71.11}$ |
| **3** | **68.33** | 83.33% | 33.33% | 83.33% | `सगळ्या प्रकारच्या मातयेचो कस सारखो नसता` | `सर्व प्रकारच्या मातीचे कोणते सारखे नसते?` | **WED**: $(5S + 0D + 0I)/6 = 83.33\%$<br>**CED**: $13/39 = 33.33\%$<br>**OOV**: $5/6 = 83.33\%$<br>$\rightarrow 0.45(83.33) + 0.30(33.33) + 0.25(83.33) = \mathbf{68.33}$ |
| **4** | **67.31** | 85.71% | 24.39% | 85.71% | `बुरशी पासून झाडांका लय प्रकारचे तरास जातत` | `बुरशीपासून झाडांचे प्रकारचे तरास जात आहेत` | **WED**: $(3S + 1D + 2I)/7 = 85.71\%$<br>**CED**: $10/41 = 24.39\%$<br>**OOV**: $6/7 = 85.71\%$<br>$\rightarrow 0.45(85.71) + 0.30(24.39) + 0.25(85.71) = \mathbf{67.31}$ |
| **5** | **67.07** | 80.00% | 53.57% | 60.00% | `कर्ज कोनाकपण फेडूक मेलता काय ?` | `कर्ज कोणते फेडले जाते?` | **WED**: $(3S + 0D + 1I)/5 = 80.00\%$<br>**CED**: $15/28 = 53.57\%$<br>**OOV**: $3/5 = 60.00\%$<br>$\rightarrow 0.45(80.00) + 0.30(53.57) + 0.25(60.00) = \mathbf{67.07}$ |
| **6** | **65.34** | 83.33% | 30.30% | 75.00% | `शेतमालाचो दर खेच्यावरना ठरवतत ?` | `शेतमालाचा दर खेच्यावर ठरवत आहे का?` | **WED**: $(3S + 2D + 0I)/6 = 83.33\%$<br>**CED**: $10/33 = 30.30\%$<br>**OOV**: $3/4 = 75.00\%$<br>$\rightarrow 0.45(83.33) + 0.30(30.30) + 0.25(75.00) = \mathbf{65.34}$ |
| **7** | **63.17** | 75.00% | 35.56% | 75.00% | `माका क्रेडिट कार्डाक आजुन किती पैशे भरूचे हत ?` | `मला क्रेडिट कार्डासाठी किती पैसे भरावे लागतात?` | **WED**: $(5S + 0D + 1I)/8 = 75.00\%$<br>**CED**: $16/45 = 35.56\%$<br>**OOV**: $6/8 = 75.00\%$<br>$\rightarrow 0.45(75.00) + 0.30(35.56) + 0.25(75.00) = \mathbf{63.17}$ |

---

### 3.2 📍 D2 Ahirani (Khandesh & North Konkan: Palghar / Thane / Jalgaon)

Ahirani exhibits Gujarati and Western Indo-Aryan substrata, featuring locative `-मा` / `-मझार`, universal copula `शे`, plural marker `-स्ले / -स्नं`, and verb inflection `-स`.

| Rank | Score | WED (%) | CED (%) | OOV (%) | Original Spoken RESPIN Transcript | Normalized Standard Pune Marathi | Step-by-Step Mathematical Calculation |
| :---: | :---: | :---: | :---: | :---: | :--- | :--- | :--- |
| **1** | **72.49** | 71.43% | 51.16% | 100.0% | `कर्जलेवावर गहाण ठेयेल मालमत्ता ईकता नयी येस` | `कर्जासाठी गहाण ठेवण्यासाठी मालमत्ता नये का?` | **WED**: $(4S + 0D + 1I)/7 = 71.43\%$<br>**CED**: $22/43 = 51.16\%$<br>**OOV**: $7/7 = 100.0\%$<br>$\rightarrow 0.45(71.43) + 0.30(51.16) + 0.25(100.0) = \mathbf{72.49}$ |
| **2** | **69.28** | 75.00% | 35.09% | 100.0% | `शेतीना अवजारेस्नं परदरशन मांडं तवय त्यास्नी ईक्री वाढी` | `शेतीसाठी अवजारे परदरशन मांडले, त्यामुळे त्याची ईक्री वाढते.` | **WED**: $(6S + 0D + 0I)/8 = 75.00\%$<br>**CED**: $20/57 = 35.09\%$<br>**OOV**: $8/8 = 100.0\%$<br>$\rightarrow 0.45(75.00) + 0.30(35.09) + 0.25(100.0) = \mathbf{69.28}$ |
| **3** | **66.28** | 77.78% | 30.19% | 88.89% | `भारतमा परतेक राज्यनी वरीसनं उतपननी वाढ आल्लग आल्लग शे` | `भारतात परतेक राज्याची वरीसने उतपनाची वाढ आली आहे.` | **WED**: $(6S + 0D + 1I)/9 = 77.78\%$<br>**CED**: $16/53 = 30.19\%$<br>**OOV**: $8/9 = 88.89\%$<br>$\rightarrow 0.45(77.78) + 0.30(30.19) + 0.25(88.89) = \mathbf{66.28}$ |
| **4** | **65.59** | 66.67% | 35.29% | 100.0% | `फोन पे वर मनी गनच बीमा करी ठेल शेत` | `फोनवर मनी गनच बीमा करायची आहे` | **WED**: $(3S + 0D + 3I)/9 = 66.67\%$<br>**CED**: $12/34 = 35.29\%$<br>**OOV**: $9/9 = 100.0\%$<br>$\rightarrow 0.45(66.67) + 0.30(35.29) + 0.25(100.0) = \mathbf{65.59}$ |
| **5** | **63.91** | 66.67% | 48.21% | 77.78% | `फळे देनारा झाडेस्ले शेतकरी जमीनना आंदाज लीसनी काटछाट करस` | `फळे देण्याच्या झाडांना शेतकरी जमीनचे आंदाज काढले जाते.` | **WED**: $(5S + 0D + 1I)/9 = 66.67\%$<br>**CED**: $27/56 = 48.21\%$<br>**OOV**: $7/9 = 77.78\%$<br>$\rightarrow 0.45(66.67) + 0.30(48.21) + 0.25(77.78) = \mathbf{63.91}$ |
| **6** | **63.34** | 66.67% | 37.04% | 88.89% | `भोपयाना भाजीना वापर रायता लोणचं खीर बनाडाना करता करतंस` | `भोपयाना भाजीना वापरून लोणचे खीर बनवण्यासाठी करत आहेत` | **WED**: $(5S + 0D + 1I)/9 = 66.67\%$<br>**CED**: $20/54 = 37.04\%$<br>**OOV**: $8/9 = 88.89\%$<br>$\rightarrow 0.45(66.67) + 0.30(37.04) + 0.25(88.89) = \mathbf{63.34}$ |
| **7** | **62.53** | 71.43% | 17.95% | 100.0% | `पानी ना स्त्रोत पुरवठाना करता उपेग पडतस` | `पानीचा स्त्रोत पुरवठा करता उपयोग पडतो` | **WED**: $(4S + 0D + 1I)/7 = 71.43\%$<br>**CED**: $7/39 = 17.95\%$<br>**OOV**: $7/7 = 100.0\%$<br>$\rightarrow 0.45(71.43) + 0.30(17.95) + 0.25(100.0) = \mathbf{62.53}$ |

---

### 3.3 📍 D4 Varhadi (Vidarbha: Amravati / Akola / Yavatmal)

Varhadi features distinctive dative pronoun `मले`, interrogatives `कोंते / कोनचा`, retroflex softening ($ळ \rightarrow ल$, $ण \rightarrow न$), and verb endings in `-न / -ीन`.

| Rank | Score | WED (%) | CED (%) | OOV (%) | Original Spoken RESPIN Transcript | Normalized Standard Pune Marathi | Step-by-Step Mathematical Calculation |
| :---: | :---: | :---: | :---: | :---: | :--- | :--- | :--- |
| **1** | **60.83** | 66.67% | 33.33% | 83.33% | `माती परीक्षन कुठ अस कस कराव ?` | `माती परीक्षन कुठे असते?` | **WED**: $(2S + 0D + 2I)/6 = 66.67\%$<br>**CED**: $9/27 = 33.33\%$<br>**OOV**: $5/6 = 83.33\%$<br>$\rightarrow 0.45(66.67) + 0.30(33.33) + 0.25(83.33) = \mathbf{60.83}$ |
| **2** | **59.72** | 71.43% | 20.51% | 85.71% | `खातं काढताना मले किती पैसा भरा लागन ?` | `खाते काढताना मला किती पैसे भरावे लागतील?` | **WED**: $(5S + 0D + 0I)/7 = 71.43\%$<br>**CED**: $8/39 = 20.51\%$<br>**OOV**: $6/7 = 85.71\%$<br>$\rightarrow 0.45(71.43) + 0.30(20.51) + 0.25(85.71) = \mathbf{59.72}$ |
| **3** | **55.32** | 66.67% | 28.85% | 66.67% | `क्रेडिट कार्ड घ्यासाठी मले कोनचा अर्ज भरनं सोप राहीन ?` | `क्रेडिट कार्ड घेण्यासाठी मला कोणता अर्ज भरावा लागेल?` | **WED**: $(5S + 0D + 1I)/9 = 66.67\%$<br>**CED**: $15/52 = 28.85\%$<br>**OOV**: $6/9 = 66.67\%$<br>$\rightarrow 0.45(66.67) + 0.30(28.85) + 0.25(66.67) = \mathbf{55.32}$ |
| **4** | **55.11** | 62.50% | 27.45% | 75.00% | `शेती ड्रीचींगनं कराची रायल्यास कायचा वापर करा लागीन ?` | `शेती ड्रीचींगने करायची आहे, कायचा वापर करावा लागेल?` | **WED**: $(5S + 0D + 0I)/8 = 62.50\%$<br>**CED**: $14/51 = 27.45\%$<br>**OOV**: $6/8 = 75.00\%$<br>$\rightarrow 0.45(62.50) + 0.30(27.45) + 0.25(75.00) = \mathbf{55.11}$ |
| **5** | **54.44** | 57.14% | 24.32% | 85.71% | `कर्ज घेतल्यावर मले ते कस फेडा लागन ?` | `कर्ज घेतल्यावर मला ते कसे फेडायचे आहे?` | **WED**: $(4S + 0D + 0I)/7 = 57.14\%$<br>**CED**: $9/37 = 24.32\%$<br>**OOV**: $6/7 = 85.71\%$<br>$\rightarrow 0.45(57.14) + 0.30(24.32) + 0.25(85.71) = \mathbf{54.44}$ |
| **6** | **53.46** | 62.50% | 11.54% | 87.50% | `कास्तकार मोबाईल वरून शेतीचं मार्गदर्शन कस भेटवू शकते ?` | `कास्तकार मोबाईलवरून शेतीचे मार्गदर्शन कसे मिळवू शकते?` | **WED**: $(4S + 0D + 1I)/8 = 62.50\%$<br>**CED**: $6/52 = 11.54\%$<br>**OOV**: $7/8 = 87.50\%$<br>$\rightarrow 0.45(62.50) + 0.30(11.54) + 0.25(87.50) = \mathbf{53.46}$ |
| **7** | **53.33** | 66.67% | 22.22% | 66.67% | `दसऱ्यासाठी मले कितीक रुपये कर्ज भेटन ?` | `दशऱ्यासाठी मला किती रुपये कर्ज मिळेल?` | **WED**: $(4S + 0D + 0I)/6 = 66.67\%$<br>**CED**: $8/36 = 22.22\%$<br>**OOV**: $4/6 = 66.67\%$<br>$\rightarrow 0.45(66.67) + 0.30(22.22) + 0.25(66.67) = \mathbf{53.33}$ |

---

## 4. Exported Clean Datasets for Manual Verification

The extracted top divergent spoken transcripts are saved in clean UTF-8 CSV format under `reports/respin_divergent_samples/` with full mathematical breakdown columns (`total_words`, `substitutions`, `deletions`, `insertions`, `total_chars`, `char_edits`, `oov_tokens`, `calculation_formula`):

* 📄 **D1 Malvani Spoken CSV**: [reports/respin_divergent_samples/respin_d1_top_divergent.csv](file:///d:/dialect-norm/reports/respin_divergent_samples/respin_d1_top_divergent.csv)
* 📄 **D2 Ahirani Spoken CSV**: [reports/respin_divergent_samples/respin_d2_top_divergent.csv](file:///d:/dialect-norm/reports/respin_divergent_samples/respin_d2_top_divergent.csv)
* 📄 **D4 Varhadi Spoken CSV**: [reports/respin_divergent_samples/respin_d4_top_divergent.csv](file:///d:/dialect-norm/reports/respin_divergent_samples/respin_d4_top_divergent.csv)
* 📄 **Summary Digest**: [reports/respin_divergent_samples/respin_divergence_summary.md](file:///d:/dialect-norm/reports/respin_divergent_samples/respin_divergence_summary.md)
