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
   $$\text{Word Edit Distance (\%)} = \frac{S + D + I}{\max(|T_{\text{orig}}|, |T_{\text{std}}|)} \times 100\% \quad \in [0, 100\%]$$
   $$\text{Character Edit Distance (\%)} = \frac{\text{CharEdits}}{\max(\text{len}(T_{\text{orig}}), \text{len}(T_{\text{std}}))} \times 100\% \quad \in [0, 100\%]$$
3. **Standard Marathi Lexical OOV Ratio**:
   Measures the percentage of words in $T_{\text{orig}}$ absent from the Standard Marathi (D3) reference lexicon.
4. **Composite Divergence Metric**:
   $$\text{Divergence Score} = 0.45 \times \text{Word Edit Distance} + 0.30 \times \text{Char Edit Distance} + 0.25 \times \text{OOV Ratio}$$

---

## 2. Dialectwise Extraction Summary

| Dialect Code | Region / Variety | Unique Spoken Transcripts | Top Extracted Chunk | Avg Divergence Score | Avg Word Edit Distance (%) | Standard OOV Ratio (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **D1** | **Malvani** *(South Konkan)* | 212 | **100** | **46.21** | **46.94%** | **78.34%** |
| **D2** | **Ahirani** *(Khandesh / North Konkan)* | 152 | **100** | **31.58** | **22.49%** | **74.05%** |
| **D4** | **Varhadi** *(Vidarbha)* | 198 | **100** | **33.37** | **26.54%** | **75.33%** |

---

## 3. Detailed Linguistic Breakdown & Top Spoken Samples

---

### 3.1 📍 D1 Malvani (South Konkan: Ratnagiri / Sindhudurg)

Malvani is characterized by nominal and verbal inflections ending in `-ो` / `-ूचे`, dative/purpose marker `खाती` (`खातिर`), and first-person pronoun `माका`.

| Rank | Score | Word Edit (%) | OOV (%) | Original Spoken RESPIN Transcript | Normalized Standard Pune Marathi | Key Morphosyntactic Features |
| :---: | :---: | :---: | :---: | :--- | :--- | :--- |
| **1** | **77.17** | 83.33% | 100.0% | `टँकरात भरलल्या दुधाक पायपातसून सगळीकडे पोचयतत` | `टँकरात भरलेल्या दुधाला पाईपमधून सगळीकडे पोहोचवतात` | • Dative suffix `-क` (`दुधाक` $\rightarrow$ `दुधाला`)<br>• Ablative suffix `-तसून` (`पायपातसून` $\rightarrow$ `पाईपमधून`)<br>• Plural verb ending `-तत` (`पोचयतत` $\rightarrow$ `पोहोचवतात`) |
| **2** | **71.11** | 85.71% | 83.33% | `ऊसाच्या पिकाक खयच्या सिंचनाचो उपयोग करतत ?` | `उसाच्या पिकाला कोणत्या सिंचनाचा उपयोग करतात?` | • Interrogative `खयच्या` (`कोणत्या`)<br>• Genitive inflection `-चो` (`सिंचनाचो` $\rightarrow$ `सिंचनाचा`) |
| **3** | **68.33** | 83.33% | 83.33% | `सगळ्या प्रकारच्या मातयेचो कस सारखो नसता` | `सर्व प्रकारच्या मातीचे कोणते सारखे नसते?` | • Suffix `-चो` (`मातयेचो` $\rightarrow$ `मातीचा/मातीचे`)<br>• Adjective agreement (`सारखो नसता` $\rightarrow$ `सारखा नसतो`) |
| **4** | **67.31** | 85.71% | 85.71% | `बुरशी पासून झाडांका लय प्रकारचे तरास जातत` | `बुरशीपासून झाडांना अनेक प्रकारचा त्रास होतो` | • Plural accusative `-ंका` (`झाडांका` $\rightarrow$ `झाडांना`)<br>• Quantifier `लय` (`अनेक / खूप`)<br>• Verb `जातत` (`होतात`) |
| **5** | **67.07** | 80.00% | 60.00% | `कर्ज कोनाकपण फेडूक मेलता काय ?` | `कर्ज कोणालाही फेडता येते का?` | • Indefinite pronoun `कोनाकपण` (`कोणालाही`)<br>• Potential verb `मेलता` (`मिळते / येते`)<br>• Infinitive `-ूक` (`फेडूक` $\rightarrow$ `फेडायला`) |
| **6** | **65.34** | 83.33% | 75.00% | `शेतमालाचो दर खेच्यावरना ठरवतत ?` | `शेतमालाचा दर कशावरून ठरवतात?` | • Interrogative `खेच्यावरना` (`कशावरून`)<br>• Verb `-तत` (`ठरवतत` $\rightarrow$ `ठरवतात`) |
| **7** | **63.17** | 75.00% | 75.00% | `माका क्रेडिट कार्डाक आजुन किती पैशे भरूचे हत ?` | `मला क्रेडिट कार्डसाठी अजून किती पैसे भरावे लागतील?` | • Dative pronoun `माका` (`मला`)<br>• Obligative suffix `-ूचे हत` (`भरूचे हत` $\rightarrow$ `भरावे लागतील`) |

---

### 3.2 📍 D2 Ahirani (Khandesh & North Konkan: Palghar / Thane / Jalgaon)

Ahirani exhibits Gujarati and Western Indo-Aryan substrata, featuring locative `-मा` / `-मझार`, universal copula `शे`, plural marker `-स्ले / -स्नं`, and verb inflection `-स`.

| Rank | Score | Word Edit (%) | OOV (%) | Original Spoken RESPIN Transcript | Normalized Standard Pune Marathi | Key Morphosyntactic Features |
| :---: | :---: | :---: | :---: | :--- | :--- | :--- |
| **1** | **72.49** | 71.43% | 100.0% | `कर्जलेवावर गहाण ठेयेल मालमत्ता ईकता नयी येस` | `कर्ज घेतल्यावर गहाण ठेवलेली मालमत्ता विकता येत नाही` | • Past participle `ठेयेल` (`ठेवलेली`)<br>• Verb `ईकता` (`विकता`)<br>• Negative auxiliary `नयी येस` (`येत नाही`) |
| **2** | **69.28** | 75.00% | 100.0% | `शेतीना अवजारेस्नं परदरशन मांडं तवय त्यास्नी ईक्री वाढी` | `शेतीची अवजारे प्रदर्शन मांडल्यावर त्यांची विक्री वाढली` | • Plural genitive `-स्नं / -स्नी` (`अवजारेस्नं` $\rightarrow$ `अवजारांचे`, `त्यास्नी` $\rightarrow$ `त्यांची`)<br>• Temporal conjunction `तवय` (`तेव्हा`)<br>• Lexical `ईक्री` (`विक्री`) |
| **3** | **66.28** | 77.78% | 88.89% | `भारतमा परतेक राज्यनी वरीसनं उतपननी वाढ आल्लग आल्लग शे` | `भारतात प्रत्येक राज्याची वार्षिक उत्पन्नाची वाढ वेगवेगळी आहे` | • Locative `-मा` (`भारतमा` $\rightarrow$ `भारतात`)<br>• Genitive `-नी` (`राज्यनी` $\rightarrow$ `राज्याची`)<br>• Adverb `आल्लग आल्लग` (`वेगवेगळी`)<br>• Copula `शे` (`आहे`) |
| **4** | **63.91** | 66.67% | 77.78% | `फळे देनारा झाडेस्ले शेतकरी जमीनना आंदाज लीसनी काटछाट करस` | `फळे देणाऱ्या झाडांना शेतकरी जमिनीचा अंदाज घेऊन छाटणी करतो` | • Dative plural `-स्ले` (`झाडेस्ले` $\rightarrow$ `झाडांना`)<br>• Conjunctive participle `लीसनी` (`घेऊन`)<br>• Present tense verb `-स` (`करस` $\rightarrow$ `करतो`) |
| **5** | **63.34** | 66.67% | 88.89% | `भोपयाना भाजीना वापर रायता लोणचं खीर बनाडाना करता करतंस` | `भोपळ्याच्या भाजीचा वापर रायता लोणचे खीर बनवण्यासाठी करतात` | • Phonological cluster `भोपयाना` (`भोपळ्याच्या`)<br>• Infinitive `बनाडाना करता` (`बनवण्यासाठी`)<br>• Habitual verb `करतंस` (`करतात`) |
| **6** | **62.10** | 63.64% | 72.73% | `पी.पी.एफ मझार जमा करेल पैसा हायातीमा काढता येतसका नैत ?` | `पी.पी.एफ. मध्ये जमा केलेले पैसे आयुष्यात काढता येतात का नाहीत?` | • Inessive postposition `मझार` (`मध्ये`)<br>• Locative postposition `-मा` (`हायातीमा` $\rightarrow$ `आयुष्यात`)<br>• Negative tag `नैत` (`नाहीत`) |

---

### 3.3 📍 D4 Varhadi (Vidarbha: Amravati / Akola / Yavatmal)

Varhadi features distinctive dative pronoun `मले`, interrogatives `कोंते / कोनचा`, retroflex softening ($ळ \rightarrow ल$, $ण \rightarrow न$), and verb endings in `-न / -ीन`.

| Rank | Score | Word Edit (%) | OOV (%) | Original Spoken RESPIN Transcript | Normalized Standard Pune Marathi | Key Morphosyntactic Features |
| :---: | :---: | :---: | :---: | :--- | :--- | :--- |
| **1** | **60.83** | 66.67% | 83.33% | `माती परीक्षन कुठ अस कस कराव ?` | `माती परीक्षण कुठे आणि कसे करावे?` | • Conjunction `अस` (`आणि`)<br>• Adverb `कुठ` (`कुठे`)<br>• Softened retroflex `परीक्सन` (`परीक्षण`) |
| **2** | **59.72** | 71.43% | 85.71% | `खातं काढताना मले किती पैसा भरा लागन ?` | `खाते उघडताना मला किती पैसे भरावे लागतील?` | • Dative pronoun `मले` (`मला`)<br>• Future auxiliary `लागन` (`लागतील`)<br>• Lexical `काढताना` (`उघडताना`) |
| **3** | **55.32** | 66.67% | 66.67% | `क्रेडिट कार्ड घ्यासाठी मले कोनचा अर्ज भरनं सोप राहीन ?` | `क्रेडिट कार्ड घेण्यासाठी मला कोणता अर्ज भरणे सोपे राहील?` | • Interrogative `कोनचा` (`कोणता`)<br>• Infinitive `घ्यासाठी` (`घेण्यासाठी`)<br>• Future auxiliary `राहीन` (`राहील`) |
| **4** | **55.11** | 62.50% | 75.00% | `शेती ड्रीचींगनं कराची रायल्यास कायचा वापर करा लागीन ?` | `शेती ड्रीचिंगने करायची राहिल्यास कशाचा वापर करावा लागेल?` | • Suffix `कराची रायल्यास` (`करायची राहिल्यास`)<br>• Interrogative `कायचा` (`कशाचा`)<br>• Future `लागीन` (`लागेल`) |
| **5** | **54.44** | 57.14% | 85.71% | `कर्ज घेतल्यावर मले ते कस फेडा लागन ?` | `कर्ज घेतल्यावर मला ते कसे फेडावे लागेल?` | • Pronoun `मले` (`मला`)<br>• Future verb `लागन` (`लागेल`) |
| **6** | **53.46** | 62.50% | 87.50% | `कास्तकार मोबाईल वरून शेतीचं मार्गदर्शन कस भेटवू शकते ?` | `शेतकरी मोबाईलवरून शेतीचे मार्गदर्शन कसे मिळवू शकतो?` | • Lexical noun `कास्तकार` (`शेतकरी`)<br>• Potential verb `भेटवू शकते` (`मिळवू शकतो`) |

---

## 4. Exported Clean Datasets for Manual Verification

The extracted top divergent spoken transcripts are saved in clean UTF-8 CSV format under `reports/respin_divergent_samples/`:

* 📄 **D1 Malvani Spoken CSV**: [reports/respin_divergent_samples/respin_d1_top_divergent.csv](file:///d:/dialect-norm/reports/respin_divergent_samples/respin_d1_top_divergent.csv)
* 📄 **D2 Ahirani Spoken CSV**: [reports/respin_divergent_samples/respin_d2_top_divergent.csv](file:///d:/dialect-norm/reports/respin_divergent_samples/respin_d2_top_divergent.csv)
* 📄 **D4 Varhadi Spoken CSV**: [reports/respin_divergent_samples/respin_d4_top_divergent.csv](file:///d:/dialect-norm/reports/respin_divergent_samples/respin_d4_top_divergent.csv)
* 📄 **Summary Digest**: [reports/respin_divergent_samples/respin_divergence_summary.md](file:///d:/dialect-norm/reports/respin_divergent_samples/respin_divergence_summary.md)
