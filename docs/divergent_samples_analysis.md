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

To objectively identify the utterances that deviate most from Pune Standard Marathi, each original spoken transcript $T_{\text{orig}}$ is processed through our two-stage computational pipeline:

1. **Machine Normalization Transformation**:
   $T_{\text{orig}}$ is passed through our SOTA fine-tuned `mT5-Small` normalizer to generate the predicted Standard Pune Marathi target $T_{\text{std}}$.
2. **Normalization Edit Distance**:
   $$\text{WER} = \text{jiwer.wer}(T_{\text{std}}, T_{\text{orig}}) \times 100\%$$
   $$\text{CER} = \text{jiwer.cer}(T_{\text{std}}, T_{\text{orig}}) \times 100\%$$
3. **Standard Marathi Lexical OOV Ratio**:
   Measures the percentage of words in $T_{\text{orig}}$ absent from the Standard Marathi (D3) lexicon.
4. **Composite Divergence Metric**:
   $$\text{Divergence Score} = 0.45 \times \text{WER} + 0.30 \times \text{CER} + 0.25 \times \text{OOV Ratio}$$

---

## 2. Dialectwise Extraction Summary

| Dialect Code | Region / Variety | Unique Spoken Transcripts | Top Extracted Chunk | Avg Divergence Score | Avg Normalization WER (%) | Standard OOV Ratio (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **D1** | **Malvani** *(South Konkan)* | 212 | **100** | **47.65** | **49.32%** | **78.32%** |
| **D2** | **Ahirani** *(Khandesh / North Konkan)* | 152 | **100** | **33.49** | **26.08%** | **74.05%** |
| **D4** | **Varhadi** *(Vidarbha)* | 198 | **100** | **34.27** | **28.31%** | **75.33%** |

---

## 3. Detailed Linguistic Breakdown & Top Spoken Samples

---

### 3.1 📍 D1 Malvani (South Konkan: Ratnagiri / Sindhudurg)

Malvani is characterized by nominal and verbal inflections ending in `-ो` / `-ूचे`, dative/purpose marker `खाती` (`खातिर`), and first-person pronoun `माका`.

| Rank | Score | WER (%) | OOV (%) | Original Spoken RESPIN Transcript | Normalized Standard Pune Marathi | Key Morphosyntactic Features |
| :---: | :---: | :---: | :---: | :--- | :--- | :--- |
| **1** | **103.25** | 125.0% | 100.0% | `टँकरात भरलल्या दुधाक पायपातसून सगळीकडे पोचयतत` | `टँकरात भरलेल्या दुधाकडे पाईपमधून सगळीकडे पोहोचवतात` | • Dative suffix `-क` (`दुधाक` $\rightarrow$ `दुधाला`)<br>• Ablative suffix `-तसून` (`पायपातसून` $\rightarrow$ `पाईपमधून`)<br>• Plural verb ending `-तत` (`पोचयतत` $\rightarrow$ `पोहोचवतात`) |
| **2** | **81.43** | 100.0% | 60.0% | `कर्ज कोनाकपण फेडूक मेलता काय ?` | `कर्ज कोणालाही फेडता येते का?` | • Indefinite pronoun `कोनाकपण` (`कोणालाही`)<br>• Potential verb `मेलता` (`मिळते / येते`)<br>• Infinitive `-ूक` (`फेडूक` $\rightarrow$ `फेडायला`) |
| **3** | **73.74** | 100.0% | 85.71% | `बुरशी पासून झाडांका लय प्रकारचे तरास जातत` | `बुरशीपासून झाडांना अनेक प्रकारचा त्रास होतो` | • Plural accusative `-ंका` (`झाडांका` $\rightarrow$ `झाडांना`)<br>• Quantifier `लय` (`अनेक / खूप`)<br>• Verb `जातत` (`होतात`) |
| **4** | **71.11** | 85.71% | 83.33% | `ऊसाच्या पिकाक खयच्या सिंचनाचो उपयोग करतत ?` | `उसाच्या पिकाला कोणत्या सिंचनाचा उपयोग करतात?` | • Interrogative `खयच्या` (`कोणत्या`)<br>• Genitive inflection `-चो` (`सिंचनाचो` $\rightarrow$ `सिंचनाचा`) |
| **5** | **69.60** | 100.0% | 57.14% | `हया कर्ज घेऊच्या खाती कोसाइनर लागतलो का ?` | `हे कर्ज घेण्यासाठी सह-स्वाक्षरीकर्ता लागेल का?` | • Purpose marker `खाती` (`घेऊच्या खाती` $\rightarrow$ `घेण्यासाठी`)<br>• Future verb `-तलो` (`लागतलो` $\rightarrow$ `लागेल`) |
| **6** | **67.99** | 85.71% | 75.0% | `माका क्रेडिट कार्डाक आजुन किती पैशे भरूचे हत ?` | `मला क्रेडिट कार्डसाठी अजून किती पैसे भरावे लागतील?` | • Dative pronoun `माका` (`मला`)<br>• Obligative suffix `-ूचे हत` (`भरूचे हत` $\rightarrow$ `भरावे लागतील`) |
| **7** | **64.00** | 66.67% | 100.0% | `सांड बैल ह्यो लय शक्तिमान प्राणी आसा` | `सांड बैल हा अतिशय शक्तिमान प्राणी आहे` | • Demonstrative `ह्यो` (`हा`)<br>• Copula `आसा` (`आहे`) |

---

### 3.2 📍 D2 Ahirani (Khandesh & North Konkan: Palghar / Thane / Jalgaon)

Ahirani exhibits Gujarati and Western Indo-Aryan substrata, featuring locative `-मा` / `-मझार`, universal copula `शे`, plural marker `-स्ले / -स्नं`, and verb inflection `-स`.

| Rank | Score | WER (%) | OOV (%) | Original Spoken RESPIN Transcript | Normalized Standard Pune Marathi | Key Morphosyntactic Features |
| :---: | :---: | :---: | :---: | :--- | :--- | :--- |
| **1** | **81.38** | 87.5% | 72.73% | `पी.पी.एफ मझार जमा करेल पैसा हायातीमा काढता येतसका नैत ?` | `पी.पी.एफ. मध्ये जमा केलेले पैसे आयुष्यात काढता येतात का नाहीत?` | • Inessive postposition `मझार` (`मध्ये`)<br>• Locative postposition `-मा` (`हायातीमा` $\rightarrow$ `आयुष्यात`)<br>• Negative tag `नैत` (`नाहीत`) |
| **2** | **78.21** | 83.33% | 100.0% | `कर्जलेवावर गहाण ठेयेल मालमत्ता ईकता नयी येस` | `कर्ज घेतल्यावर गहाण ठेवलेली मालमत्ता विकता येत नाही` | • Past participle `ठेयेल` (`ठेवलेली`)<br>• Verb `ईकता` (`विकता`)<br>• Negative auxiliary `नयी येस` (`येत नाही`) |
| **3** | **71.60** | 87.5% | 88.89% | `भारतमा परतेक राज्यनी वरीसनं उतपननी वाढ आल्लग आल्लग शे` | `भारतात प्रत्येक राज्याची वार्षिक उत्पन्नाची वाढ वेगवेगळी आहे` | • Locative `-मा` (`भारतमा` $\rightarrow$ `भारतात`)<br>• Genitive `-नी` (`राज्यनी` $\rightarrow$ `राज्याची`)<br>• Adverb `आल्लग आल्लग` (`वेगवेगळी`)<br>• Copula `शे` (`आहे`) |
| **4** | **69.28** | 75.0% | 100.0% | `शेतीना अवजारेस्नं परदरशन मांडं तवय त्यास्नी ईक्री वाढी` | `शेतीची अवजारे प्रदर्शन मांडल्यावर त्यांची विक्री वाढली` | • Plural genitive `-स्नं / -स्नी` (`अवजारेस्नं` $\rightarrow$ `अवजारांचे`, `त्यास्नी` $\rightarrow$ `त्यांची`)<br>• Temporal conjunction `तवय` (`तेव्हा`)<br>• Lexical `ईक्री` (`विक्री`) |
| **5** | **68.48** | 75.0% | 77.78% | `फळे देनारा झाडेस्ले शेतकरी जमीनना आंदाज लीसनी काटछाट करस` | `फळे देणाऱ्या झाडांना शेतकरी जमिनीचा अंदाज घेऊन छाटणी करतो` | • Dative plural `-स्ले` (`झाडेस्ले` $\rightarrow$ `झाडांना`)<br>• Conjunctive participle `लीसनी` (`घेऊन`)<br>• Present tense verb `-स` (`करस` $\rightarrow$ `करतो`) |
| **6** | **67.51** | 75.0% | 88.89% | `भोपयाना भाजीना वापर रायता लोणचं खीर बनाडाना करता करतंस` | `भोपळ्याच्या भाजीचा वापर रायता लोणचे खीर बनवण्यासाठी करतात` | • Phonological cluster `भोपयाना` (`भोपळ्याच्या`)<br>• Infinitive `बनाडाना करता` (`बनवण्यासाठी`)<br>• Habitual verb `करतंस` (`करतात`) |

---

### 3.3 📍 D4 Varhadi (Vidarbha: Amravati / Akola / Yavatmal)

Varhadi features distinctive dative pronoun `मले`, interrogatives `कोंते / कोनचा`, retroflex softening ($ळ \rightarrow ल$, $ण \rightarrow न$), and verb endings in `-न / -ीन`.

| Rank | Score | WER (%) | OOV (%) | Original Spoken RESPIN Transcript | Normalized Standard Pune Marathi | Key Morphosyntactic Features |
| :---: | :---: | :---: | :---: | :--- | :--- | :--- |
| **1** | **63.93** | 75.0% | 72.73% | `मले माया बईनीले मोबाईल भेट द्यासाठी ऑनलाईन पैसे कसे देता येईन ?` | `मला माझ्या बहिणीला मोबाईल भेट देण्यासाठी ऑनलाईन पैसे कसे देता येतील?` | • Dative pronoun `मले` (`मला`)<br>• Possessive `माया` (`माझ्या`)<br>• Kinship term `बईनीले` (`बहिणीला`)<br>• Future verb `येईन` (`येतील`) |
| **2** | **59.72** | 71.43% | 85.71% | `खातं काढताना मले किती पैसा भरा लागन ?` | `खाते उघडताना मला किती पैसे भरावे लागतील?` | • Pronoun `मले` (`मला`)<br>• Future auxiliary `लागन` (`लागतील`)<br>• Lexical `काढताना` (`उघडताना`) |
| **3** | **59.24** | 75.0% | 66.67% | `क्रेडिट कार्ड घ्यासाठी मले कोनचा अर्ज भरनं सोप राहीन ?` | `क्रेडिट कार्ड घेण्यासाठी मला कोणता अर्ज भरणे सोपे राहील?` | • Interrogative `कोनचा` (`कोणता`)<br>• Infinitive `घ्यासाठी` (`घेण्यासाठी`)<br>• Future auxiliary `राहीन` (`राहील`) |
| **4** | **57.48** | 71.43% | 87.5% | `कास्तकार मोबाईल वरून शेतीचं मार्गदर्शन कस भेटवू शकते ?` | `शेतकरी मोबाईलवरून शेतीचे मार्गदर्शन कसे मिळवू शकतो?` | • Lexical noun `कास्तकार` (`शेतकरी`)<br>• Potential verb `भेटवू शकते` (`मिळवू शकतो`) |
| **5** | **55.14** | 60.0% | 66.67% | `कर्ज घ्याचे टायमाले साक्षीदार लागन का ?` | `कर्ज घेण्याच्या वेळी साक्षीदार लागेल का?` | • Temporal loan-blend `टायमाले` (`वेळी / वेळेस`)<br>• Future verb `लागन` (`लागेल`) |
| **6** | **53.33** | 66.67% | 66.67% | `दसऱ्यासाठी मले कितीक रुपये कर्ज भेटन ?` | `दसऱ्यासाठी मला किती रुपये कर्ज मिळेल?` | • Quantifier `कितीक` (`किती`)<br>• Verb `भेटन` (`मिळेल`) |

---

## 4. Exported Clean Datasets for Manual Verification

The extracted top divergent spoken transcripts are saved in clean UTF-8 CSV format under `reports/respin_divergent_samples/`:

* 📄 **D1 Malvani Top Spoken CSV**: [reports/respin_divergent_samples/respin_d1_top_divergent.csv](file:///d:/dialect-norm/reports/respin_divergent_samples/respin_d1_top_divergent.csv)
* 📄 **D2 Ahirani Top Spoken CSV**: [reports/respin_divergent_samples/respin_d2_top_divergent.csv](file:///d:/dialect-norm/reports/respin_divergent_samples/respin_d2_top_divergent.csv)
* 📄 **D4 Varhadi Top Spoken CSV**: [reports/respin_divergent_samples/respin_d4_top_divergent.csv](file:///d:/dialect-norm/reports/respin_divergent_samples/respin_d4_top_divergent.csv)
* 📄 **Summary Digest**: [reports/respin_divergent_samples/respin_divergence_summary.md](file:///d:/dialect-norm/reports/respin_divergent_samples/respin_divergence_summary.md)
