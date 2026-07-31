"""
Advanced fine-grained linguistic filter module for dialect datasets.
"""

import csv
import re
import sys
from pathlib import Path

def is_flawed(dialect: str, dialect_text: str, standard_text: str) -> tuple[bool, str]:
    reasons = []

    hindi_words = [r"\bकी\b", r"\bऔर\b", r"\bहोती\b", r"\bहै\b", r"\bसिंचाई\b", r"\bखाद\b", r"\bबचत\b", r"\bसे\b"]
    hindi_matches = [w for w in hindi_words if re.search(w, standard_text)]
    if len(hindi_matches) >= 2 or re.search(r"की विधि से|होती है|होता है|किया जाता है|और", standard_text):
        reasons.append(f"Wrong language / Hindi leakage: {', '.join(hindi_matches)}")

    d1_markers = [
        r"\bमाका\b", r"\bतुका\b", r"\bह्यो\b", r"\bह्या\b", r"\bआसा\b", r"\bआसत\b", r"\bआसतत\b",
        r"\bलागात\b", r"\bलागातला\b", r"\bकरूक\b", r"\bजावूक\b", r"\bजावक\b", r"\bदेऊक\b", r"\bघेऊक\b",
        r"\bबघूक\b", r"\bभरूक\b", r"\bव्हया\b", r"\bहोया\b", r"\bकरूचा\b", r"\bफेडूचा\b",
        r"\bकाडूचा\b", r"\bभरूचा\b", r"\bखय\b", r"\bखयच्या\b", r"\bखयल्या\b", r"\bखयली\b",
        r"\bइलो\b", r"\bमगे\b", r"\bकितक्या\b", r"\bकितको\b", r"\bइतक्या\b", r"\bअसो\b", r"\bतसो\b"
    ]
    d2_markers = [r"\bशेतस\b", r"\bगनज\b", r"\bकरस\b", r"\bकरास\b", r"\bगावता\b", r"\bकाढुक\b", r"\bपोरीस्नी\b", r"\bलोकेस्ना\b", r"\bमना\b", r"\bतुना\b"]
    d4_markers = [r"\bमाथून\b", r"\bमले\b", r"\bतुले\b", r"\bआम्हाले\b", r"\bशिकायले\b", r"\bव्हाता\b", r"\bपाह्यजे\b", r"\bकहाले\b"]

    all_residues = d1_markers + d2_markers + d4_markers
    found_residue = []
    for marker in all_residues:
        if marker == r"\bआसा\b" and ("आसाम" in standard_text or "आभास" in standard_text or "आकार" in standard_text):
            continue
        if re.search(marker, standard_text):
            found_residue.append(marker.replace(r"\b", ""))

    if found_residue:
        reasons.append(f"Standard text retains dialect residue: {', '.join(found_residue)}")

    garbled_patterns = [
        r"हा पदार्थ आहे", r"सांगत आहे\.$", r"करणार आहे हा", r"किती दिसतात\?",
        r"मधुमेह", r"बटाटे"
    ]
    for g_pat in garbled_patterns:
        if g_pat == r"मधुमेह" and "डायबॅक" in dialect_text:
            reasons.append("Garbled translation: 'डायबॅक' misinterpreted as 'मधुमेह'")
        elif g_pat == r"बटाटे" and "कांदो" in dialect_text:
            reasons.append("Garbled translation: 'कांदो' (onion) mistranslated as 'बटाटे' (potato)")
        elif g_pat == r"किती दिसतात\?" and "पोचाक" in dialect_text:
            reasons.append("Garbled translation: 'पोचाक' (delivery/reaching) mistranslated as 'दिसतात' (visible)")
        elif re.search(g_pat, standard_text):
            reasons.append(f"Garbled/Incoherent structure: '{g_pat}'")

    if "?" in dialect_text and "?" not in standard_text:
        reasons.append("Question mark dropped in translation")
    elif "?" in dialect_text and "?" in standard_text:
        if not re.search(r"का|काय|कसे|कसा|कशी|कुठे|कोणी|केव्हा|किती|कोणत्या|कोणता|कोणती|खाय", standard_text):
            reasons.append("Question syntax lost in standard text")

    if reasons:
        return True, " | ".join(reasons)
    return False, ""

def filter_flawed_d1(data_dir: Path = Path("data/synthetic_parallel")):
    csv_files = sorted(data_dir.glob("marathi_parallel_part_*.csv"))
    
    clean_d1, clean_d2, clean_d4, flawed_list = [], [], [], []
    fieldnames = ["id", "text_id", "dialect", "domain", "distortion_score", "dialect_text", "standard_text"]

    for csv_file in csv_files:
        with open(csv_file, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                dial = row.get("dialect", "").strip().upper()
                d_text = row.get("dialect_text", "").strip()
                s_text = row.get("standard_text", "").strip()

                flawed, reason = is_flawed(dial, d_text, s_text)
                if flawed:
                    flawed_row = dict(row)
                    flawed_row["flaw_reason"] = reason
                    flawed_list.append(flawed_row)
                else:
                    if dial == "D1":
                        clean_d1.append(row)
                    elif dial == "D2":
                        clean_d2.append(row)
                    elif dial == "D4":
                        clean_d4.append(row)
                    else:
                        clean_d1.append(row)

    with open(data_dir / "d1.csv", "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(clean_d1)

    with open(data_dir / "d2.csv", "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(clean_d2)

    with open(data_dir / "d4.csv", "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(clean_d4)

    flawed_fieldnames = fieldnames + ["flaw_reason"]
    with open(data_dir / "flawed.csv", "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=flawed_fieldnames)
        writer.writeheader()
        writer.writerows(flawed_list)

    print("=" * 60)
    print("FINISHED DATASET SEGREGATION AND REARRANGEMENT")
    print("=" * 60)
    print(f"Clean D1 Rows (d1.csv) : {len(clean_d1):,}")
    print(f"Clean D2 Rows (d2.csv) : {len(clean_d2):,}")
    print(f"Clean D4 Rows (d4.csv) : {len(clean_d4):,}")
    print(f"Flawed Rows (flawed.csv): {len(flawed_list):,}")
    print("=" * 60)

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    filter_flawed_d1()

if __name__ == "__main__":
    main()
