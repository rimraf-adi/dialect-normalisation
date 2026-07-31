"""
Module to split synthetic parallel data into clean dialect partitions (d1, d2, d4) and flawed rows.
"""

import csv
import re
import sys
from pathlib import Path

RESIDUE_PATTERNS = {
    "Malvani (D1) Residue in Standard": [
        r"\bमाका\b", r"\bतुका\b", r"\bह्यो\b", r"\bआसा\b", r"\bआसत\b", r"\bआसतत\b",
        r"\bलागात\b", r"\bलागातला\b", r"\bकरूक\b", r"\bजावूक\b", r"\bदेऊक\b", r"\bघेऊक\b",
        r"\bबघूक\b", r"\bभरूक\b", r"\bव्हया\b", r"\bहोया\b", r"\bकरूचा\b", r"\bफेडूचा\b",
        r"\bकाडूचा\b", r"\bभरूचा\b", r"\bखय\b", r"\bखयच्या\b", r"\bखयल्या\b", r"\bखयली\b",
        r"\bइलो\b", r"\bमगे\b", r"\bकितक्या\b", r"\bकितको\b", r"\bइतक्या\b"
    ],
    "Ahirani (D2) Residue in Standard": [
        r"\bशेतस\b", r"\bगनज\b", r"\bकरस\b", r"\bकरास\b", r"\bगावता\b", r"\bकाढुक\b",
        r"\bपोरीस्नी\b", r"\bलोकेस्ना\b", r"\bमना\b", r"\bतुना\b", r"\bक्यारे\b", r"\bकठाडे\b"
    ],
    "Varhadi (D4) Residue in Standard": [
        r"\bमाथून\b", r"\bमले\b", r"\bतुले\b", r"\bआम्हाले\b", r"\bशिकायले\b", r"\bव्हाता\b",
        r"\bपाह्यजे\b", r"\bकहाले\b", r"\bनव्हता\b"
    ]
}

HINDI_LEAKAGE_PATTERNS = [
    r"\bकी विधि से\b", r"\bहोती है\b", r"\bहोता है\b", r"\bकिया जाता है\b",
    r"\bऔर\b", r"\bसे पानी\b", r"\bकी बचत\b", r"\bके लिए\b", r"\bसबसे\b"
]

def detect_flaws(orig: str, trans: str) -> str:
    flaws = []

    if re.search(r"```|^Output:|^Standard Output:|^\d+[\.\)]", trans, re.IGNORECASE):
        flaws.append("Formatting leakage")

    for h_pat in HINDI_LEAKAGE_PATTERNS:
        if re.search(h_pat, trans):
            flaws.append(f"Hindi language leakage: '{h_pat}'")
            break

    for cat, patterns in RESIDUE_PATTERNS.items():
        for pat in patterns:
            if pat == r"\bआसा\b" and ("आसाम" in trans or "आभास" in trans):
                continue
            if re.search(pat, trans):
                flaws.append(f"Dialect residue in standard text ({cat}): {pat}")
                break

    if "?" in orig and "?" not in trans:
        flaws.append("Question mark dropped in translation")
    elif "?" not in orig and "?" in trans and "का" not in orig:
        flaws.append("Unprompted question mark added")

    orig_w = len(orig.split())
    trans_w = len(trans.split())
    if orig_w > 3:
        ratio = trans_w / orig_w
        if ratio < 0.45 or ratio > 2.2:
            flaws.append(f"Word length anomaly ({orig_w} vs {trans_w} words)")

    if re.search(r"हा पदार्थ आहे$|करणार आहे हा|सांगत आहे\.$", trans):
        flaws.append("Garbled/Incoherent translation structure")

    return "; ".join(flaws) if flaws else ""

def split_and_clean_dataset(data_dir: Path = Path("data/synthetic_parallel")):
    csv_files = sorted(data_dir.glob("marathi_parallel_part_*.csv"))
    if not csv_files:
        print(f"No input CSV files found in {data_dir}!")
        return

    d1_rows, d2_rows, d4_rows, flawed_rows = [], [], [], []
    total_processed = 0

    for csv_file in csv_files:
        with open(csv_file, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                total_processed += 1
                dial = row.get("dialect", "").upper().strip()
                orig = row.get("dialect_text", "").strip()
                trans = row.get("standard_text", "").strip()

                flaw_reason = detect_flaws(orig, trans)
                if flaw_reason:
                    flawed_row = dict(row)
                    flawed_row["flaw_reason"] = flaw_reason
                    flawed_rows.append(flawed_row)
                else:
                    if dial == "D1":
                        d1_rows.append(row)
                    elif dial == "D2":
                        d2_rows.append(row)
                    elif dial == "D4":
                        d4_rows.append(row)
                    else:
                        d1_rows.append(row)

    fieldnames = ["id", "text_id", "dialect", "domain", "distortion_score", "dialect_text", "standard_text"]

    with open(data_dir / "d1.csv", "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(d1_rows)

    with open(data_dir / "d2.csv", "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(d2_rows)

    with open(data_dir / "d4.csv", "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(d4_rows)

    with open(data_dir / "flawed.csv", "w", encoding="utf-8-sig", newline="") as f:
        flawed_fieldnames = fieldnames + ["flaw_reason"]
        writer = csv.DictWriter(f, fieldnames=flawed_fieldnames)
        writer.writeheader()
        writer.writerows(flawed_rows)

    print("=" * 70)
    print("DATASET SEPARATION AND REARRANGEMENT COMPLETED")
    print("=" * 70)
    print(f"Total Sentences Processed : {total_processed:,}")
    print(f"Clean D1 Sentences (d1.csv): {len(d1_rows):,}")
    print(f"Clean D2 Sentences (d2.csv): {len(d2_rows):,}")
    print(f"Clean D4 Sentences (d4.csv): {len(d4_rows):,}")
    print(f"Flawed Sentences (flawed.csv): {len(flawed_rows):,}")
    print("=" * 70)

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    split_and_clean_dataset()

if __name__ == "__main__":
    main()
