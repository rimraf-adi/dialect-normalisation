r"""
CSV Audit Script for synthetic parallel Marathi dataset.
Checks for missing values, structural issues, unparsed LLM output, and quality anomalies.
"""
import csv
import re
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def audit_csv(csv_path: Path):
    if not csv_path.exists():
        print(f"File not found: {csv_path}")
        return

    print("=" * 70)
    print(f"AUDITING CSV FILE: {csv_path.name}")
    print("=" * 70)

    rows = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        print(f"Header Fieldnames: {fieldnames}")
        for i, r in enumerate(reader, 1):
            rows.append((i, r))

    print(f"Total Rows Read: {len(rows):,}")

    expected_fields = ["id", "text_id", "dialect", "domain", "distortion_score", "dialect_text", "standard_text"]
    header_ok = set(expected_fields).issubset(set(fieldnames or []))
    print(f"Header Integrity: {'✅ VALID' if header_ok else '❌ INVALID'}")

    # Audit checks
    empty_standard = []
    empty_dialect = []
    unparsed_json = []
    thought_tags = []
    markdown_fences = []
    leading_prefixes = []
    identical_text = []
    length_anomalies = []
    dialect_counts = {}

    for row_num, row in rows:
        d_text = (row.get("dialect_text") or "").strip()
        s_text = (row.get("standard_text") or "").strip()
        dialect = row.get("dialect") or "Unknown"

        dialect_counts[dialect] = dialect_counts.get(dialect, 0) + 1

        if not s_text:
            empty_standard.append(row_num)
        if not d_text:
            empty_dialect.append(row_num)

        # Check for unparsed raw LLM structures
        if s_text.startswith("[") or s_text.endswith("]") or s_text.startswith("{") or s_text.endswith("}"):
            unparsed_json.append((row_num, s_text))

        if "<thought>" in s_text or "<think>" in s_text or "</thought>" in s_text:
            thought_tags.append((row_num, s_text))

        if "```" in s_text:
            markdown_fences.append((row_num, s_text))

        if re.search(r"^(?:Standard Output|Output|Standard Pune|Standard Marathi)\s*[:\-]", s_text, re.I):
            leading_prefixes.append((row_num, s_text))

        # Check for identical dialect/standard text (meaning no translation occurred)
        if d_text and s_text and d_text == s_text:
            identical_text.append(row_num)

        # Check for extreme length ratio discrepancies
        if d_text and s_text:
            ratio = len(s_text) / len(d_text)
            if ratio < 0.3 or ratio > 3.0:
                length_anomalies.append((row_num, len(d_text), len(s_text), d_text[:40], s_text[:40]))

    print("\n--- SUMMARY OF AUDIT RESULTS ---")
    print(f"Dialect Breakdown       : {dialect_counts}")
    print(f"Empty Standard Text     : {len(empty_standard)} rows {empty_standard[:5] if empty_standard else '✅ Clean'}")
    print(f"Empty Dialect Text      : {len(empty_dialect)} rows {empty_dialect[:5] if empty_dialect else '✅ Clean'}")
    print(f"Unparsed JSON Fences [] : {len(unparsed_json)} rows {unparsed_json[:3] if unparsed_json else '✅ Clean'}")
    print(f"Unstripped <thought>    : {len(thought_tags)} rows {thought_tags[:3] if thought_tags else '✅ Clean'}")
    print(f"Unstripped Markdown ``` : {len(markdown_fences)} rows {markdown_fences[:3] if markdown_fences else '✅ Clean'}")
    print(f"Unstripped Prefixes     : {len(leading_prefixes)} rows {leading_prefixes[:3] if leading_prefixes else '✅ Clean'}")
    print(f"Identical Text Pairings : {len(identical_text)} rows (Dialect text equal to Standard)")
    print(f"Extreme Length Anomaly  : {len(length_anomalies)} rows")

    if length_anomalies:
        print("\nSample Length Anomalies:")
        for r_num, l_d, l_s, d_snippet, s_snippet in length_anomalies[:5]:
            print(f"  Row {r_num}: dialect len={l_d}, standard len={l_s}")
            print(f"     Dialect : {d_snippet}...")
            print(f"     Standard: {s_snippet}...")

    # Print 3 random sample pairs to visually verify standard Marathi quality
    print("\n--- RANDOM SAMPLE DATASET PAIRS FOR VISUAL QUALITY CHECK ---")
    sample_indices = [1, len(rows) // 2, len(rows)]
    for idx in sample_indices:
        if 1 <= idx <= len(rows):
            r_num, r = rows[idx - 1]
            print(f"\nRow #{r_num} [{r.get('dialect')} | Distortion: {r.get('distortion_score')}]:")
            print(f"  Dialect  : {r.get('dialect_text')}")
            print(f"  Standard : {r.get('standard_text')}")

    total_errors = len(empty_standard) + len(empty_dialect) + len(unparsed_json) + len(thought_tags) + len(markdown_fences) + len(leading_prefixes)
    print("\n" + "=" * 70)
    if total_errors == 0:
        print("✅ AUDIT COMPLETE: ZERO ERRORS FOUND! CSV FILE IS CLEAN & READY.")
    else:
        print(f"⚠️ AUDIT COMPLETE: {total_errors} ISSUES FOUND. SEE BREAKDOWN ABOVE.")
    print("=" * 70)

if __name__ == "__main__":
    csv_file = Path("data/synthetic_parallel/marathi_parallel_part_001.csv")
    audit_csv(csv_file)
