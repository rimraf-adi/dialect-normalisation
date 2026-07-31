"""
Module to compute comprehensive dataset summary statistics for IISc RESPIN Marathi Train Set
and export the results to stats_mr.yaml.
"""

import json
import sys
import time
from pathlib import Path
import yaml


def generate_stats_yaml(meta_path: Path, output_path: Path):
    print("=" * 70)
    print("GENERATING STATS FOR IISc RESPIN MARATHI TRAIN SET (CLEAN)")
    print("=" * 70)
    print(f"Loading metadata file: {meta_path}...")
    
    start_time = time.time()
    with open(meta_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    load_time = time.time() - start_time
    print(f"Loaded metadata in {load_time:.2f} seconds.")

    entries = list(data.values()) if isinstance(data, dict) else data
    total_utts = len(entries)
    print(f"Total utterances to process: {total_utts:,}")

    total_duration_sec = 0.0
    all_texts = set()
    all_text_ids = set()
    all_speakers = set()
    all_pincodes = set()

    dialect_stats = {}
    domain_stats = {}
    gender_stats = {}
    age_stats = {}
    slab_stats = {}

    for e in entries:
        dur = float(e.get("duration", 0.0))
        txt = e.get("text", "").strip()
        tid = e.get("text_id", "")
        spk = e.get("speaker_id", "")
        pin = e.get("pincode", "")
        dial = e.get("dialect", "Unknown")
        dom = e.get("domain", "Unknown")
        gen = e.get("gender", "Unknown")
        age = e.get("age_group", "Unknown")
        slab = e.get("slab", "Unknown")

        total_duration_sec += dur
        if txt:
            all_texts.add(txt)
        if tid:
            all_text_ids.add(tid)
        if spk:
            all_speakers.add(spk)
        if pin:
            all_pincodes.add(pin)

        if dial not in dialect_stats:
            dialect_stats[dial] = {
                "utterances": 0,
                "duration_seconds": 0.0,
                "texts": set(),
                "text_ids": set(),
                "speakers": set(),
                "domains": {},
                "genders": {},
                "slabs": {},
            }
        d_ref = dialect_stats[dial]
        d_ref["utterances"] += 1
        d_ref["duration_seconds"] += dur
        if txt:
            d_ref["texts"].add(txt)
        if tid:
            d_ref["text_ids"].add(tid)
        if spk:
            d_ref["speakers"].add(spk)
        d_ref["domains"][dom] = d_ref["domains"].get(dom, 0) + 1
        d_ref["genders"][gen] = d_ref["genders"].get(gen, 0) + 1
        d_ref["slabs"][slab] = d_ref["slabs"].get(slab, 0) + 1

        if dom not in domain_stats:
            domain_stats[dom] = {
                "utterances": 0,
                "duration_seconds": 0.0,
                "texts": set(),
                "speakers": set(),
            }
        dom_ref = domain_stats[dom]
        dom_ref["utterances"] += 1
        dom_ref["duration_seconds"] += dur
        if txt:
            dom_ref["texts"].add(txt)
        if spk:
            dom_ref["speakers"].add(spk)

        gender_stats[gen] = gender_stats.get(gen, 0) + 1
        age_stats[age] = age_stats.get(age, 0) + 1
        slab_stats[slab] = slab_stats.get(slab, 0) + 1

    total_duration_hours = round(total_duration_sec / 3600.0, 3)
    avg_duration_sec = round(total_duration_sec / total_utts, 2) if total_utts > 0 else 0.0

    yaml_data = {
        "dataset_name": "IISc RESPIN Marathi Train Set (Clean)",
        "language_name": "Marathi",
        "language_code": "mr",
        "metadata_file": str(meta_path.as_posix()),
        "overall_summary": {
            "total_utterances": total_utts,
            "total_duration_seconds": round(total_duration_sec, 2),
            "total_duration_hours": total_duration_hours,
            "avg_duration_seconds": avg_duration_sec,
            "total_unique_reference_texts": len(all_texts),
            "total_unique_prompt_ids": len(all_text_ids),
            "total_unique_speakers": len(all_speakers),
            "total_unique_pincodes": len(all_pincodes),
        },
        "dialect_breakdown": {},
        "domain_breakdown": {},
        "demographics_breakdown": {
            "gender_distribution": gender_stats,
            "age_group_distribution": age_stats,
            "slab_distribution": slab_stats,
        },
    }

    for dial in sorted(dialect_stats.keys()):
        d_data = dialect_stats[dial]
        d_dur_hrs = round(d_data["duration_seconds"] / 3600.0, 3)
        d_pct_utts = round((d_data["utterances"] / total_utts) * 100.0, 2)
        
        yaml_data["dialect_breakdown"][dial] = {
            "utterances": d_data["utterances"],
            "percentage_of_total_utts": d_pct_utts,
            "total_duration_hours": d_dur_hrs,
            "avg_duration_seconds": round(d_data["duration_seconds"] / d_data["utterances"], 2),
            "unique_reference_texts": len(d_data["texts"]),
            "unique_prompt_ids": len(d_data["text_ids"]),
            "unique_speakers": len(d_data["speakers"]),
            "domain_distribution": d_data["domains"],
            "gender_distribution": d_data["genders"],
            "slab_distribution": d_data["slabs"],
        }

    for dom in sorted(domain_stats.keys()):
        dom_data = domain_stats[dom]
        dom_dur_hrs = round(dom_data["duration_seconds"] / 3600.0, 3)
        dom_pct_utts = round((dom_data["utterances"] / total_utts) * 100.0, 2)

        yaml_data["domain_breakdown"][dom] = {
            "utterances": dom_data["utterances"],
            "percentage_of_total_utts": dom_pct_utts,
            "total_duration_hours": dom_dur_hrs,
            "unique_reference_texts": len(dom_data["texts"]),
            "unique_speakers": len(dom_data["speakers"]),
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_data, f, allow_unicode=True, sort_keys=False, default_flow_style=False)

    print("\n" + "=" * 70)
    print(f"✅ STATS SUCCESSFULLY GENERATED AND SAVED TO: {output_path.resolve()}")
    print("=" * 70)

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    meta_path1 = Path("IISc_RESPIN_train_mr_clean/IISc_RESPIN_train_mr_clean/meta_train_mr_clean.json")
    meta_path2 = Path("IISc_RESPIN_train_mr_clean/meta_train_mr_clean.json")

    if meta_path1.exists():
        meta_path = meta_path1
    elif meta_path2.exists():
        meta_path = meta_path2
    else:
        print(f"Error: Could not find meta_train_mr_clean.json at {meta_path1} or {meta_path2}", file=sys.stderr)
        sys.exit(1)

    out_file = Path("stats_mr.yaml")
    generate_stats_yaml(meta_path, out_file)

if __name__ == "__main__":
    main()
