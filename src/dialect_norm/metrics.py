import re
import jiwer


def normalize_text(text: str) -> str:
    """Normalizes Indic text by removing punctuation and collapsing whitespace."""
    if not text:
        return ""
    punct_pattern = r'[।॥?!.,;:"\'\-\(\)\[\]\{\}—«»“”‘’]'
    text = re.sub(punct_pattern, " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_metrics_for_set(references_raw, hypotheses_raw, references_norm, hypotheses_norm) -> dict:
    """Computes WER, CER, SER, and edit distance breakdowns for raw and normalized text."""
    if not references_raw:
        return {}

    raw_wer = jiwer.wer(references_raw, hypotheses_raw)
    raw_cer = jiwer.cer(references_raw, hypotheses_raw)
    raw_word_output = jiwer.process_words(references_raw, hypotheses_raw)
    raw_ser = sum(r != h for r, h in zip(references_raw, hypotheses_raw)) / len(references_raw)

    norm_wer = jiwer.wer(references_norm, hypotheses_norm)
    norm_cer = jiwer.cer(references_norm, hypotheses_norm)
    norm_word_output = jiwer.process_words(references_norm, hypotheses_norm)
    norm_ser = sum(r != h for r, h in zip(references_norm, hypotheses_norm)) / len(references_norm)

    return {
        "num_samples": len(references_raw),
        "raw": {
            "wer": float(round(raw_wer, 4)),
            "wer_percentage": float(round(raw_wer * 100, 2)),
            "cer": float(round(raw_cer, 4)),
            "cer_percentage": float(round(raw_cer * 100, 2)),
            "ser": float(round(raw_ser, 4)),
            "ser_percentage": float(round(raw_ser * 100, 2)),
            "exact_match_acc_percentage": float(round((1.0 - raw_ser) * 100, 2)),
            "substitutions": int(raw_word_output.substitutions),
            "deletions": int(raw_word_output.deletions),
            "insertions": int(raw_word_output.insertions),
            "hits": int(raw_word_output.hits),
            "total_ref_words": int(sum(len(ref.split()) for ref in references_raw)),
        },
        "normalized": {
            "wer": float(round(norm_wer, 4)),
            "wer_percentage": float(round(norm_wer * 100, 2)),
            "cer": float(round(norm_cer, 4)),
            "cer_percentage": float(round(norm_cer * 100, 2)),
            "ser": float(round(norm_ser, 4)),
            "ser_percentage": float(round(norm_ser * 100, 2)),
            "exact_match_acc_percentage": float(round((1.0 - norm_ser) * 100, 2)),
            "substitutions": int(norm_word_output.substitutions),
            "deletions": int(norm_word_output.deletions),
            "insertions": int(norm_word_output.insertions),
            "hits": int(norm_word_output.hits),
            "total_ref_words": int(sum(len(ref.split()) for ref in references_norm)),
        },
    }


def compute_group_metrics(samples: list, all_ref_raw: list, all_hyp_raw: list, all_ref_norm: list, all_hyp_norm: list, group_key: str) -> dict:
    """Groups samples by group_key and computes set metrics per group."""
    groups = {}
    for idx, sample in enumerate(samples):
        val = sample.get(group_key, "Unknown")
        if val not in groups:
            groups[val] = {"ref_raw": [], "hyp_raw": [], "ref_norm": [], "hyp_norm": [], "duration": 0.0}
        groups[val]["ref_raw"].append(all_ref_raw[idx])
        groups[val]["hyp_raw"].append(all_hyp_raw[idx])
        groups[val]["ref_norm"].append(all_ref_norm[idx])
        groups[val]["hyp_norm"].append(all_hyp_norm[idx])
        groups[val]["duration"] += sample.get("duration", 0.0)

    result_group = {}
    for name, data in sorted(groups.items()):
        res = compute_metrics_for_set(data["ref_raw"], data["hyp_raw"], data["ref_norm"], data["hyp_norm"])
        res["total_duration_hours"] = float(round(data["duration"] / 3600.0, 3))
        result_group[str(name)] = res
    return result_group


def compute_composite_group_metrics(samples: list, all_ref_raw: list, all_hyp_raw: list, all_ref_norm: list, all_hyp_norm: list, key1: str, key2: str) -> dict:
    """Groups samples by composite key (key1_key2) and computes set metrics."""
    groups = {}
    for idx, sample in enumerate(samples):
        val1 = sample.get(key1, "Unknown")
        val2 = sample.get(key2, "Unknown")
        comp_key = f"{val1}_{val2}"
        if comp_key not in groups:
            groups[comp_key] = {"ref_raw": [], "hyp_raw": [], "ref_norm": [], "hyp_norm": [], "duration": 0.0}
        groups[comp_key]["ref_raw"].append(all_ref_raw[idx])
        groups[comp_key]["hyp_raw"].append(all_hyp_raw[idx])
        groups[comp_key]["ref_norm"].append(all_ref_norm[idx])
        groups[comp_key]["hyp_norm"].append(all_hyp_norm[idx])
        groups[comp_key]["duration"] += sample.get("duration", 0.0)

    result_group = {}
    for name, data in sorted(groups.items()):
        res = compute_metrics_for_set(data["ref_raw"], data["hyp_raw"], data["ref_norm"], data["hyp_norm"])
        res["total_duration_hours"] = float(round(data["duration"] / 3600.0, 3))
        result_group[str(name)] = res
    return result_group


def compute_samplewise_analytics(samples: list, all_ref_raw: list, all_hyp_raw: list, all_ref_norm: list, all_hyp_norm: list) -> list:
    """Generates detailed itemized metrics for every evaluated utterance."""
    samplewise_analytics = []
    for idx, s in enumerate(samples):
        r_raw = all_ref_raw[idx]
        h_raw = all_hyp_raw[idx]
        r_norm = all_ref_norm[idx]
        h_norm = all_hyp_norm[idx]

        r_wer = float(round(jiwer.wer(r_raw, h_raw), 4)) if r_raw else 0.0
        n_wer = float(round(jiwer.wer(r_norm, h_norm), 4)) if r_norm else 0.0
        r_cer = float(round(jiwer.cer(r_raw, h_raw), 4)) if r_raw else 0.0
        n_cer = float(round(jiwer.cer(r_norm, h_norm), 4)) if r_norm else 0.0

        samplewise_analytics.append({
            "utterance_id": s.get("utterance_id", s.get("utt_key")),
            "dialect": s.get("dialect"),
            "domain": s.get("domain"),
            "gender": s.get("gender"),
            "slab": s.get("slab"),
            "age_group": s.get("age_group"),
            "speaker_id": s.get("speaker_id"),
            "duration_seconds": s.get("duration"),
            "reference_raw": r_raw,
            "hypothesis_raw": h_raw,
            "reference_normalized": r_norm,
            "hypothesis_normalized": h_norm,
            "raw_wer": r_wer,
            "normalized_wer": n_wer,
            "raw_cer": r_cer,
            "normalized_cer": n_cer,
            "exact_match": (r_norm == h_norm),
        })
    return samplewise_analytics
