"""
Exhaustive evaluation script to evaluate ALL models and ALL 5 folds across all dialect partitions
(D1 Malvani, D2 Ahirani, D3 Standard, D4 Varhadi, and Combined) on the official IISc_RESPIN_test_mr test set
with full verbose dialectwise logs per fold.
"""

import argparse
import csv
import json
import logging
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import evaluate
import jiwer
import torch
import yaml
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, MBartForConditionalGeneration

from dialect_norm.metrics import normalize_text

logger = logging.getLogger("dialect_norm.fold_eval")


def setup_logger():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def compute_mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    if len(values) == 1:
        return round(mean, 2), 0.0
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return round(mean, 2), round(math.sqrt(variance), 2)


def load_respin_test_data() -> Dict[str, List[Dict]]:
    p1 = Path("IISc_RESPIN_test_mr/IISc_RESPIN_test_mr/meta_test_mr.json")
    p2 = Path("IISc_RESPIN_test_mr/meta_test_mr.json")
    meta_path = p1 if p1.exists() else p2

    if not meta_path.exists():
        raise FileNotFoundError(f"Cannot locate RESPIN test metadata at {p1} or {p2}")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    groups = {"D1": [], "D2": [], "D3": [], "D4": []}
    for item in meta.values():
        d_code = item.get("dialect", "").upper()
        if d_code in groups:
            groups[d_code].append(item)

    logger.info(f"Loaded RESPIN Test Set: D1 ({len(groups['D1'])}), D2 ({len(groups['D2'])}), D3 ({len(groups['D3'])}), D4 ({len(groups['D4'])}) => Total {sum(len(v) for v in groups.values())} utterances")
    return groups


def compute_metrics(predictions: List[str], references: List[str]) -> Dict:
    sacrebleu = evaluate.load("sacrebleu")
    chrf = evaluate.load("chrf")

    norm_preds = [normalize_text(p) for p in predictions]
    norm_refs = [normalize_text(r) for r in references]

    refs_nested = [[r] for r in norm_refs]
    bleu_res = sacrebleu.compute(predictions=norm_preds, references=refs_nested)
    chrf_res = chrf.compute(predictions=norm_preds, references=refs_nested, word_order=2)

    wer_val = float(jiwer.wer(norm_refs, norm_preds)) * 100.0
    cer_val = float(jiwer.cer(norm_refs, norm_preds)) * 100.0

    exact_matches = sum(1 for p, r in zip(norm_preds, norm_refs) if p == r)
    exact_acc = round((exact_matches / max(1, len(norm_refs))) * 100.0, 2)

    return {
        "num_samples": len(norm_refs),
        "bleu": round(bleu_res["score"], 2),
        "chrf": round(chrf_res["score"], 2),
        "wer": round(wer_val, 2),
        "cer": round(cer_val, 2),
        "exact_match_acc": exact_acc,
    }


def generate_predictions_batch(model, tokenizer, texts: List[str], is_indicbart: bool, device: str, batch_size: int = 64) -> List[str]:
    predictions = []
    model.eval()

    target_lang_id = None
    if is_indicbart:
        target_lang_id = tokenizer.convert_tokens_to_ids("<2mr>")

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        if is_indicbart:
            batch_inputs = [f"<2mr> {t} <2mr>" for t in batch]
            enc = tokenizer(batch_inputs, return_tensors="pt", padding=True, truncation=True, max_length=96).to(device)
            with torch.no_grad():
                outs = model.generate(
                    **enc,
                    forced_bos_token_id=target_lang_id,
                    max_length=96,
                    num_beams=2,
                )
            dec = [tokenizer.decode(o, skip_special_tokens=True).replace("<2mr>", "").strip() for o in outs]
        else:
            enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=96).to(device)
            with torch.no_grad():
                outs = model.generate(
                    **enc,
                    max_length=96,
                    num_beams=2,
                )
            dec = [tokenizer.decode(o, skip_special_tokens=True).strip() for o in outs]

        predictions.extend(dec)

    return predictions


def eval_checkpoint(model_path: Path, is_indicbart: bool, respin_groups: Dict, device: str) -> Dict:
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    if is_indicbart:
        model = MBartForConditionalGeneration.from_pretrained(str(model_path)).to(device)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(str(model_path)).to(device)

    results = {}
    all_preds = []
    all_refs = []

    for d_code in ["D1", "D2", "D3", "D4"]:
        items = respin_groups[d_code]
        texts = [it["text"] for it in items]
        preds = generate_predictions_batch(model, tokenizer, texts, is_indicbart, device)
        res = compute_metrics(preds, texts)
        results[d_code] = res
        logger.info(f"    --> [{d_code}]: BLEU: {res['bleu']:.2f} | chrF++: {res['chrf']:.2f} | WER: {res['wer']:.2f}% | Exact Match: {res['exact_match_acc']:.2f}%")

        all_preds.extend(preds)
        all_refs.extend(texts)

    comb_res = compute_metrics(all_preds, all_refs)
    results["Combined"] = comb_res
    logger.info(f"    --> [Combined (2,170)]: BLEU: {comb_res['bleu']:.2f} | chrF++: {comb_res['chrf']:.2f} | WER: {comb_res['wer']:.2f}% | Exact Match: {comb_res['exact_match_acc']:.2f}%")

    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


def main():
    setup_logger()
    parser = argparse.ArgumentParser(description="Exhaustive Evaluation of ALL Models and Folds on RESPIN Test Set.")
    parser.add_argument("--output-dir", type=str, default="reports/respin_fold_eval", help="Output directory for reports")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    respin_groups = load_respin_test_data()

    # All models to evaluate
    all_model_configs = [
        # IndicBART Multi-Fold Models
        {"name": "IndicBART Combined (16k)", "key": "indicbart_combined", "dir": Path("models/indicbart_combined"), "is_indicbart": True, "has_folds": True},
        {"name": "IndicBART Combined (32k)", "key": "indicbart_combined_32k", "dir": Path("models/indicbart_combined_32k"), "is_indicbart": True, "has_folds": True},
        {"name": "IndicBART Raw Unverified (32k)", "key": "indicbart_raw_unverified_32k", "dir": Path("models/indicbart_raw_unverified_32k"), "is_indicbart": True, "has_folds": True},
        {"name": "IndicBART D1 Malvani (16k)", "key": "indicbart_d1", "dir": Path("models/indicbart_d1"), "is_indicbart": True, "has_folds": True},
        {"name": "IndicBART D1 Malvani (32k)", "key": "indicbart_d1_32k", "dir": Path("models/indicbart_d1_32k"), "is_indicbart": True, "has_folds": True},
        {"name": "IndicBART D2 Ahirani (16k)", "key": "indicbart_d2", "dir": Path("models/indicbart_d2"), "is_indicbart": True, "has_folds": True},
        {"name": "IndicBART D2 Ahirani (32k)", "key": "indicbart_d2_32k", "dir": Path("models/indicbart_d2_32k"), "is_indicbart": True, "has_folds": True},
        {"name": "IndicBART D4 Varhadi (16k)", "key": "indicbart_d4", "dir": Path("models/indicbart_d4"), "is_indicbart": True, "has_folds": True},
        {"name": "IndicBART D4 Varhadi (32k)", "key": "indicbart_d4_32k", "dir": Path("models/indicbart_d4_32k"), "is_indicbart": True, "has_folds": True},
        # mT5 Models
        {"name": "mT5-Small Combined (16k)", "key": "mt5_combined_16k", "dir": Path("models/mt5_combined_16k"), "is_indicbart": False, "has_folds": False},
        {"name": "mT5-Small Combined (32k)", "key": "mt5_combined_32k", "dir": Path("models/mt5_combined_32k"), "is_indicbart": False, "has_folds": False},
        {"name": "mT5-Small Raw Unverified (32k)", "key": "mt5_raw_unverified_32k", "dir": Path("models/mt5_raw_unverified_32k"), "is_indicbart": False, "has_folds": False},
        {"name": "mT5-Small D1 Malvani (16k)", "key": "mt5_d1_16k", "dir": Path("models/mt5_d1_16k"), "is_indicbart": False, "has_folds": False},
        {"name": "mT5-Small D1 Malvani (32k)", "key": "mt5_d1_32k", "dir": Path("models/mt5_d1_32k"), "is_indicbart": False, "has_folds": False},
        {"name": "mT5-Small D2 Ahirani (16k)", "key": "mt5_d2_16k", "dir": Path("models/mt5_d2_16k"), "is_indicbart": False, "has_folds": False},
        {"name": "mT5-Small D2 Ahirani (32k)", "key": "mt5_d2_32k", "dir": Path("models/mt5_d2_32k"), "is_indicbart": False, "has_folds": False},
        {"name": "mT5-Small D4 Varhadi (16k)", "key": "mt5_d4_16k", "dir": Path("models/mt5_d4_16k"), "is_indicbart": False, "has_folds": False},
        {"name": "mT5-Small D4 Varhadi (32k)", "key": "mt5_d4_32k", "dir": Path("models/mt5_d4_32k"), "is_indicbart": False, "has_folds": False},
    ]

    all_eval_results = {}

    for m_info in all_model_configs:
        m_name = m_info["name"]
        m_dir = m_info["dir"]
        logger.info("=" * 80)
        logger.info(f"PROCESSING MODEL: {m_name} ({m_dir})")
        logger.info("=" * 80)

        fold_results = {}
        if m_info["has_folds"]:
            for fold in range(1, 6):
                ckpt = m_dir / f"fold_{fold}" / "best_model"
                if ckpt.exists():
                    logger.info(f"Evaluating {m_name} - Fold {fold} ({ckpt})...")
                    res = eval_checkpoint(ckpt, m_info["is_indicbart"], respin_groups, device)
                    fold_results[f"fold_{fold}"] = res
        else:
            ckpt = m_dir / "best_model"
            if ckpt.exists():
                logger.info(f"Evaluating {m_name} ({ckpt})...")
                res = eval_checkpoint(ckpt, m_info["is_indicbart"], respin_groups, device)
                fold_results["fold_1"] = res

        all_eval_results[m_info["key"]] = {
            "name": m_name,
            "dir": str(m_dir),
            "has_folds": m_info["has_folds"],
            "folds": fold_results,
        }

    # Save JSON and YAML
    json_path = output_dir / "respin_all_models_all_folds_eval.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_eval_results, f, indent=2, ensure_ascii=False)

    yaml_path = output_dir / "respin_all_models_all_folds_eval.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(all_eval_results, f, allow_unicode=True)

    # Generate Full Dialectwise Foldwise Markdown Matrix Reports
    md_file = output_dir / "respin_all_models_all_folds_report.md"
    docs_md_file = Path("docs/respin_5fold_dialectwise_matrix.md")

    for target_path in [md_file, docs_md_file]:
        with open(target_path, "w", encoding="utf-8") as f:
            f.write("# Master Empirical Benchmark: Dialectwise & Foldwise Evaluation on IISc RESPIN Test Set\n\n")
            f.write("**Official Held-Out Spoken Test Set**: `IISc_RESPIN_test_mr` (2,170 Utterances: D1=559, D2=540, D3=555, D4=516)\n\n")
            f.write("---\n\n")

            # 1. BLEU Score Matrix
            f.write("## 1. Dialectwise & Foldwise BLEU Score Matrix\n\n")
            f.write("| Model Name | Dialect Variety | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | **Mean BLEU ($\pm \sigma$)** |\n")
            f.write("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")

            for key, m_data in all_eval_results.items():
                name = m_data["name"]
                folds_dict = m_data["folds"]
                for split in ["D1 (Malvani)", "D2 (Ahirani)", "D3 (Standard)", "D4 (Varhadi)", "Combined"]:
                    split_code = split.split()[0]
                    bleus = [folds_dict[f][split_code]["bleu"] for f in sorted(folds_dict.keys()) if split_code in folds_dict[f]]
                    mb, sb = compute_mean_std(bleus)
                    b_row = " | ".join(f"{b:.2f}" for b in bleus) if len(bleus) == 5 else f"{bleus[0]:.2f} | - | - | - | -" if bleus else "- | - | - | - | -"
                    std_str = f"**{mb:.2f} $\pm$ {sb:.2f}**" if len(bleus) > 1 else f"**{mb:.2f}**"
                    f.write(f"| **{name}** | **{split}** | {b_row} | {std_str} |\n")

            f.write("\n---\n\n")

            # 2. WER Matrix
            f.write("## 2. Dialectwise & Foldwise Word Error Rate (WER %) Matrix\n\n")
            f.write("| Model Name | Dialect Variety | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | **Mean WER ($\pm \sigma$)** |\n")
            f.write("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")

            for key, m_data in all_eval_results.items():
                name = m_data["name"]
                folds_dict = m_data["folds"]
                for split in ["D1 (Malvani)", "D2 (Ahirani)", "D3 (Standard)", "D4 (Varhadi)", "Combined"]:
                    split_code = split.split()[0]
                    wers = [folds_dict[f][split_code]["wer"] for f in sorted(folds_dict.keys()) if split_code in folds_dict[f]]
                    mw, sw = compute_mean_std(wers)
                    w_row = " | ".join(f"{w:.2f}%" for w in wers) if len(wers) == 5 else f"{wers[0]:.2f}% | - | - | - | -" if wers else "- | - | - | - | -"
                    std_str = f"**{mw:.2f}% $\pm$ {sw:.2f}%**" if len(wers) > 1 else f"**{mw:.2f}%**"
                    f.write(f"| **{name}** | **{split}** | {w_row} | {std_str} |\n")

            f.write("\n---\n\n")

            # 3. chrF++ Matrix
            f.write("## 3. Dialectwise & Foldwise chrF++ Score Matrix\n\n")
            f.write("| Model Name | Dialect Variety | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | **Mean chrF++ ($\pm \sigma$)** |\n")
            f.write("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")

            for key, m_data in all_eval_results.items():
                name = m_data["name"]
                folds_dict = m_data["folds"]
                for split in ["D1 (Malvani)", "D2 (Ahirani)", "D3 (Standard)", "D4 (Varhadi)", "Combined"]:
                    split_code = split.split()[0]
                    chrfs = [folds_dict[f][split_code]["chrf"] for f in sorted(folds_dict.keys()) if split_code in folds_dict[f]]
                    mc, sc = compute_mean_std(chrfs)
                    c_row = " | ".join(f"{c:.2f}" for c in chrfs) if len(chrfs) == 5 else f"{chrfs[0]:.2f} | - | - | - | -" if chrfs else "- | - | - | - | -"
                    std_str = f"**{mc:.2f} $\pm$ {sc:.2f}**" if len(chrfs) > 1 else f"**{mc:.2f}**"
                    f.write(f"| **{name}** | **{split}** | {c_row} | {std_str} |\n")

            f.write("\n---\n\n")

    logger.info(f"Master all-model all-fold evaluation complete! Saved to: {md_file} and {docs_md_file}")


if __name__ == "__main__":
    main()
