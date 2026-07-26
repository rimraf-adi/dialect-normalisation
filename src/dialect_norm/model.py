import os
import sys
import time
import torch
from tqdm import tqdm
from transformers import AutoModel
from .audio import load_and_preprocess_audio


def load_indic_conformer_model(
    model_name: str = "ai4bharat/indic-conformer-600m-multilingual",
    token: str = None,
    device: str = "cuda",
):
    """Loads IndicConformer model from Hugging Face with authentication token handling."""
    print(f"Loading model: {model_name}...")
    try:
        model_kwargs = {"trust_remote_code": True}
        if token:
            model_kwargs["token"] = token
        elif os.environ.get("HF_TOKEN"):
            model_kwargs["token"] = os.environ.get("HF_TOKEN")

        model = AutoModel.from_pretrained(model_name, **model_kwargs)
    except Exception as e:
        if "401" in str(e) or "gated" in str(e).lower() or "restricted" in str(e).lower():
            print(
                f"\n[ERROR] Model '{model_name}' requires Hugging Face authentication/gated repo access.\n"
                "Please accept model terms at https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual\n"
                "and pass your access token via `--token YOUR_HF_TOKEN` or set `set HF_TOKEN=YOUR_HF_TOKEN`.\n",
                file=sys.stderr,
            )
        raise e

    model = model.to(device)
    model.eval()
    return model


def run_decoder_inference(
    model,
    samples: list,
    decoder_name: str,
    model_lang_code: str,
    device: str,
) -> list:
    """Runs inference for a specific decoder (CTC or RNNT) over a list of samples."""
    print(f"\n--- Running Inference with Decoder: {decoder_name.upper()} (Lang Code: {model_lang_code}) ---")
    predictions = []
    start_time = time.time()

    for sample in tqdm(samples, desc=f"Transcribing ({decoder_name.upper()})"):
        wav_path = sample["full_wav_path"]
        try:
            wav = load_and_preprocess_audio(wav_path).to(device)
            with torch.no_grad():
                transcription = model(wav, model_lang_code, decoder_name)
            if isinstance(transcription, (list, tuple)):
                transcription = transcription[0]
            predictions.append(str(transcription))
        except Exception as e:
            print(f"\nWarning: Error processing {wav_path}: {e}", file=sys.stderr)
            predictions.append("")

    elapsed_time = time.time() - start_time
    print(f"Inference ({decoder_name.upper()}) completed in {elapsed_time:.2f}s ({len(samples)/elapsed_time:.2f} UTT/s).")
    return predictions
