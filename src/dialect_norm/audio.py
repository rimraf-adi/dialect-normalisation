import torch
import torchaudio


def load_and_preprocess_audio(audio_path: str, target_sample_rate: int = 16000) -> torch.Tensor:
    """Loads an audio file, converts multi-channel audio to mono, and resamples to target_sample_rate."""
    wav, sr = torchaudio.load(audio_path)
    if wav.ndim > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)
    if sr != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sample_rate)
        wav = resampler(wav)
    return wav
