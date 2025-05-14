import torchaudio 
import numpy as np

def load_audio(path, sr=None):
    wav, orig_sr = torchaudio.load(path)   # (channels, samples) Tensor, sr
    return wav.squeeze(0).numpy(), orig_sr

def preprocess_audio(wav, orig_sr, target_sr=16000, duration=1.0):
    import torch
    # 1) to Tensor
    wav_t = torch.from_numpy(wav).unsqueeze(0)            # [1, n]
    # 2) resample
    wav_rs = torchaudio.functional.resample(
        wav_t, orig_freq=orig_sr, new_freq=target_sr
    )
    wav = wav_rs.squeeze(0).numpy()
    # 3) pad/trim
    max_len = int(target_sr * duration)
    if len(wav) < max_len:
        wav = np.pad(wav, (0, max_len - len(wav)))
    else:
        wav = wav[:max_len]
    # 4) normalize & cast
    wav = wav / (np.max(np.abs(wav)) + 1e-9)
    return wav.astype(np.float32)
