"""Optional frozen SSL embedder: VGGish (AudioSet-pretrained, 16 kHz native).

Same detect.py/eval.py backend as log-mel; only the per-clip vector changes.
Loaded lazily via torch.hub so importing mimii stays torch-free until asked.
"""
from __future__ import annotations

import numpy as np

_model = None


def _get_model():
    global _model
    if _model is None:
        import torch

        m = torch.hub.load("harritaylor/torchvggish", "vggish", trust_repo=True)
        m.postprocess = False  # want continuous 128-d, not PCA-quantized bytes
        m.eval()
        _model = m
    return _model


def clip_vector_vggish(x: np.ndarray) -> np.ndarray:
    """16 kHz waveform [-1,1] -> 128-d clip embedding (mean over VGGish frames)."""
    import torch

    m = _get_model()
    with torch.no_grad():
        emb = m(x.astype("float32"), fs=16000)
    emb = emb.detach().cpu().numpy()
    if emb.ndim == 1:  # single 0.96 s frame
        return emb.astype(np.float32)
    if emb.size == 0:  # clip shorter than one frame — shouldn't happen on MIMII
        return np.zeros(128, dtype=np.float32)
    return emb.mean(axis=0).astype(np.float32)


if __name__ == "__main__":  # self-check: 1 s synthetic tone -> 128-d finite vector
    t = np.linspace(0, 1, 16000, endpoint=False, dtype=np.float32)
    v = clip_vector_vggish(0.2 * np.sin(2 * np.pi * 440 * t))
    assert v.shape == (128,) and np.isfinite(v).all(), v.shape
    print("PASS embed:", v.shape)

