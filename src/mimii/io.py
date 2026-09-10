"""List clips, load 16-bit PCM WAV via stdlib `wave`, build the manifest.

MIMII layout on disk (confirm, do not invent):
    data/{type}/id_0X/{normal,abnormal}/*.wav
Each clip is 8-channel, 16-bit PCM, 16 kHz.
"""
from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .const import CHANNEL, SR


@dataclass(frozen=True)
class Clip:
    path: str
    machine_type: str
    model_id: str        # "00", "02", ...
    snr_db: int
    y: int               # 0 normal, 1 anomaly


def load_wav(path: str | Path, channel: int = CHANNEL) -> np.ndarray:
    """Return one channel as float32 in [-1, 1], shape [T]. Fails loud on non-PCM16."""
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2:
            raise ValueError(f"{path}: expected 16-bit PCM, got {w.getsampwidth()*8}-bit")
        if w.getframerate() != SR:
            raise ValueError(f"{path}: expected {SR} Hz, got {w.getframerate()} Hz")
        nch = w.getnchannels()
        raw = w.readframes(w.getnframes())
    x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if nch > 1:
        x = x.reshape(-1, nch)[:, channel]
    return x


def list_clips(data_dir: str | Path, machine_type: str, snr_db: int) -> list[Clip]:
    """Walk {type}/id_*/{normal,abnormal}. Unknown layout -> fail loud (empty result)."""
    root = Path(data_dir) / machine_type
    if not root.is_dir():
        raise FileNotFoundError(f"no such machine dir: {root}")
    clips: list[Clip] = []
    for id_dir in sorted(root.glob("id_*")):
        model_id = id_dir.name.split("_", 1)[1]
        for label, y in (("normal", 0), ("abnormal", 1)):
            for wav in sorted((id_dir / label).glob("*.wav")):
                clips.append(Clip(str(wav), machine_type, model_id, snr_db, y))
    if not clips:
        raise FileNotFoundError(
            f"no clips under {root} (expected id_*/{{normal,abnormal}}/*.wav)"
        )
    return clips
