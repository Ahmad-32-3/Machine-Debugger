"""Slice-1 constants. One place for the numbers so nothing invents its own."""

SR = 16000                       # MIMII sample rate (Hz)
TYPES = ("fan", "pump", "slider", "valve")
IDS = ("00", "02", "04", "06")   # public MIMII model IDs
SNRS = (-6, 0, 6)                # dB tags on the zip names

DEFAULT_TYPE = "fan"
DEFAULT_SNR = 0
CHANNEL = 0                      # ch0; mean-of-8 is an ablation, not the first run

# Zenodo record 3384388 (Purohit et al., DCASE 2019). Files: "{snr}_dB_{type}.zip".
ZENODO_BASE = "https://zenodo.org/records/3384388/files"

# log-mel feature params
N_FFT = 1024
HOP = 512
N_MELS = 64
FAR_TARGET = 0.1                # FAR operating point we report (10% false alarms)
