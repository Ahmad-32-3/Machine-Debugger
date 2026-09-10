"""Fetch one Zenodo MIMII zip into data/ and extract. Stdlib only.

    python scripts/download.py --type fan --snr 0 --data data/
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from mimii.const import SNRS, TYPES, ZENODO_BASE  # noqa: E402


def zip_url(machine_type: str, snr_db: int) -> str:
    return f"{ZENODO_BASE}/{snr_db}_dB_{machine_type}.zip?download=1"


def _report(count: list[int], block: int, total: int) -> None:
    count[0] += 1
    done = count[0] * block
    pct = 100 * done / total if total > 0 else 0
    print(f"\r  {done/1e6:8.1f} MB / {total/1e6:.1f} MB ({pct:4.1f}%)", end="", flush=True)


def ensure(data_dir: str | Path, machine_type: str, snr_db: int) -> Path:
    """Download+extract if data/<type> is missing. Returns data dir. Idempotent."""
    data = Path(data_dir)
    target = data / machine_type
    if target.is_dir() and any(target.glob("id_*/normal/*.wav")):
        print(f"  {target} already populated, skipping download")
        return data
    data.mkdir(parents=True, exist_ok=True)
    url = zip_url(machine_type, snr_db)
    zpath = data / f"{snr_db}_dB_{machine_type}.zip"
    print(f"  downloading {url}")
    counter = [0]
    urllib.request.urlretrieve(url, zpath, lambda c, b, t: _report(counter, b, t))
    print("\n  extracting ...")
    with zipfile.ZipFile(zpath) as z:
        z.extractall(data)
    zpath.unlink()  # zip is gitignored anyway; free the disk
    return data


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--type", default="fan", choices=TYPES)
    ap.add_argument("--snr", type=int, default=0, choices=SNRS)
    ap.add_argument("--data", default="data")
    a = ap.parse_args()
    ensure(a.data, a.type, a.snr)
