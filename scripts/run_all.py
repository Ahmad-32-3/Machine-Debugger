"""Slice 2: run the same protocol across all four MIMII types.

Subprocess-calls run.py per type (it downloads if missing, evals, writes
out/metrics.json), then snapshots each type to out/<type>.json. Idempotent:
already-populated types skip their ~10 GB download.

    python scripts/run_all.py --test-id 06 --snr 0 --data data
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from mimii.const import TYPES  # noqa: E402

OUT = ROOT / "out"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-id", default="06")
    ap.add_argument("--snr", type=int, default=0)
    ap.add_argument("--data", default="data")
    ap.add_argument("--features", default="logmel", choices=["logmel", "vggish"])
    a = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    suffix = "" if a.features == "logmel" else f"_{a.features}"
    for t in TYPES:
        print(f"\n===== {t} ({a.features}) =====", flush=True)
        r = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "run.py"),
             "--type", t, "--test-id", a.test_id, "--snr", str(a.snr), "--data", a.data,
             "--features", a.features],
            cwd=str(ROOT),
        )
        if r.returncode != 0:
            print(f"  {t} FAILED (exit {r.returncode}); stopping", flush=True)
            return r.returncode
        shutil.copy(OUT / "metrics.json", OUT / f"{t}{suffix}.json")
        print(f"  snapshot -> out/{t}{suffix}.json", flush=True)

    print("\nall four types done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
