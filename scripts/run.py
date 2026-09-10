"""Slice-1 pipeline: download-if-missing -> manifest -> features -> loio + same_id.

    python scripts/run.py --type fan --test-id 06 --snr 0 --data data/

Writes out/metrics.json and out/scores.jsonl. Prints the loio-vs-same_id table.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from mimii import split as splits  # noqa: E402
from mimii.const import IDS, SNRS, TYPES  # noqa: E402
from mimii.detect import Detector  # noqa: E402
from mimii.eval import evaluate, print_table  # noqa: E402
from mimii.features import clip_vector  # noqa: E402
from mimii.io import Clip, list_clips, load_wav  # noqa: E402

OUT = Path("out")
FEAT = clip_vector  # feature fn; swapped to the VGGish embedder by --features vggish


def _vectors(clips, cache):
    skipped = 0
    X, y, ids = [], [], []
    for c in clips:
        if c.path not in cache:
            try:
                cache[c.path] = FEAT(load_wav(c.path))
            except (OSError, ValueError):
                skipped += 1
                continue
        X.append(cache[c.path])
        y.append(c.y)
        ids.append(c.model_id)
    if skipped:
        print(f"  skipped {skipped} unreadable clip(s)")
    return np.array(X), np.array(y), np.array(ids)


def _run_one(clips, train, test, cache, split_name, records):
    Xtr, _, tr_ids = _vectors(train, cache)
    Xte, yte, _ = _vectors(test, cache)
    det = Detector().fit(Xtr, tr_ids)  # per-machine norm needs train model_ids
    scores = det.score(Xte)
    for c, s in zip([c for c in test if c.path in cache], scores):
        records.append({"path": c.path, "machine_type": c.machine_type,
                        "model_id": c.model_id, "y": c.y, "score": float(s),
                        "split": split_name})
    return evaluate(yte, scores)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--type", default="fan", choices=TYPES)
    ap.add_argument("--test-id", default="06", choices=IDS)
    ap.add_argument("--snr", type=int, default=0, choices=SNRS)
    ap.add_argument("--data", default="data")
    ap.add_argument("--features", default="logmel", choices=["logmel", "vggish"])
    a = ap.parse_args()

    if a.features == "vggish":
        global FEAT
        from mimii.embed import clip_vector_vggish
        FEAT = clip_vector_vggish

    from download import ensure  # scripts/ is on sys.path via __main__
    ensure(a.data, a.type, a.snr)

    clips = list_clips(a.data, a.type, a.snr)
    debug_id = next(i for i in IDS if i != a.test_id)  # same_id uses a DIFFERENT id

    cache: dict[str, np.ndarray] = {}
    records: list[dict] = []
    tr, te = splits.loio(clips, a.test_id)
    m_loio = _run_one(clips, tr, te, cache, "loio", records)
    tr, te = splits.same_id(clips, debug_id)
    m_same = _run_one(clips, tr, te, cache, "same_id", records)

    print_table(m_loio, m_same, a.type, a.test_id)

    OUT.mkdir(exist_ok=True)
    metrics = {
        "machine_type": a.type, "snr_db": a.snr, "channel": 0, "features": a.features,
        "test_id": a.test_id, "train_ids": [i for i in IDS if i != a.test_id],
        "loio": m_loio, "same_id": {**m_same, "debug_id": debug_id},
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    with (OUT / "scores.jsonl").open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"\n  wrote {OUT/'metrics.json'} and {OUT/'scores.jsonl'}")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))  # for `import download`
    raise SystemExit(main())
