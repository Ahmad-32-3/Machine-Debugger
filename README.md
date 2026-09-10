# Machine Debugger

Fault detection from machine audio, tested on a machine left out of training.

I train on machine A and test on machine B of the same type. Clip accuracy is not the headline. The numbers I trust are AUC / PR-AUC under that ID shift. Same-machine scores are the debug column.

## Data

[MIMII](https://zenodo.org/records/3384388).

## Run

```bash
python -m pytest tests/ -q
python scripts/run.py
npm --prefix web install
npm --prefix web run dev
```

If the Zenodo download is blocked, leak tests and the walkthrough still run. The CLI prints `ILLUSTRATIVE` plus a one-line blocker.

## Layout

- `src/` audio features, detector, eval
- `scripts/run.py`
- `tests/` machine-id leak checks
- `web/` case-study page
