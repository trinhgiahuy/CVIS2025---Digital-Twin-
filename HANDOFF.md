# CVIS25 code reconstruction handoff

This is a NEW, runnable reconstruction from the paper and our experiment logs,
not the byte-for-byte lost implementation. The published results have NOT
been independently rerun without the original data and model checkpoints.

## What it includes

- `src/cvis_cdr/records.py`: explicit CSV/NPY/NPZ/H5 JSON radar file ingestion;
  support for precomputed RD dB and ADC complex I/Q (only if format matches)
- `src/cvis_cdr/processing.py`: configurable 2-D FFT, floor sampling,
  calibration mu/sigma, viridis image conversion
- `src/cvis_cdr/experiment.py`: scratch ResNet18, simulation-only train/val,
  three methods, test-once evaluation, weights/metrics/confusion
- `src/cvis_cdr/compare.py`: three method comparison and NPZ prediction cache
- `src/cvis_cdr/figures.py`, `feature_maps.py`: paper-style diagnostic plots
- `src/cvis_cdr/inspect_data.py`, `extract_h5.py`, `manifest.py`: adapt the
  recovered data with explicit schema / provenance; do not guess H5 keys
- `src3_frame`, `src_3_classes`, `src_pt`: **reconstructed compatibility names**
- `tests/`: numerical unit tests and CPU train/evaluation smoke test

## Quick start

1. `pip install -e '.[test]'`
2. `pytest -q tests/` (synthetic unit/smoke tests)
3. `python -m cvis_cdr.inspect_data PATH_TO_FIRST_REAL_H5 PATH_TO_FIRST_SIM_ARRAY`
4. Create and **audit** `manifest.csv` (see `examples/manifest_template.csv`)
5. `python -m cvis_cdr.experiment train --manifest manifest.csv --task counting --method cdr --seed 42 --epochs 30`
6. `python -m src3_frame.sweep_frame_cnn --manifest manifest.csv --task counting --seeds 42`
7. `python -m src3_frame.eval_save_preds_3methods_3class --manifest manifest.csv --task counting --seed 42`
8. Repeat binary with `--task binary --seed 40`.

## Before attempting 2025 exact scores

Verify RD physics convention and original normalization values; 128x128 sim vs
128x256 real, and 224x224 ResNet18 resize. Confirm original train/val splits,
100-frame separate calibration set, real test frames with original labels, and
actual 2025 model hyperparameters. The reconstruction's AdamW, LR=1e-4,
weight_decay=1e-4, 30 epochs, patience=7, no augmentation, FFT defaults,
RGB resize order, `sigma` population SD and per-sample RNG are ASSUMPTIONS,
not recovered original settings. You can adjust via `--help`.

Known original held-out frame support: 427 empty, 184 one, 153 two.
Archived CDR seed42 3class confusion: [[423,4,0],[3,88,93],[2,45,106]].
Archived CDR three-class metrics: ACC=.8075916230, BAL=.7205678819, macroF1=.7133443386.
Binary seed40 CDR confusion: [[417,10],[8,176]] gives ACC=.9705401,
BAL=.9665513.

## Reviewer questions

`frame_counts != session_counts != participant_counts`. Please extract
`session_id`, `participant_id`, acquisition setup, repetitions and any simultaneous
recording information from ORIGINAL acquisition logs. The attached paper only
specifies the real test frame support, not unique participants/sessions or exact
number of simultaneous/independent measurements. Do not guess. Infineon ADC
clipping is not the same as a fixed dB image clipping window; distinguish the
sensor clipping/saturation operation from model-input vmin/vmax. This package
implements the latter and does not claim to reconstruct the former.

The selected input is `viridis`, 3-channel RGB into random-initialized ResNet18.
Real test images are NEVER randomized; CDR injects noise-floor values into SIM
RD dB training inputs ONLY. Three class model evaluation is with simulation
training and held-out real test, not real-label fine-tuning.

Archive exact data, manifest, hashes, environments, checkpoints and run logs
when you rerun this, and distinguish any outcome from the journal publication.
