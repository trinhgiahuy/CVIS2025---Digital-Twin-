# Start here: CVIS CDR rerun

**This archive is a new code reconstruction, not the historical 2025 source.
No paper reproduction was asserted without original data and weights.**

## First, inspect the recovered files

```bash
python -m venv .venv
# macOS/Linux: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e '.[test]'
pytest -q tests/
python -m cvis_cdr.inspect_data FULL_PATH_TO_DATA_FILE.h5
```

Determine whether each file contains RAW COMPLEX ADC or ALREADY PROCESSED
RANGE-DOPPLER dB. **Never apply the FFT to an already processed RD map**.
The paper input is RGB viridis images from clipped/normalized RD dB arrays.

## Then, produce an audited manifest

For per-frame CSV or NumPy files organized as:

```
data/
  sim_train/empty,one,two/
  sim_val/empty,one,two/
  real_calibration/empty/
  real_test/empty,one,two/
```

Run:

```bash
python -m cvis_cdr.manifest data --kind rd_db --output manifest.csv
```

**You MUST audit the manifest** for actual session/participant provenance and
calibration/test separation. File count is not session count. If your recovered
files are H5 arrays rather than individual files, use inspect_data and
extract_h5 with exact known dataset keys. See README.md.

## Run three methods against the same frozen split

```bash
python -m src3_frame.sweep_frame_cnn --manifest manifest.csv --task counting --seeds 42 --epochs 30
python -m src3_frame.eval_save_preds_3methods_3class --manifest manifest.csv --task counting --seed 42
python -m src3_frame.sweep_frame_cnn --manifest manifest.csv --task binary --seeds 40 --epochs 30
python -m src3_frame.eval_save_preds_3methods_3class --manifest manifest.csv --task binary --seed 40
```

Or in Windows PowerShell: `./scripts/run_all.ps1 -Manifest manifest.csv`.

The defaults for optimizer, LR, epoch count, FFT options and clipping selection
are reconstruction choices. The paper does not specify all these details.

## Published numbers to compare against (do NOT hardcode into predictions)

- Three-class real test support: 427 empty, 184 one, 153 two = 764 frames.
- Counting CDR: overall accuracy 0.8075916230; balanced accuracy
  0.7205678819; macro-F1 0.7133443386. Confusion:
  `[[423,4,0],[3,88,93],[2,45,106]]`.
- Binary CDR: confusion `[[417,10],[8,176]]` = 611 frames.
- Original execution path clues: `src_3_classes.config`, `src3_frame`,
  `processed_data_3_class/test_final.h5`, `norm_stats.json`,
  `artifacts_3_class/resnet18_cdr_seed_42/best_model.pth`.
- Older binary run directory: `~/seb_paper/clip_sim2real/zero_shot_sim2real_binary_occupancy`;
  its `src_pt.train_baselines` pipeline used `processed_data_3_class_pt/`
  H5 filenames `train_frames`, `val_frames`, `train_frames_matched`,
  `val_frames_matched`, `test_frames_raw`. Those were earlier experiments,
  not evidence of the final trained binary CDR code.

Any mismatch with the published metrics must be reported, not concealed.
Please send source data schemas, capture/session logs, and resulting model
metrics to co-authors before asserting exact reproduction.
