# CVIS CDR — independently reconstructed runnable research project

**Status: RECONSTRUCTED 2026-10-30, NOT THE ORIGINAL 2025 CODE OR MODEL WEIGHTS.**

This package reconstructs the journal method and incorporates recovered historical
configuration details from earlier fragmented directory. It has not been verified
against the original dataset or recovered checkpoint. Do not claim regenerated
figures or performance are identical to those in the publication before reproducing
them and documenting any differences. No private data, pretrained weights, or
original model checkpoint is included.

## Paper and recovered history

*A Physics-Informed Digital Twin Framework for Calibrated Sim-to-Real FMCW Radar Occupancy Estimation*, H. Trinh, S. Ratto V., E. Creager, G. Shaker.

| Item | Record from prior run |
|---|---|
| Native synthetic maps | 128x128 |
| Native real maps | 128x256 |
| Model input | 224x224 viridis RGB |
| ResNet18 | **random initialization**, binary / three-class head |
| Binary CDR | seed 40; accuracy ~97.1%, balanced ~96.7% |
| Three-class CDR | seed 42; accuracy 80.759%, balanced 72.057%, macro-F1 71.334% |
| Three-class checkpoint path (LOST) | `artifacts_3_class/resnet18_cdr_seed_42/best_model.pth` |
| Three-class test path (LOST) | `processed_data_3_class/test_final.h5` |
| Three-class norm path (LOST) | `processed_data_3_class/norm_stats.json` |

**These paths represent OLD filenames; the new checkpoints have a distinct `counting/` or `binary/` task subdirectory to prevent collisions.**

## Install on Sebastian's Windows/macOS/Linux machine

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e '.[test]'
python -m pytest -q tests/test_processing.py
```

PyTorch wheel variants may need platform-specific installation. If PyTorch
installation errors occur, install PyTorch/Torchvision from pytorch.org for the
machine's CPU/CUDA platform, then `pip install -e .`.

## Discover exact data structures FIRST

```bash
python -m cvis_cdr.inspect_data /path/to/real.h5 /path/to/norm_stats.json /path/to/sim_sample.npy
```

### Required input manifest

CSV with columns `path,domain,label,split,kind,session_id,participant_id`.
Use `/absolute/path/file.npy`, or path relative to the manifest's directory.
Labels `0=empty`, `1=one`, `2=two`; `-1=unlabeled empty-room calibration`.

`domain`: `sim` or `real`. `split`: `train`, `val`, `test`, `calibration`.
`kind`: `rd_db` (already RD dB; **DO NOT FFT again**), `rd_linear` (positive
magnitude), `rd_complex` (complex RD), or `adc_complex`/`adc_iq_json`
(raw ADC to FFT). See `examples/manifest_template.csv`.

A raw JSON must contain `{ "real": nested_arrays, "imag": nested_arrays }`
for each frame; **Infineon/vendor JSON structures vary**, so do not label
vendor JSON with this kind until conversion is verified.

For folder convention `root/{sim_train,sim_val,real_test,real_calibration}/{empty,one,two}/sample.csv`:

```bash
python -m cvis_cdr.manifest /path/to/root --kind rd_db --output manifest.csv
```

Then **manually check each row**. Generate a reviewer-ready provenance audit with:

```bash
python -m cvis_cdr.audit --manifest manifest.csv --out audit.json
```

 Do not invent `session_id` or `participant_id`.
The scanner leaves unknown IDs blank and warns. Simulation train/validation
must be disjoint at session/scene level, not adjacent frames from same trial.
The calibration real empty-room session must be isolated from labeled real test
sessions. No labeled real samples enter training or validation.

Use `--no-cache` on the training command when the recovered dataset is too large for RAM.

If Sebastian recovered a 3D H5 array, inspect it and export individual frames:

```bash
python -m cvis_cdr.extract_h5 data/real_test.h5 --frames-key /rd_db --labels-key /labels --domain real --split test --out data/real_from_h5
```

Use the **actual** observed key names, and merge its `manifest_part.csv` with
your other manifest rows. The tool does NOT infer sessions/participants.

## Train and evaluate

```bash
# Three-class paper reproduction attempts, historical seed 42
python -m src3_frame.sweep_frame_cnn --manifest manifest.csv --task counting --seeds 42 --epochs 30

# Binary paper reproduction attempts, historical seed 40
python -m src3_frame.sweep_frame_cnn --manifest manifest.csv --task binary --seeds 40 --epochs 30
```

Or individual runs:

```bash
python -m cvis_cdr.experiment train --manifest manifest.csv --task counting --method cdr --seed 42 --epochs 30 --input-size 224
python -m cvis_cdr.experiment train --manifest manifest.csv --task binary --method cdr --seed 40 --epochs 30 --input-size 224
```

Use the original normalization JSON if recovered (after confirming semantics):

```bash
python -m cvis_cdr.experiment train --manifest manifest.csv --task counting --method cdr --seed 42 --norm-stats processed_data_3_class/norm_stats.json
```

If exact original clipping values are confirmed, use `--vmin X --vmax Y`.
Otherwise the program uses **simulation TRAIN min/max**, an explicit
reconstruction consistent with the paper's description but not necessarily the
same implementation. Historical preprocessing logs mentioned -200 and ~-50.22,
but those are **not verified to be the final ResNet image clip bounds**.

At each epoch only SIMULATED validation selects the checkpoint; the real test
is evaluated only once after training. Test results for all 3 model methods are
saved to per-run JSON, confusion matrices, and prediction arrays.

After a training run, compare its saved metrics against the published targets:

```bash
python -m cvis_cdr.compare_to_paper --task counting --metrics-json artifacts_3_class/counting/resnet18_cdr_seed_42/test_metrics.json
```

To evaluate a checkpoint without retraining:

```bash
python -m cvis_cdr.experiment evaluate --manifest manifest.csv --checkpoint artifacts_3_class/counting/resnet18_cdr_seed_42/best_model.pth --output-dir artifacts_3_class/evaluation
```

To compare all trained methods and save prediction caches and comparison plot:

```bash
python -m src3_frame.eval_save_preds_3methods_3class --manifest manifest.csv --task counting --seed 42
```

To generate qualitative paper-style diagnostics (via Python):

```python
from cvis_cdr.figures import plot_diagnostics
plot_diagnostics('manifest.csv',[-200.,-50.22],'artifacts_3_class/diagnostics')
```

Feature-map plot: `python -m cvis_cdr.feature_maps --checkpoint CHECKPOINT.pth --sample SAMPLE.npy --out feature_maps.png`.

The chosen numeric bounds in this example are HISTORICAL CANDIDATES, not
verified final values. Figure diagnostics use an independently reconstructed
static-energy formula and are not exact original plots.

## Mathematical contract

Let `S` be the dB magnitude of the simulation RD map.

- RT: `S_rt = S`.
- Random DR: per frame sample `m ~ Uniform(-130,-100)`, `s ~ Uniform(1,4)`;
  independently per cell `N_ij ~ Normal(m,s^2)`;
  `S_random = max(S,N)`.
- CDR: obtain pooled `mu_t`, `sigma_t` from approximately 100 real unlabeled
  empty calibration frames in RD **dB** values; for each sim frame draw
  `N_ij ~ Normal(mu_t,sigma_t^2)` i.i.d.; `S_cdr=max(S,N)`.
- All methods: clip to fixed simulation-derived `[vmin,vmax]`, normalize to
  `[0,1]`, convert to viridis 3 channel, resize 224x224, ResNet18 random init.
- Real held-out test frames: identical transform, **no DR**, **no CDR**.

**Unknown original details (must not be silently treated as recovered):** raw
ADC JSON key names, FFT axes/window/zero-padding/receiver handling, time
alignment, image resize interpolation, per-pixel vs per-frame balancing, exact
train/val split, augmentation, optimizer, LR scheduler, early stopping, batch
size, padding/interpolation of 128x128 versus 128x256, exact clipping values,
random number sequence, and exact trained weights. This reconstruction makes
explicit, changeable assumptions to run; it cannot guarantee the 2025 result.

## Test data supports from actual paper confusion matrices

- Binary test: 427 empty and 184 one-person = **611 frames**.
- Counting test: 427 empty, 184 one-person, 153 two-person = **764 frames**.
- Counting CDR confusion matrix rows true / columns predicted:
  `[[423,4,0],[3,88,93],[2,45,106]]`.
- Three-class CDR: accuracy 0.8075916, balanced 0.7205679, F1 0.7133443.

These are **frame** supports, NOT counts of sessions or participants. The
manifests must be sourced from real recording metadata to answer reviewers.

## Toy smoke test: no actual paper data

```bash
python -m cvis_cdr.smoke_data --root toy_data --train-per-class 4 --val-per-class 2 --test-per-class 2 --calibration-frames 6
python -m cvis_cdr.experiment train --manifest toy_data/manifest.csv --task binary --method cdr --seed 40 --epochs 1 --batch-size 4 --input-size 64 --vmin -200 --vmax -50
```

Never report these synthetic toy scores as paper results.

## Sharing checklist

Include: manifest with provenance, original simulation source files, real
unlabeled calibration frames, labeled real test frames, exact preprocessing
configuration, frozen train/val split, code archive and version lock,
checkpoint(s), evaluation JSON, confusion matrices, environment versions.
Regenerate the paper numbers and report deviations rather than editing result
files. Share unpublished measurements and attribution according to PI/co-author
permissions; the ZIP intentionally does not include data.
