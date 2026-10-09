# Battery SOH Multi-Dataset Pipeline

Trains a multi-scale BiLSTM + self-attention model with MC-Dropout and
conformal prediction to forecast State-of-Health (SOH) from capacity-fade
curves, across NASA, CALCE (CS2/CX2), Oxford, and MIT/Severson datasets.

## 0. What each file does

| File | Purpose |
|---|---|
| `config.py` | All paths and hyperparameters — **edit this first**. |
| `inspect_data.py` | Prints the real internal structure of your raw files. **Run this before anything else.** |
| `data_loaders/*.py` | One parser per dataset — turns raw files into `(cycle, capacity, cell_id, dataset)` rows. |
| `preprocessing.py` | Combines all datasets, computes SOH, splits cell-independently, builds sliding windows, caches to `cache/processed.npz`. |
| `model.py` | The neural network + MC-Dropout inference helper. |
| `conformal.py` | Calibrated prediction intervals (fixes MC-Dropout's overconfidence). |
| `train.py` | Trains the model, evaluates, saves results to `outputs/`. |

## 1. Install dependencies

Open a terminal in this folder and run:

```bash
pip install -r requirements.txt
```

(If you're on Windows and `pip` isn't recognized, use `py -m pip install -r requirements.txt`.)

## 2. Point the code at your data

Open `config.py` and change **one line**:

```python
ROOT_DIR = r"C:\path\to\your\battery_datasets"
```

Set this to the folder where you have `5. Battery Data Set/`, `CS2_8/`,
`CX2_16/`, `Oxford_Battery_Degradation_Dataset_1.mat`, and your `FastCharge/`
folder (rename your extracted MIT JSON folder to `FastCharge` or update
`MIT_DIR` in `config.py` to match your actual folder name).

Also double check `CALCE_DIRS` in `config.py` lists every CS2/CX2 cell folder
you actually downloaded — add or remove entries to match your files.

## 3. Inspect your raw files (mandatory first run)

```bash
python inspect_data.py
```

Battery dataset files are notoriously inconsistent in field/column naming
even within "the same" dataset released by different labs. This script
prints out what's actually inside your files. **Read the output.** If it
prints `[!]` next to any dataset, that loader's assumed field names don't
match your files — see step 4.

## 4. Fix a loader if inspect_data.py flagged one

Each loader file (`data_loaders/nasa_loader.py`, `calce_loader.py`,
`oxford_loader.py`, `mit_loader.py`) has lines marked `# ADJUST HERE`. Open
the flagged loader, compare the field names `inspect_data.py` printed
against what the code expects, and edit those lines to match. This is
normal — these datasets don't ship with a single consistent schema.

The **Oxford loader is the most likely to need adjustment** — its MATLAB
struct nesting varies by MATLAB version used to save it. If it fails, run
this in a Python shell to explore manually:

```python
import scipy.io as sio
mat = sio.loadmat("Oxford_Battery_Degradation_Dataset_1.mat", squeeze_me=True, struct_as_record=False)
root = mat[[k for k in mat if not k.startswith("__")][0]]
print(dir(root))                      # cell names
print(dir(getattr(root, "Cell1")))    # cycle field names for Cell1
```

## 5. Run preprocessing

```bash
python preprocessing.py
```

This will print how many cells/rows were parsed per dataset, how the
cell-independent train/val/test split came out, and the final windowed
array shapes. **A dataset that fails to parse is skipped with a warning**,
not a crash — so check the printed output to make sure all the datasets you
expect actually made it in.

## 6. Train

```bash
python train.py
```

This trains with early stopping, then reports:
- Test MAE / RMSE
- Conformal prediction coverage (should be close to 90% if `CONFORMAL_ALPHA=0.10`)
- Per-dataset MAE breakdown (useful for your paper's cross-dataset
  generalization discussion)

Outputs land in `outputs/`:
- `best_model.pt` — trained weights
- `test_predictions.npz` — predictions, MC-Dropout uncertainty, conformal
  intervals, per-sample dataset labels (for plotting later)

## 7. Common first-run problems

- **"No datasets loaded successfully"** — none of the 4 loaders worked.
  Re-run `inspect_data.py`, check `ROOT_DIR` is correct, and confirm the
  subfolders actually exist where `config.py` expects them.
- **CALCE cells parse very few rows** — the Arbin excel export sometimes
  splits sheets differently across CALCE's own file batches. Open one file
  manually in Excel and compare its sheet names to what
  `calce_loader.CANDIDATE_CYCLE_COLS` / `CANDIDATE_CAP_COLS` expect.
- **Training set is empty** — usually means one dataset has too few cells
  and `cell_independent_split` couldn't split it 3 ways; check the
  `[split] WARNING` messages preprocessing.py prints.
- **CUDA not available** — the code auto-falls-back to CPU. Training on CPU
  with this data size (capacity-fade curves, not raw waveform data) is still
  fast — should be minutes, not hours.

## 8. Notes on what this pipeline does NOT do yet

- It does not include your **Turku dataset** — none of your recent uploads
  showed it downloaded yet, so no loader exists for it. Once you download
  it, tell me its file format and I'll add `data_loaders/turku_loader.py`
  following the same pattern.
- It uses only the **capacity-fade curve** (one number per cycle) as the
  input signal, not full per-cycle voltage/current/temperature curves. This
  is standard for SOH forecasting and keeps parsing tractable across four
  differently-formatted datasets — but if your novelty claim specifically
  needs richer per-cycle features, that's a bigger extension to
  `preprocessing.py`'s `make_windows()` function, not a small tweak.
