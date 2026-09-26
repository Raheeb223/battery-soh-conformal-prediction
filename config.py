"""
Central configuration for the SOH pipeline.
Set the BATTERY_DATA_ROOT environment variable (or edit ROOT_DIR below) to
point at the folder where you saved all downloaded datasets. See README.md
for the expected folder layout.
"""

import os

# ---- 1. Paths -------------------------------------------------------------
# Root folder containing all your downloaded dataset folders/zips (already extracted).
# Defaults to ./data next to this file.
ROOT_DIR = os.environ.get(
    "BATTERY_DATA_ROOT",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"),
)

NASA_DIR   = os.path.join(ROOT_DIR, "5. Battery Data Set")     # contains B0005.mat, B0006.mat, ...
CALCE_DIRS = {
    # CS2_8 and CS2_21 excluded: CALCE tested those two on a different
    # instrument that outputs .txt (not Arbin .xlsx), so they don't match
    # the parser used for every other CS2/CX2 cell below.
    "CS2": [os.path.join(ROOT_DIR, "CS2_33"), os.path.join(ROOT_DIR, "CS2_34"),
            os.path.join(ROOT_DIR, "CS2_35"), os.path.join(ROOT_DIR, "CS2_38")],
    "CX2": [os.path.join(ROOT_DIR, "CX2_16"), os.path.join(ROOT_DIR, "CX2_31"),
            os.path.join(ROOT_DIR, "CX2_33"), os.path.join(ROOT_DIR, "CX2_34"),
            os.path.join(ROOT_DIR, "CX2_35"), os.path.join(ROOT_DIR, "CX2_36"),
            os.path.join(ROOT_DIR, "CX2_37")],
}
OXFORD_MAT = os.path.join(ROOT_DIR, "Oxford_Battery_Degradation_Dataset_1.mat")
# If your FastCharge_..._structure.json files sit loose directly in ROOT_DIR
# (as opposed to inside their own subfolder), leave this as ROOT_DIR — the
# loader searches recursively either way, so this also works if you later
# move them into a subfolder.
MIT_DIR    = ROOT_DIR
BIT_DIR    = os.path.join(ROOT_DIR, "kw34hhw7xg-3")
XJTU_DIR   = os.path.join(ROOT_DIR, "Battery Dataset")
RWTH_DIR   = os.path.join(ROOT_DIR, "RWTH-2021-04545_818642", "Rawdata", "Rohdaten")

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
CACHE_DIR  = os.path.join(os.path.dirname(__file__), "cache")   # parsed capacity curves get cached here

# ---- 2. Windowing / sequence settings --------------------------------------
INPUT_WINDOW  = 20   # number of past cycles fed to the model
FORECAST_HORIZON = 1 # predict SOH this many cycles ahead (1 = next cycle)

# ---- 3. Split settings ------------------------------------------------------
# Cell-independent split: whole cells go to one split only (prevents leakage).
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
TEST_FRAC  = 0.15
RANDOM_SEED = 42

# ---- 4. Model hyperparameters -----------------------------------------------
HIDDEN_SIZE   = 64
NUM_LSTM_LAYERS = 2
DROPOUT       = 0.2        # also used for MC-Dropout at inference
ATTENTION_DIM = 32
BATCH_SIZE    = 64
LEARNING_RATE = 1e-3
NUM_EPOCHS    = 100
PATIENCE      = 15         # early stopping patience

# ---- 5. Conformal prediction -------------------------------------------------
CONFORMAL_ALPHA = 0.10     # 90% prediction interval
MC_DROPOUT_SAMPLES = 30    # forward passes at inference for uncertainty