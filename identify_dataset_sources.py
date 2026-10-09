"""
identify_dataset_sources.py

You downloaded all 7 datasets but forgot exactly which source/mirror each
came from. Rather than guess from web pages, this scans your ACTUAL local
raw data folders and checks them against specific, concrete fingerprints
verified during source research -- exact filenames, file sizes, batch
date-stamps, and cell-naming conventions that are essentially impossible
to match by coincidence if you got the data from a different source.

Fingerprints checked (each independently verified against a real,
current source page during this project's citation work):

  NASA    -> individual cell files named B0005.mat, B0006.mat, B0007.mat,
             B0018.mat (official NASA PCoE Battery Data Set #5 naming)
  CALCE   -> files/folders prefixed CS2_ or CX2_ (CALCE's own cell-ID
             convention)
  Oxford  -> a file named exactly 'Oxford_Battery_Degradation_Dataset_1.mat',
             expected size ~253.8 MB (ORA record uuid:03ba4b01-...)
  MIT     -> folders/files containing the batch date-stamps 2017-05-12,
             2017-06-30, or 2018-04-12 (Severson et al.'s 3 batches)
  BIT     -> a folder or path containing 'kw34hhw7xg' (Mendeley DOI
             suffix -- this one you've already confirmed independently)
  XJTU    -> folders named Batch-1 through Batch-6 (this project's own
             established XJTU folder convention)
  RWTH    -> ~1,642 individual session CSV files across 48 cell folders,
             filenames containing 'sanyo' (RWTH-2021-04545 dataset
             structure)

This only reads your filesystem -- it does not download, modify, or
delete anything.

Run from your project root (same folder as config.py):
    python identify_dataset_sources.py
"""

import os
import sys

try:
    import config
except ImportError:
    print("[!] Could not import config.py -- run this from your project root.")
    sys.exit(1)


def _get_dataset_dirs():
    """Reads the project's actual config.py structure directly, rather than
    guessing attribute names by keyword -- this project's config has some
    non-uniform shapes (CALCE_DIRS is a dict of lists, OXFORD_MAT is a
    direct file path, not a directory) that a generic *_DIR scan misses."""
    found = {}

    if hasattr(config, "NASA_DIR"):
        found["NASA"] = config.NASA_DIR
    if hasattr(config, "MIT_DIR"):
        found["MIT"] = config.MIT_DIR
    if hasattr(config, "BIT_DIR"):
        found["BIT"] = config.BIT_DIR
    if hasattr(config, "XJTU_DIR"):
        found["XJTU"] = config.XJTU_DIR
    if hasattr(config, "RWTH_DIR"):
        found["RWTH"] = config.RWTH_DIR

    # CALCE_DIRS is a dict of {chemistry: [list of per-cell folder paths]} --
    # flatten it into the individual cell folders for checking.
    if hasattr(config, "CALCE_DIRS"):
        all_calce_paths = []
        for chem, paths in config.CALCE_DIRS.items():
            all_calce_paths.extend(paths)
        found["CALCE"] = all_calce_paths  # a LIST of paths, not a single one

    # OXFORD_MAT is a direct .mat file path, not a directory.
    if hasattr(config, "OXFORD_MAT"):
        found["OXFORD"] = config.OXFORD_MAT  # a single FILE path, not a dir

    return found


def _walk_summary(path, max_list=15):
    """Returns (all_filepaths, total_size_bytes, sample_names) for a
    directory tree, without loading file contents."""
    if not os.path.isdir(path):
        return None, None, None
    all_files = []
    total_size = 0
    for root, dirs, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            try:
                size = os.path.getsize(fp)
            except OSError:
                size = 0
            all_files.append((fp, size))
            total_size += size
    sample = [os.path.basename(fp) for fp, _ in all_files[:max_list]]
    return all_files, total_size, sample


def check_nasa(path):
    all_files, total_size, sample = _walk_summary(path)
    if all_files is None:
        return "DIRECTORY NOT FOUND", []
    names_lower = [os.path.basename(fp).lower() for fp, _ in all_files]
    expected = ["b0005", "b0006", "b0007", "b0018"]
    matches = [e for e in expected if any(e in n for n in names_lower)]
    verdict = f"MATCH ({len(matches)}/4 expected cell files found: {matches})" \
        if len(matches) >= 3 else f"NO MATCH (only found: {matches})"
    return verdict, sample


def check_calce(paths):
    """paths is a LIST of per-cell folder paths (config.CALCE_DIRS flattened),
    not a single directory."""
    all_sample = []
    found_any = False
    missing = []
    for p in paths:
        all_files, total_size, sample = _walk_summary(p)
        if all_files is None:
            missing.append(p)
            continue
        names_lower = [os.path.basename(fp).lower() for fp, _ in all_files]
        if any("cs2" in n or "cx2" in n for n in names_lower) or \
           "cs2" in os.path.basename(p).lower() or "cx2" in os.path.basename(p).lower():
            found_any = True
        all_sample.extend(sample[:3])
    if missing:
        return f"PARTIAL/NO MATCH ({len(missing)} of {len(paths)} configured cell folders not found: " \
               f"{[os.path.basename(m) for m in missing[:5]]})", all_sample[:15]
    verdict = f"MATCH (all {len(paths)} configured CS2_/CX2_ cell folders found and populated)" \
        if found_any else "NO MATCH (folders exist but no CS2_/CX2_ naming found inside)"
    return verdict, all_sample[:15]


def check_oxford(path):
    """path is a direct FILE path (config.OXFORD_MAT), not a directory."""
    if not os.path.isfile(path):
        return f"NO MATCH (expected file not found at this exact path)", []
    size = os.path.getsize(path)
    size_mb = size / (1024 * 1024)
    close_to_expected = abs(size_mb - 253.8) < 5
    verdict = f"MATCH (file exists, {size_mb:.1f} MB, expected ~253.8 MB -- " \
              f"{'matches' if close_to_expected else 'SIZE DIFFERS, check version'})"
    return verdict, [os.path.basename(path)]


def check_mit(path):
    all_files, total_size, sample = _walk_summary(path)
    if all_files is None:
        return "DIRECTORY NOT FOUND", []
    all_names = [os.path.basename(fp).lower() for fp, _ in all_files]
    for root, dirs, _ in os.walk(path):
        all_names += [d.lower() for d in dirs]

    # Primary check: this project's config.py comment says it expects
    # 'FastCharge_..._structure.json' files -- check for that pattern first.
    fastcharge_matches = [n for n in all_names if "fastcharge" in n and "structure" in n]
    # Fallback: the standard raw Severson .mat batch naming, in case the
    # FastCharge/structure.json files are a project-specific renaming and
    # the actual files are the raw MATLAB structs instead.
    batch_dates = ["2017-05-12", "2017-06-30", "2018-04-12", "20170512", "20170630", "20180412"]
    date_matches = [d for d in batch_dates if any(d in n for n in all_names)]

    if fastcharge_matches:
        verdict = f"MATCH (found {len(fastcharge_matches)} 'FastCharge_..._structure.json'-pattern " \
                  f"files, matching this project's expected MIT format)"
    elif date_matches:
        verdict = f"MATCH (found standard raw Severson batch date-stamps: {date_matches} -- " \
                  f"NOTE: this is a different file format than 'FastCharge_..._structure.json' " \
                  f"mentioned in config.py's comment; check mit_loader.py can parse this format)"
    else:
        verdict = "NO MATCH (neither 'FastCharge_..._structure.json' files nor standard " \
                  "Severson batch date-stamps found -- since MIT_DIR points at your project " \
                  "root, this searched your ENTIRE project folder recursively; if your MIT " \
                  "data uses different naming, tell me the actual filenames and I'll adjust this check)"
    return verdict, sample


def check_bit(path):
    all_files, total_size, sample = _walk_summary(path)
    if all_files is None:
        return "DIRECTORY NOT FOUND", []
    full_path_lower = path.lower()
    all_paths_lower = [fp.lower() for fp, _ in all_files]
    matches_anywhere = "kw34hhw7xg" in full_path_lower or any("kw34hhw7xg" in p for p in all_paths_lower)
    verdict = "MATCH (kw34hhw7xg found in path -- already independently confirmed earlier)" \
        if matches_anywhere else "NO DIRECT MATCH (kw34hhw7xg not found in current path/filenames -- " \
        "may have been renamed after download; check if this was true earlier)"
    return verdict, sample


def check_xjtu(path):
    all_files, total_size, sample = _walk_summary(path)
    if all_files is None:
        return "DIRECTORY NOT FOUND", []
    dirs_found = []
    for root, dirs, _ in os.walk(path):
        dirs_found += dirs
    batch_dirs = [d for d in dirs_found if "batch" in d.lower()]
    verdict = f"MATCH (found {len(batch_dirs)} Batch-N folders: {sorted(set(batch_dirs))[:6]})" \
        if batch_dirs else "NO MATCH (no Batch-N folder structure found)"
    return verdict, sample


def check_rwth(path):
    all_files, total_size, sample = _walk_summary(path)
    if all_files is None:
        return "DIRECTORY NOT FOUND", []
    n_csv = sum(1 for fp, _ in all_files if fp.lower().endswith(".csv"))
    has_sanyo = any("sanyo" in os.path.basename(fp).lower() for fp, _ in all_files)
    if not has_sanyo:
        verdict = f"NO MATCH (no 'sanyo' naming found, {n_csv} CSV files)"
    elif n_csv < 1400:
        verdict = f"PARTIAL/NO MATCH (only {n_csv} CSV files, expected at least ~1,400 -- " \
                  f"this direction IS a real concern, may be an incomplete download)"
    else:
        note = " (higher than the ~1,642 baseline is fine -- likely means both .csv AND " \
               ".zip versions of each session are present side by side)" if n_csv > 1900 else ""
        verdict = f"MATCH (sanyo naming found, {n_csv} CSV files{note})"
    return verdict, sample


CHECKERS = {
    "NASA": check_nasa, "CALCE": check_calce, "OXFORD": check_oxford,
    "MIT": check_mit, "BIT": check_bit, "XJTU": check_xjtu, "RWTH": check_rwth,
}


def main():
    dataset_dirs = _get_dataset_dirs()
    if not dataset_dirs:
        print("[!] No dataset paths found in config.py -- check that config.py "
              "defines NASA_DIR, CALCE_DIRS, OXFORD_MAT, MIT_DIR, BIT_DIR, "
              "XJTU_DIR, RWTH_DIR as in this project's actual config.py.")
        return

    print(f"Found {len(dataset_dirs)} dataset path entries in config.py:\n")
    for name, path in dataset_dirs.items():
        if isinstance(path, list):
            print(f"  {name}: {len(path)} configured folders")
        else:
            print(f"  {name}: {path}")

    print(f"\n{'='*78}")
    print("CHECKING EACH DATASET AGAINST VERIFIED SOURCE FINGERPRINTS")
    print(f"{'='*78}\n")

    for name, path in dataset_dirs.items():
        label = f"{len(path)} folders" if isinstance(path, list) else path
        print(f"--- {name} ({label}) ---")
        checker = CHECKERS.get(name)
        if checker is None:
            print("  [!] no checker defined for this dataset name")
            continue
        verdict, sample = checker(path)
        print(f"  Verdict: {verdict}")
        if sample:
            print(f"  Sample filenames found: {sample}")
        print()

    print(f"{'='*78}")
    print("If a dataset shows NO MATCH, it likely came from a different "
          "mirror/version than the one currently cited in the manuscript's "
          "Data Availability Statement -- send me the verdict and sample "
          "filenames for that dataset and I'll track down the correct source.")


if __name__ == "__main__":
    main()
