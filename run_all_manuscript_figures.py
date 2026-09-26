"""
Runs every figure-generation script in order. Nothing is retrained; the
scripts read the outputs of experiments.py and remaining_analysis.py.

Prerequisites:
  - outputs/*_results.json and outputs/*_predictions.npz  (experiments.py)
  - outputs/significance_testing_results.json              (remaining_analysis.py)
  - cache/processed.npz                                     (preprocessing.py)
  - outputs/*_model.pt checkpoints                          (experiments.py; used by
                                                             benchmark_inference_efficiency.py
                                                             and visualize_attention.py)

Run:
    python run_all_manuscript_figures.py
Output:
    outputs/figures/ -- see each script's docstring for the file names.
"""

import subprocess
import sys

SCRIPTS = [
    ("generate_all_figures.py",
     "Core manuscript figures: prediction-vs-actual, residuals, per-cell "
     "curves, conformal coverage comparison (headline result), baseline/"
     "ablation bars, window-size ablation, LODO, significance forest plot, "
     "late-stage aging analysis. (10 figures)"),
    ("benchmark_inference_efficiency.py",
     "Model size and inference latency comparison across architectures."),
    ("visualize_attention.py",
     "Attention-weight visualization on healthy vs. degraded input curves. "
     "NOTE: check this one's console output carefully -- it introspects "
     "your model's structure and may need ATTENTION_MODULE_NAME_HINT set "
     "if auto-detection picks the wrong module."),
    ("diagnose_calce_generalization_gap.py",
     "CALCE-vs-others distributional comparison (degradation rate, "
     "knee/nonlinearity score, min SOH, cycle count) -- supports the "
     "manuscript's CALCE generalization-gap discussion."),
]


def run_script(path, description):
    print("\n" + "=" * 70)
    print(f"Running: {path}")
    print(f"  {description}")
    print("=" * 70)
    result = subprocess.run([sys.executable, path])
    if result.returncode != 0:
        print(f"\n[!] {path} exited with code {result.returncode} -- "
              f"check its output above for the specific error before "
              f"trusting any figures it may have partially produced.")
        return False
    return True


def main():
    print("[run_all_manuscript_figures] generating every manuscript figure "
          "from already-completed experiment runs...")

    results = {}
    for path, description in SCRIPTS:
        results[path] = run_script(path, description)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for path, ok in results.items():
        status = "OK" if ok else "FAILED -- see output above"
        print(f"  [{status}] {path}")

    if all(results.values()):
        print("\nAll figures generated successfully. Check outputs/figures/.")
    else:
        print("\nSome scripts failed -- fix those specifically before "
              "assuming all figures are ready for the manuscript.")


if __name__ == "__main__":
    main()
