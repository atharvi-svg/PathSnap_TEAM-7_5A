"""Create comparison graphs from recorded CV and student-test metrics."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from models.common import MODEL_DIR


def main() -> None:
    comparison = pd.read_csv(MODEL_DIR / "model_comparison.csv")
    labels = comparison.model.astype(str).str.replace(" (Bayesian tuned)", "\n(tuned)", regex=False)
    x = np.arange(len(comparison))
    width = 0.36
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), constrained_layout=True)
    axes = axes.ravel()
    axes[0].bar(x - width / 2, comparison.cv_observation_top1_accuracy, width, label="Grouped CV")
    axes[0].bar(x + width / 2, comparison.student_test_observation_top1_accuracy, width, label="Student test")
    axes[0].set_title("Observation-level top-1 accuracy")
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[1].bar(x - width / 2, comparison.cv_macro_f1, width, label="Grouped CV")
    axes[1].bar(x + width / 2, comparison.student_test_macro_f1, width, label="Student test")
    axes[1].set_title("Candidate-level macro F1 (threshold 0.5)")
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Macro F1")
    axes[1].legend()
    axes[2].bar(x, comparison.student_test_log_loss, color="#7c3aed")
    axes[2].set_title("Student-test log loss")
    axes[2].set_ylabel("Log loss (lower is better)")
    axes[2].grid(axis="y", alpha=.2)
    axes[3].bar(x, comparison.inference_ms_per_candidate, color="#0891b2")
    axes[3].set_title("Inference time per candidate")
    axes[3].set_ylabel("Milliseconds")
    axes[3].grid(axis="y", alpha=.2)
    for ax in axes:
        ax.set_xticks(x, labels, rotation=25, ha="right")
        if ax in axes[:2]:
            ax.grid(axis="y", alpha=.2)
    fig.suptitle("PathSnap model comparison — weak route-proximity labels")
    fig.savefig(MODEL_DIR / "model_comparison.png", dpi=180)
    plt.close(fig)
    print(f"Wrote {MODEL_DIR / 'model_comparison.png'}")


if __name__ == "__main__":
    main()
