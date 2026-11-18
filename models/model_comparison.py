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
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    axes[0].bar(x - width / 2, comparison.cv_observation_top1_accuracy, width, label="Grouped CV")
    axes[0].bar(x + width / 2, comparison.student_test_observation_top1_accuracy, width, label="Student test")
    axes[0].set_title("Observation-level top-1 accuracy")
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[1].bar(x - width / 2, comparison.cv_f1, width, label="Grouped CV")
    axes[1].bar(x + width / 2, comparison.student_test_f1, width, label="Student test")
    axes[1].set_title("Candidate-level F1 (threshold 0.5)")
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("F1")
    axes[1].legend()
    for ax in axes:
        ax.set_xticks(x, labels, rotation=25, ha="right")
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("PathSnap model comparison — weak route-proximity labels")
    fig.savefig(MODEL_DIR / "model_comparison.png", dpi=180)
    plt.close(fig)
    print(f"Wrote {MODEL_DIR / 'model_comparison.png'}")


if __name__ == "__main__":
    main()
