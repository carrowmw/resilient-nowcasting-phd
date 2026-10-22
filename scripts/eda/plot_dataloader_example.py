#!/usr/bin/env python3
"""
scripts/eda/plot_dataloader_example.py

Generates a publication-quality figure for Chapter 5 (Figure 5.2):
Fixed-size data loading window illustrating:
  - Input context window (W = 24 steps, 6 hours)
  - Unpredicted horizon gap (h = 24 steps, 6 hours lead time)
  - Single point forecast target at t + h (n = 47)
  - Primary sensor readings (blue solid) vs deterministic temporal covariates (green dashed)

Outputs:
  - /Users/administrator/Code/latex-documents/thesis/figures/experiment1/dataloader_example_training_sequence.png
  - artifacts/dataloader_example_training_sequence.png
"""
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# GOV.UK / Publication Design Palette
# -----------------------------------------------------------------------------
C_PRIMARY = "#1d70b8"      # GOV.UK Blue (Primary Sensor Reading)
C_COVARIATE = "#00703c"    # GOV.UK Green (Temporal Covariate)
C_TARGET = "#d4351c"       # GOV.UK Red (Target Label)
C_TEXT = "#0b0c0c"         # Dark Charcoal Text
C_SEC_TEXT = "#505a5f"     # Secondary Slate Text
C_GRID = "#e5e7eb"         # Light Grey Grid
C_BORDER = "#b1b4b6"       # Border Grey
C_GUIDE = "#8e959b"        # Faint Guide Line

def main():
    # 1. Trajectory Data
    x_input = np.arange(24)
    y_input = np.array([
        -1.87, -1.86, -1.24, -0.58,  0.04,  0.10,  0.16,  0.22, 
         0.28,  0.35,  0.40,  0.91,  1.42,  1.93,  2.45,  1.00, 
        -0.43, -1.87, -0.31, -0.58, -0.87, -1.15, -1.43, -1.71
    ])

    x_all = np.arange(48)
    # Temporal covariate (smooth sinusoidal cycle representing harmonic calendar feature)
    y_covariate = 1.55 * np.cos(np.pi * (x_all + 4) / 48) - 0.15

    target_x = 47
    target_y = -0.31

    # 2. Figure Setup
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = 11

    fig, ax = plt.subplots(figsize=(14.5, 7.8), dpi=300)

    # Shaded backgrounds for Input Window vs Horizon Gap
    ax.axvspan(-0.8, 23.5, color="#f0f4f8", alpha=0.6, zorder=0, label="_nolegend_")
    ax.axvspan(23.5, 48.8, color="#fdfbf7", alpha=0.6, zorder=0, label="_nolegend_")

    # Grid
    ax.grid(True, linestyle="-", color=C_GRID, alpha=0.9, zorder=1)

    # Vertical dividing line at window boundary
    ax.axvline(23.5, color="#d4351c", linestyle=":", linewidth=2.2, alpha=0.85, zorder=2)

    # 3. Plots
    # Dotted projection guide line from last input to target
    ax.plot([x_input[-1], target_x], [y_input[-1], target_y], 
            color=C_GUIDE, linestyle=":", linewidth=2.0, alpha=0.75, zorder=2, label="_nolegend_")

    # Temporal Covariate (Green dashed across entire sequence)
    line_cov, = ax.plot(x_all, y_covariate, color=C_COVARIATE, linestyle="--", linewidth=2.4,
                        alpha=0.9, zorder=3, label="Temporal Covariate (green dashed: sin harmonic, known in advance)")

    # Primary Sensor Reading (Blue solid with markers)
    line_prim, = ax.plot(x_input, y_input, color=C_PRIMARY, linestyle="-", linewidth=3.2,
                         marker="o", markersize=7.5, markerfacecolor=C_PRIMARY, markeredgewidth=1.2,
                         markeredgecolor="#ffffff", zorder=4, label="Primary Sensor Reading (blue solid: observed counts up to t)")

    # Target Label (Red Star at t+h)
    line_target = ax.scatter([target_x], [target_y], color=C_TARGET, s=340, marker="*",
                             edgecolor="#991b1b", linewidth=1.2, zorder=5, 
                             label="Target Label (red star: single point forecast at t + h)")

    # 4. Text Annotations & Callouts
    # Header regions at top
    ax.text(11.5, 2.95, "Input Window (W = 24 steps / 6 hours)", 
            ha="center", va="center", fontsize=13, fontweight="bold", color=C_TEXT)
    ax.text(35.5, 2.95, "Horizon Gap (h = 24 steps / 6 hours lead time)", 
            ha="center", va="center", fontsize=13, fontweight="bold", color=C_SEC_TEXT)
    ax.text(35.5, 2.70, "(Unobserved & unpredicted intermediate steps)", 
            ha="center", va="center", fontsize=10.5, fontstyle="italic", color=C_SEC_TEXT)

    # Callout for Target Point Forecast
    ax.annotate(
        "Point Forecast at t + h = 47\n(Direct multi-step target,\n not seq-to-seq rollout)",
        xy=(target_x, target_y),
        xytext=(target_x - 11.8, target_y + 1.15),
        arrowprops=dict(
            arrowstyle="->",
            connectionstyle="arc3,rad=-0.15",
            color=C_TARGET,
            lw=1.8
        ),
        fontsize=10.5,
        fontweight="bold",
        color=C_TARGET,
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#fff5f5", edgecolor=C_TARGET, alpha=0.95, lw=1.2),
        zorder=6
    )

    # Callout at boundary (Forecast Origin)
    ax.annotate(
        "Forecast Origin t = 23\n(Boundary of observed history)",
        xy=(23.5, -1.75),
        xytext=(23.5, -2.45),
        arrowprops=dict(
            arrowstyle="->",
            connectionstyle="arc3,rad=0.0",
            color="#a32015",
            lw=1.5
        ),
        ha="center", va="center",
        fontsize=9.5, fontweight="bold", color="#a32015",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#ffffff", edgecolor="#d4351c", alpha=0.95, lw=1.0),
        zorder=6
    )

    # 5. Axes formatting
    ax.set_title("Fixed-Size Data Loading Window (Context W = 24, Forecast Horizon h = 24)", 
                 fontsize=15, fontweight="bold", pad=26, color=C_TEXT)
    ax.set_xlabel("Temporal Index (n) [15-minute intervals]", fontsize=12.5, fontweight="bold", labelpad=12, color=C_TEXT)
    ax.set_ylabel("Standardised Value (z-score)", fontsize=12.5, fontweight="bold", labelpad=12, color=C_TEXT)

    ax.set_xlim(-1.0, 49.0)
    ax.set_ylim(-2.8, 3.25)
    ax.set_xticks(np.arange(0, 49, 4))

    for spine in ax.spines.values():
        spine.set_color(C_BORDER)
        spine.set_linewidth(1.2)

    # 6. Legend positioned at upper left below title
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.02, 0.88),
        fontsize=10.2,
        frameon=True,
        facecolor="#ffffff",
        edgecolor=C_BORDER,
        framealpha=0.95,
        borderpad=0.7,
        labelspacing=0.5
    )

    plt.tight_layout()

    # Save to artifacts and thesis
    os.makedirs("artifacts", exist_ok=True)
    out_artifact = "artifacts/dataloader_example_training_sequence.png"
    fig.savefig(out_artifact, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_artifact}")

    out_artifact_pdf = "artifacts/dataloader_example_training_sequence.pdf"
    fig.savefig(out_artifact_pdf, bbox_inches="tight")
    print(f"Saved: {out_artifact_pdf}")

    # Copy to thesis directory
    thesis_fig_dir = "/Users/administrator/Code/latex-documents/thesis/figures/experiment1"
    if os.path.exists(thesis_fig_dir):
        dest_png = os.path.join(thesis_fig_dir, "dataloader_example_training_sequence.png")
        shutil.copyfile(out_artifact, dest_png)
        print(f"Copied to: {dest_png}")
        dest_pdf = os.path.join(thesis_fig_dir, "dataloader_example_training_sequence.pdf")
        shutil.copyfile(out_artifact_pdf, dest_pdf)
        print(f"Copied to: {dest_pdf}")

if __name__ == "__main__":
    main()
