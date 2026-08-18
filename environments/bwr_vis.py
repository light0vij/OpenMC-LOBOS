"""
bwr_vis.py — Visualisation tools for the reactor enrichment-layout platform
-----------------------------------------------------------------------------

Functions in this file:-

plot_enr_grid(enr_grid, title, ax)--------------> Enrichment plotting
    
    
plot_bo_convergence(cfg, bo_data, save, dark)-----------------------> BO convergence plot
    
validate_and_plot_hifi(cfg, hifi_ref, best_lofi, hifi_opt, ...)--------------> for both GA and BO
    
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import Optional

from .mapping import ENR_LOW, ENR_HIGH



# Shared enrichment grid plot


def plot_enr_grid(
    enr_grid: np.ndarray,
    title: str = "Enrichment Layout",
    ax: Optional[plt.Axes] = None,
    enr_low:  float = ENR_LOW,
    enr_high: float = ENR_HIGH,
    dark: bool = True,
    save_path: Optional[str] = None,
) -> plt.Axes:
    """
    Render enrichment grid as a colour-coded cell map.

    Low-enrichment cells -> blue; high-enrichment cells -> red.
    Enrichment values are printed inside each cell.

    
    Visualizes the AI-generated enrichment layout.
    
    This is method is used to plot the geometry instead of OpenMC's native geometry plot, because:- 
    1. Conceptual Clarity: Focuses on the AI's binary decision grid (High vs. Low enrichment) 
        rather than cluttered physical geometry (cladding, gaps, etc.).
    2. High-Speed Reporting: Lightweight Matplotlib rendering avoids the heavy computational overhead 
        and kernel instability of OpenMC's geometry engine.
    3. AI Data Overlays: Designed to dynamically display AI-specific metrics like k-infinity and PPF 
        directly on the layout for rapid optimization feedback.
    
     """


    

    
    bg_ax  = "midnightblue" if dark else "whitesmoke"
    bg_fig = "black" if dark else "white"

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
        fig.patch.set_facecolor(bg_fig)

    ax.set_facecolor(bg_ax)
    n    = enr_grid.shape[0]
    cmap = {enr_low: "steelblue", enr_high: "crimson"}

    for r in range(n):
        for c in range(n):
            val = enr_grid[r, c]
            col = cmap.get(round(val, 4), "gray")
            rect = Rectangle([c, n - 1 - r], 1, 1,
                              facecolor=col, edgecolor="black", lw=1.5)
            ax.add_patch(rect)
            ax.text(c + 0.5, n - 0.5 - r, f"{val:.2f}",
                    ha="center", va="center", fontsize=7, color="white",
                    fontweight="bold")

    ax.set_xlim(0, n); ax.set_ylim(0, n)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=9, color="white" if dark else "black")
    ax.tick_params(colors="silver" if dark else "dimgray")
    for sp in ax.spines.values():
        sp.set_edgecolor("darkslateblue" if dark else "lightgray")

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=cmap[enr_high]),
        plt.Rectangle((0, 0), 1, 1, color=cmap[enr_low]),
    ]
    ax.legend(handles,
              [f"{enr_high}% (High)", f"{enr_low}% (Low)"],
              fontsize=6, facecolor=bg_ax,
              labelcolor="white" if dark else "black",
              edgecolor="darkslateblue" if dark else "lightgray",
              loc="lower right")
    
    #Plot saving
    if save_path:
        fig = ax.figure
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=bg_fig)
        print(f"Saved grid plot to -> {save_path}")
    
    return ax



