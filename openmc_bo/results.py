"""
results.py — Result analysis and CSV export 
--------------------------------------------

Doesnt  deal with OpenMC os Heuristic constraints. 
ONly dataprocessing.
(Was a part of the notebook first. later moved to this file)



This module strictly handles the Agent's internal math, convergence 
diagnostics, and data processing. It does NOT deal with OpenMC physics 
or heuristic constraints (which belong in the Environment).

Functions:
  summarise_results: Identifies the best layout and prints a clean console summary.
  plot_convergence: Generates a 4-panel diagnostic dashboard tracking the GP score, k_inf, and PPF.
  _save_bo_results_csv: Exports the full iteration history to CSV.
"""

from __future__ import annotations
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

from .config import RunConfig



# Summarise Results


def summarise_results(
    cfg: RunConfig,
    bo_data: dict,
    save: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Identify the best design and print a results summary.
    """
    score    = bo_data["score"]
    keff     = bo_data["keff"]
    keff_std = bo_data["keff_std"]
    ppf      = bo_data["ppf"]
    X        = bo_data["X"]
    n_seed    = bo_data.get("n_seed", 0)
    n_total  = len(score)

    best_i     = int(np.argmax(score))
    best_k     = float(keff[best_i])
    best_kstd  = float(keff_std[best_i])
    best_p     = float(ppf[best_i])
    best_s     = float(score[best_i])
    best_x     = X[best_i].copy()
    delta_k    = abs(cfg.k_target - best_k)
    enr_grid   = cfg.decode(best_x)
    phase      = "LLM" if best_i < n_seed else f"BO (iter {best_i - n_seed + 1})"
    feas_count = ((ppf <= cfg.ppf_target).sum()
                  if cfg.ppf_target is not None else n_total)
    n_high     = int((np.round(enr_grid, 4) == round(cfg.enr_high, 4)).sum())
    avg_enr    = enr_grid.mean()

    best = dict(
        x        = best_x,
        keff     = best_k,
        keff_std = best_kstd,
        ppf      = best_p,
        score    = best_s,
        delta_k  = delta_k,
        enr_grid = enr_grid,
    )

    if verbose:
        std_str = f" ± {best_kstd:.5f}" if best_kstd > 0 else ""
        print("═" * 70)
        print(f"  OPTIMAL ENRICHMENT LAYOUT (BO) — {cfg.model_name}")
        print("═" * 70)
        print(f"  Found at                           : {phase}")
        print(f"  High-enrichment rods ({cfg.enr_high}%)  : "
              f"{n_high} / {cfg.n_rods_total}")
        print(f"  Average enrichment                 : {avg_enr:.4f} wt%")
        print()
        print(f"  k∞                                 : {best_k:.5f}{std_str}")
        print(f"  k∞ target                          : {cfg.k_target}")
        print(f"  |Δk∞|                              : {delta_k:.5f}  "
              f"({delta_k*1e5:.1f} pcm)")
        if cfg.ppf_target is not None:
            sat = "SATISFIED  " if best_p <= cfg.ppf_target else "NOT SATISFIED"
            print(f"  PPF                                : {best_p:.4f}  "
                  f"(target ≤ {cfg.ppf_target})")
            print(f"  PPF constraint                     : {sat}")
        print(f"  Composite score                    : {best_s:.6f}")
        print(f"  Total evaluations                  : {n_total}")
        print(f"  Feasible designs found             : {feas_count} / {n_total}")
        print("═" * 70)

    if save:
        _save_bo_results_csv(cfg, bo_data)

    return best



# Plot Convergence Dashboard (4-Panel)


def plot_convergence(             #–––––––––––––––––––––––> moved from environments.bwr_vis.py
    cfg: RunConfig,
    bo_data: dict,
    save: bool = True,
    dark: bool = True,
) -> plt.Figure:
    """
    Diagnoses the AI's performance by tracking the internal math. 
    Charts the score landscape, feasibility, and the transition from LLM guesiing to surrogate modelling (BO).
    """
    keff    = bo_data["keff"]
    ppf     = bo_data["ppf"]
    score   = bo_data["score"]
    n_seed   = bo_data.get("n_seed", 0)
    total   = len(keff)
    delta_k = np.abs(cfg.k_target - keff)
    feas    = (ppf <= cfg.ppf_target) if cfg.ppf_target is not None \
              else np.ones(total, bool)

    bg_fig = "black"         if dark else "white"
    bg_ax  = "midnightblue"  if dark else "whitesmoke"
    tc     = "silver"        if dark else "dimgray"
    spine  = "darkslateblue" if dark else "lightgray"
    wc     = "white"         if dark else "black"

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor(bg_fig)
    fig.suptitle(f"Bayesian Optimisation Convergence — {cfg.model_name}",
                 color=wc, fontsize=14, y=1.01)
    
    for ax in axes.flat:
        ax.set_facecolor(bg_ax); ax.tick_params(colors=tc)
        ax.xaxis.label.set_color(tc); ax.yaxis.label.set_color(tc)
        ax.title.set_color(wc)
        for sp in ax.spines.values(): sp.set_edgecolor(spine)

    idxs = np.arange(1, total + 1)

    # k∞ history
    ax = axes[0, 0]
    llm_inf  = ~feas[:n_seed]; llm_feas = feas[:n_seed]
    ax.scatter(np.where(llm_inf)[0]  + 1, keff[:n_seed][llm_inf],
               c="slategray", s=35, alpha=0.6, marker="x", label="LLM sampling infeasible")
    ax.scatter(np.where(llm_feas)[0] + 1, keff[:n_seed][llm_feas],
               c="steelblue", s=55, alpha=0.8, label="LLM sampling feasible")
    bo_colors = ["orange" if feas[n_seed + j] else "gray"
                 for j in range(total - n_seed)]
    ax.scatter(range(n_seed + 1, total + 1), keff[n_seed:],
               c=bo_colors, s=75, alpha=0.9, marker="D", label="BO samples")
    ax.axhline(cfg.k_target, color="mediumseagreen", ls="--", lw=1.5,
               label=f"Target = {cfg.k_target}")
    ax.axvline(n_seed + 0.5, color="dimgray", ls="--", lw=1, label="BO start")
    ax.set_xlabel("Sample"); ax.set_ylabel("k∞"); ax.set_title("k∞ history")
    ax.legend(fontsize=7, facecolor=bg_ax, labelcolor=wc, edgecolor=spine)
    ax.grid(True, alpha=0.15)

    # PPF history
    ax = axes[0, 1]
    ax.plot(idxs, ppf, "o-", color="brown", alpha=0.7, lw=1.5, ms=4)
    if cfg.ppf_target is not None:
        ax.axhline(cfg.ppf_target, color="firebrick", ls="--", lw=1.5,
                   label=f"Limit = {cfg.ppf_target}")
        ax.fill_between(idxs, cfg.ppf_target, ppf,
                        where=(ppf > cfg.ppf_target), color="firebrick",
                        alpha=0.15, label="Above limit")
    ax.axvline(n_seed + 0.5, color="dimgray", ls=":", lw=1, label="BO start")
    ax.set_xlabel("Sample"); ax.set_ylabel("PPF"); ax.set_title("PPF history")
    ax.legend(fontsize=7, facecolor=bg_ax, labelcolor=wc, edgecolor=spine)
    ax.grid(True, alpha=0.15)

    # |Δk∞|
    ax = axes[1, 0]
    ax.plot(idxs[:n_seed],  delta_k[:n_seed],
            "o-", color="steelblue", alpha=0.7, lw=1.5, ms=4, label="LLM")
    ax.plot(idxs[n_seed:],  delta_k[n_seed:],
            "D-", color="orange",    alpha=0.8, lw=1.5, ms=5, label="BO")
    ax.axvline(n_seed + 0.5, color="dimgray", ls="--", lw=1)
    best_i = int(np.argmax(score))
    ax.scatter(best_i + 1, delta_k[best_i],
               c="green", s=250, marker="*", zorder=5, label="Best design")
    ax.set_xlabel("Sample"); ax.set_ylabel(f"|k∞ − {cfg.k_target}|")
    ax.set_title(f"|Δk∞| from target ({cfg.k_target})")
    ax.legend(fontsize=7, facecolor=bg_ax, labelcolor=wc, edgecolor=spine)
    ax.grid(True, alpha=0.15)

    # Best score convergence
    ax = axes[1, 1]
    best_so_far = np.maximum.accumulate(score)
    ax.plot(idxs[:n_seed],    best_so_far[:n_seed],
            "-", color="steelblue", lw=2, label="LLM (best of all)")
    ax.plot(idxs[n_seed - 1:], best_so_far[n_seed - 1:],
            "-", color="orange",   lw=2, label="BO  (best of all)")
    ax.axvline(n_seed + 0.5, color="dimgray", ls="--", lw=1, label="BO start")
    ax.set_xlabel("Sample"); ax.set_ylabel("Best score so far")
    ax.set_title("Convergence")
    ax.legend(fontsize=7, facecolor=bg_ax, labelcolor=wc, edgecolor=spine)
    ax.grid(True, alpha=0.15)

    plt.tight_layout()

    if save:
        os.makedirs(cfg.results_dir, exist_ok=True)
        path = os.path.join(cfg.results_dir, f"{cfg.model_name}_bo_convergence.png")
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=bg_fig)
        print(f"Convergence plot saved → {path}")

    return fig



# CSV Saving


def _save_bo_results_csv(cfg: RunConfig, bo_data: dict) -> str: 
    """Save all evaluated samples (LLM + BO) to a CSV file."""
    os.makedirs(cfg.results_dir, exist_ok=True)
    path = os.path.join(cfg.results_dir, f"{cfg.model_name}_all_evaluations.csv")

    X        = bo_data["X"]
    keff     = bo_data["keff"]
    keff_std = bo_data["keff_std"]
    ppf      = bo_data["ppf"]
    score    = bo_data["score"]
    n_seed    = bo_data.get("n_seed", 0)
    n_total  = len(keff)

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        x_names = [f"x{i:02d}" for i in range(cfg.n_vars)]
        w.writerow(["index", "phase"] + x_names
                   + ["keff", "keff_std", "ppf", "score"])
        for i in range(n_total):
            phase = "LLM" if i < n_seed else f"BO_{i - n_seed + 1}"
            row   = ([i, phase]
                     + list(X[i])
                     + [keff[i], keff_std[i], ppf[i], score[i]])
            w.writerow(row)

    print(f"BO results saved –> {path}")
    return path