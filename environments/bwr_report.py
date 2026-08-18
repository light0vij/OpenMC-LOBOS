"""
bwr_report.py — Design report generation for the reactor platform
------------------------------------------------------------------

export_design_report(cfg, result_data, enr_grid_opt, model_params,
                     algorithm, filename)
    Writes a detailed .txt report covering:
      • Physics results (keff, PPF, fitness/score)
      • Algorithm parameters
      • Geometry & material constants
      • Per-pin enrichment table 
"""

from __future__ import annotations

import os
import numpy as np
from typing import Optional

from .mapping import make_sym_index, ENR_LOW, ENR_HIGH


def export_design_report(
    cfg,
    result_data:  dict,
    enr_grid_opt: np.ndarray,
    model_params: dict,
    algorithm:    str = "BO",
    filename:     Optional[str] = None,
) -> str:
    """
    Generate and save a detailed plain-text design report for any NxN assembly.

    """
    algo_lower = algorithm.lower()
    if filename is None:
        filename = f"optimised_enrichment_layout_{algo_lower}.txt"

    save_dir = getattr(cfg, "results_dir", "results")
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)

    # Geometry
    n_rods_side  = model_params.get("n_rods_side", enr_grid_opt.shape[0])
    n_rods_total = n_rods_side ** 2
    sym_index    = make_sym_index(n_rods_side)

    best_keff  = float(result_data["keff"])
    best_ppf   = float(result_data["ppf"])
    delta_k    = float(result_data.get("delta_k", abs(cfg.k_target - best_keff)))
    ppf_target = getattr(cfg, "ppf_target", None)
    sat_str    = (
        "SATISFIED "
        if (ppf_target is None or best_ppf <= ppf_target)
        else "NOT SATISFIED "
    )

    n_high  = int((np.round(enr_grid_opt, 4) == round(ENR_HIGH, 4)).sum())
    n_low   = n_rods_total - n_high
    avg_enr = enr_grid_opt.mean()

    # Algorithm — GA branch commented out (see note in docstring above and
    # _ga_params_block below); this pipeline only ever runs BO.
    # if algorithm.upper() in ("GA",):
    #     perf_line  = (f"  Best fitness          : "
    #                   f"{result_data.get('fitness', 'N/A'):.6f}  (0 = perfect)")
    #     chrom_line = f"  HOF chromosome        : {list(result_data.get('chromosome', []))}"
    #     algo_block = _ga_params_block(cfg)
    # else:
    perf_line  = (f"  Composite score       : "
                  f"{result_data.get('score', 'N/A'):.6f}")
    chrom_line = ""
    algo_block = _bo_params_block(cfg)

    # Full Report 
    spec = f"""  OPTIMAL {n_rods_side}×{n_rods_side} ENRICHMENT LAYOUT ({algorithm}) — {cfg.model_name}
======================================================================================
  Algorithm             : {algorithm}
  Assembly size         : {n_rods_side}×{n_rods_side} = {n_rods_total} fuel pins
  Symmetric positions   : {len(sym_index)}  (1/2-diagonal symmetry)
  Enrichment options    : {ENR_LOW}% (Low)  /  {ENR_HIGH}% (High)
  High-enrichment rods  : {n_high} / {n_rods_total}
  Low-enrichment rods   : {n_low}  / {n_rods_total}
  Average enrichment    : {avg_enr:.4f} wt%
  k∞                    : {best_keff:.5f}
  k∞ target             : {cfg.k_target}
  |Δk∞|                 : {delta_k:.5f}  ({delta_k*1e5:.1f} pcm)
  PPF                   : {best_ppf:.4f}  (limit ≤ {ppf_target})
  PPF constraint        : {sat_str}
{perf_line}
{chrom_line}
======================================================================================
  {algorithm} PARAMETERS
======================================================================================
{algo_block}

======================================================================================
  GEOMETRY & MATERIAL PARAMETERS
======================================================================================
  Layout              : {n_rods_side} × {n_rods_side} = {n_rods_total} fuel pins
  Pin pitch           : {model_params['pin_pitch']} cm
  Assembly side       : {model_params['assembly_side']:.4f} cm
  Fuel radius         : {model_params['fuel_radius']} cm
  He gap outer (IR)   : {model_params['clad_ir']} cm
  Clad outer (OR)     : {model_params['clad_or']} cm
  T_fuel              : {model_params['t_fuel']} K
  T_moderator         : {model_params['t_mod']} K
  ρ UO₂               : {model_params['rho_uo2']} g/cm³
  ρ Coolant           : {model_params['rho_water']} g/cm³
  ρ Clad              : {model_params['rho_clad']} g/cm³

======================================================================================
  PER-PIN PARAMETER TABLE ({n_rods_total} pins)
======================================================================================
  Pin  Row  Col  x_ctr(cm)  y_ctr(cm)  Enr(wt%)  Material   Sym.pos
"""

    pin  = 1
    half = model_params["assembly_side"] / 2.0
    pp   = model_params["pin_pitch"]

    for r in range(n_rods_side):
        for c in range(n_rods_side):
            x_c  = -half + pp / 2.0 + c * pp
            y_c  =  half - pp / 2.0 - r * pp
            enr  = enr_grid_opt[r, c]
            mat  = "UO2_high" if round(enr, 4) == round(ENR_HIGH, 4) else "UO2_low "
            sym  = "   YES"   if c <= r else "mirror"
            spec += (f"  {pin:>3d}  {r:>3d}  {c:>3d}"
                     f"  {x_c:>9.3f}  {y_c:>9.3f}"
                     f"  {enr:>8.2f}  {mat:<9s}  {sym}\n")
            pin += 1

    spec += (f"\n  High-enr count check : {n_high}"
             f" (expected {cfg.n_high_rods})\n")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(spec)

    print(spec)
    print(f"\n{algorithm} report saved → {filepath}")
    return filepath


#Help***********************************#######################
'''
def _ga_params_block(cfg) -> str:
    lines = [
        f"  Population size       : {cfg.population_size}",
        f"  Generations           : {cfg.n_generations}",
        f"  Crossover rate        : {cfg.crossover_rate}",
        f"  Mutation rate         : {cfg.mutation_rate}"
        f"  ({cfg.n_swaps_per_mut} swap(s)/event)",
        f"  Tournament size       : {cfg.tournament_size}",
        f"  Elitism count         : {cfg.elitism_count}",
        f"  Random seed           : {cfg.random_seed}",
        f"  Corner masking        : {getattr(cfg, 'use_corner_masking', False)}"
        f"  (weight={getattr(cfg, 'corner_penalty_weight', 'N/A')})",
        f"  Adjacency penalty     : {getattr(cfg, 'use_adjacency_penalty', False)}"
        f"  (weight={getattr(cfg, 'adj_penalty_weight', 'N/A')})",
    ]
    return "\n".join(lines)
'''

def _bo_params_block(cfg) -> str:
    lines = [
        f"  LHS samples           : {cfg.n_initial_samples}",
        f"  BO iterations         : {cfg.n_bo_iterations}",
        f"  Candidate pool        : {cfg.n_candidates}",
        f"  EI xi                 : {cfg.ei_xi}",
        f"  GP restarts           : {cfg.gp_restarts}",
        f"  Random seed           : {cfg.random_seed}",
        f"  Corner masking        : {getattr(cfg, 'use_corner_masking', False)}",
        f"  Adjacency penalty     : {getattr(cfg, 'use_adjacency_penalty', False)}"
        f"  (weight={getattr(cfg, 'adj_penalty_weight', 'N/A')})",
    ]
    return "\n".join(lines)
