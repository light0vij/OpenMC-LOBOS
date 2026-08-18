"""
bo_loop.py — Bayesian Optimisation loop 
----------------------------------------

This module is 100% pure algorithm (zero environment imports).

1. Setup (Caller/Notebook)
    environments.env  ––> Import evaluate_for_bo
    functools.partial ––> Bind config: evaluate_fn = partial(evaluate_for_bo, cfg=cfg_bo)

2. Injection
    evaluate_fn ––> Pass to module: run_bo(..., evaluate_fn)  

3. Execution 
    Input: x_cont (np.ndarray) ––> Call: evaluate_fn(x_cont)–––––––––––––––––––––––––––––––––––––> (from environments.env.py)
        ↓
    Output: ––> Returns dict
    ↳ Required Keys: {'keff', 'keff_std', 'ppf', 'adj_high_rods'}

4. Objective (maximisation of the composite score)

    score = −alpha_k·|k_target − k∞| − alpha_ppf·max(0, PPF − ppf_target)² − adj_penalty_weight·adj_high_rods    (if cfg.use_adjacency_penalty)



- The LLM orchestration is added on top of the existing BO model.
- After each evaluation, if an LLMBoundsOrchestrator is passed as llm_orchestrator, the LLM receives the adjacency count together with
  k_inf, PPF and the current GP uncertainty.
- The LLM then recommends how to adjust the BO search bounds for the next iteration.
- Passing llm_orchestrator=None will disable the LLM orchestration.
    

"""

from __future__ import annotations

import numpy as np
from tqdm.notebook import tqdm
from typing import Callable, Optional

from .config             import RunConfig
from .gp_surrogate        import train_gp, expected_improvement, gp_summary, predict
from .llm_orchestrator    import LLMBoundsOrchestrator

import warnings
from sklearn.exceptions import ConvergenceWarning
import time #––––––––––––––––––> to slow down the openmc evaluation

# Removes GP convergence warnings caused by argsort plateaus
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def run_bo(
    cfg: RunConfig,
    evaluate_fn: Callable[[np.ndarray], dict],
    llm_seed_data: dict,
    llm_orchestrator: Optional[LLMBoundsOrchestrator] = None,
    verbose: bool = True,
) -> dict:
    """
    Run the Bayesian Optimisation loop.

    
    With the LLM_orchestration, expected Improvement still mathematically selects the exact next sampling point — strictly inside whatever bounds are active
    for that iteration; the LLM never proposes a design point itself.
    
    """
    X             = llm_seed_data["X"].copy()
    keff          = llm_seed_data["keff"].copy()
    keff_std      = llm_seed_data["keff_std"].copy()
    ppf           = llm_seed_data["ppf"].copy()
    score         = llm_seed_data["score"].copy()
    base_score    = llm_seed_data.get("base_score", score.copy()).copy()
    adj_high_rods = llm_seed_data.get("adj_high_rods", np.zeros_like(score)).copy()
    
    n_seed = len(X) 
    

    
    adj_weight = cfg.adj_penalty_weight

    # Initial GP fit
    gp, scaler = train_gp(X, score, cfg)
    if verbose:
        print(gp_summary(gp, cfg))

    # The GP models the continuous score landscape and never sees the binary grid directly.
    rng = np.random.default_rng(cfg.random_seed + 1)

    '''
    Active search bounds for candidate generation — start at the full, static config bounds; the LLM orchestrator may narrow/shift these 
    each iteration once results start coming in.
    '''
    active_lower = cfg.lower.copy()
    active_upper = cfg.upper.copy()

    bo_log = []
    llm_log = []

   
    llm_active = llm_orchestrator is not None

    if verbose:
        print(f"\nRunning {cfg.n_bo_iterations} BO iterations …")
        print(f"Objective : − {cfg.alpha_k}·|{cfg.k_target} − k∞|"
              f" − {cfg.alpha_ppf}·max(0, PPF − {cfg.ppf_target})²"
              + ("  − adj_weight·adj_high_rods" if cfg.use_adjacency_penalty else ""))
        print(f"Adjacency penalty (score term)  : {'ENABLED' if cfg.use_adjacency_penalty else 'disabled'}"
              f"  (weight={adj_weight})")
        if llm_active:
            print(f"LLM-guided bounds orchestration : ENABLED (provider={llm_orchestrator.provider})")
        else:
            print("LLM bounds : disabled")
        if cfg.use_adjacency_penalty and llm_active:
            print("--> HYBRID mode: adjacency penalty + LLM prompt.\n")
        elif llm_active and not cfg.use_adjacency_penalty:
            print("--> LLM-ONLY: no score-level penalty;")
        else:
            print()

    for i in tqdm(range(cfg.n_bo_iterations),
                  desc="BO iterations", disable=not verbose):

        
        X_cand = rng.uniform(active_lower, active_upper,
                             size=(cfg.n_candidates, cfg.n_vars))

        
        ei     = expected_improvement(X_cand, gp, scaler,
                                      y_best=score.max(), xi=cfg.ei_xi)
        x_next = X_cand[np.argmax(ei)].copy()

        
        res  = evaluate_fn(x_next)
        k    = float(res["keff"])
        kstd = float(res.get("keff_std", 0.0))
        p    = float(res.get("ppf", 0.0))
        adj  = float(res.get("adj_high_rods", 0))

        base_s = cfg.composite_score(k, p)  # base composite score (k_target / ppf_target only)

        #*************** Applies adjacency penalty using the fixed weight ***********#########
        if cfg.use_adjacency_penalty:
            s = base_s - adj_weight * adj
        else:
            s = base_s
        #********************************************##############

        X             = np.vstack([X, x_next])
        keff          = np.append(keff,          k)
        keff_std      = np.append(keff_std,      kstd)
        ppf           = np.append(ppf,           p)
        base_score    = np.append(base_score,    base_s)
        adj_high_rods = np.append(adj_high_rods, adj)
        score         = np.append(score,         s)

        # Retrain GP on the newly observed point
        gp, scaler = train_gp(X, score, cfg)

        feas  = (p <= cfg.ppf_target) if cfg.ppf_target is not None else True
        entry = dict(
            iter         = i + 1,
            x            = x_next,
            keff         = k,
            keff_std     = kstd,
            ppf          = p,
            adj_high_rods= adj,
            adj_weight_used = adj_weight,
            score        = s,
            feasible     = feas,
            best_score   = score.max(),
            delta_k      = abs(cfg.k_target - k),
            bounds_lower = active_lower.copy(),
            bounds_upper = active_upper.copy(),
        )
        bo_log.append(entry)

        if verbose:
            _print_bo_iter(cfg, entry, n_seed + i + 1)

            '''
             LLM orchestrator step — ask the LLM to recommend
            bounds for the next candifdate model and parse its JSON reply. Never crashes the loop: on any failure it falls
            back to a safe default. 
            '''
        
        if llm_active:
            mu_cand, sigma_cand = predict(X_cand, gp, scaler)
            gp_uncertainty = float(np.mean(sigma_cand))

          #  time.sleep(20)    #––––––––––––––––––––––––-> to slow down the Openmc run since i was exceeding my gemini quota as i'm not using a paid version
            suggestion = llm_orchestrator.suggest_bounds(
                last_result    = res,   # includes 'adj_high_rods' — see llm_orchestrator._build_prompt
                gp_uncertainty = gp_uncertainty,
                best_score     = float(score.max()),
                current_lower  = active_lower,
                current_upper  = active_upper,
                iteration      = i + 1,
            )
            active_lower = suggestion.lower
            active_upper = suggestion.upper
            llm_log.append(dict(
                model_name    = llm_orchestrator.provider_settings["model"],
                iter          = i + 1,
                mode          = suggestion.mode,
                reasoning     = suggestion.reasoning,
                used_fallback = suggestion.used_fallback,
                lower         = active_lower.copy(),
                upper         = active_upper.copy(),
            ))

            if verbose:
                tag = "FALLBACK" if suggestion.used_fallback else "LLM"
                print(f"      [{tag}] bounds updated — {suggestion.reasoning}")

    if verbose:
        _print_bo_summary(cfg, keff[n_seed:], ppf[n_seed:], score[n_seed:])

    return dict(
        X             = X,
        keff          = keff,
        keff_std      = keff_std,
        ppf           = ppf,
        score         = score,
        base_score    = base_score,
        adj_high_rods = adj_high_rods,
        adj_weight    = adj_weight,
        bo_log   = bo_log,
        llm_log  = llm_log,
        gp       = gp,
        scaler   = scaler,
        n_seed    = n_seed,
        llm_model     = llm_orchestrator.provider_settings["model"] if llm_active else "None"    #llm model and provider details. 
    )

###************** SMOKE TEST BELOW ********************###############

def run_smoke_test(
    cfg: RunConfig,
    evaluate_fn: Callable[[np.ndarray], dict],
    smoke_x: np.ndarray,
    smoke_label: Optional[str] = None,
) -> dict:
    
    # Store original values to restore after test
    _prod = (cfg.n_particles, cfg.n_inactive, cfg.n_active)
    cfg.n_particles = cfg.n_particles_smoke
    cfg.n_inactive  = cfg.n_inactive_smoke
    cfg.n_active    = cfg.n_active_smoke

    label    = smoke_label or cfg.model_name
    enr_grid = cfg.decode(smoke_x)
    avg_enr  = enr_grid.mean()
    n_high   = int((enr_grid == cfg.enr_high).sum())

    print(f"  SMOKE TEST — {label}")
    print("═" * 70)
    print(f"  Continuous vector (first 6): {np.round(smoke_x[:6], 3)}")
    
    # FIXED: Replaced hardcoded '36' with cfg.n_rods_total
    print(f"  Decoded grid       : {n_high} × {cfg.enr_high}%  +  "
          f"{cfg.n_rods_total - n_high} × {cfg.enr_low}%  "
          f"(avg {avg_enr:.4f} wt%)")
    
    print(f"  Particles          : {cfg.n_particles}  "
          f"({cfg.n_inactive} inactive + {cfg.n_active} active)")
    print()
    
    smoke_res = evaluate_fn(smoke_x)

    k    = float(smoke_res["keff"])
    kstd = float(smoke_res.get("keff_std", 0.0))
    ppf  = float(smoke_res.get("ppf", 0.0))

    std_str = f" ± {kstd:.5f}" if kstd > 0 else ""
    print(f"  k∞    = {k:.5f}{std_str}   (target = {cfg.k_target})")
    print(f"  |Δk∞| = {abs(cfg.k_target - k):.5f}")
    if cfg.ppf_target is not None:
        feas_str = " feasible" if ppf <= cfg.ppf_target else " infeasible"
        print(f"  PPF   = {ppf:.4f}   (limit ≤ {cfg.ppf_target})  {feas_str}")

    # Restore original settings
    cfg.n_particles, cfg.n_inactive, cfg.n_active = _prod
    print("=" * 70)
    return smoke_res


#**************** Helpers ******************


def _print_bo_iter(cfg, entry, total_idx):
    ppf_str = (f"  PPF={entry['ppf']:.3f}"
               if cfg.ppf_target is not None else "")
    adj_str = (f"  adj={entry['adj_high_rods']:.0f}(w={entry['adj_weight_used']:.2f})"
               if cfg.use_adjacency_penalty else "")
    std_str = (f" ±{entry['keff_std']:.5f}"
               if entry["keff_std"] > 0 else "")
    print(f"  BO {entry['iter']:2d} [total={total_idx}]:  "
          f"k∞={entry['keff']:.5f}{std_str}  |Δk|={entry['delta_k']:.5f}"
          f"{ppf_str}{adj_str}  "
          f"score={entry['score']:.5f}  best={entry['best_score']:.5f}")


def _print_bo_summary(cfg, keff_bo, ppf_bo, score_bo):
    best_i = int(np.argmax(score_bo))
    print(f"\n── BO summary {'─'*50}")
    print(f"BO iterations    : {len(keff_bo)}")
    print(f"k∞ range (BO)    : [{keff_bo.min():.5f}, {keff_bo.max():.5f}]")
    print(f"k∞ target        : {cfg.k_target}")
    print(f"Best |Δk∞|       : {abs(cfg.k_target - keff_bo[best_i]):.5f}")
    if cfg.ppf_target is not None:
        feas = (ppf_bo <= cfg.ppf_target).sum()
        print(f"Feasible (BO)    : {feas} / {len(keff_bo)}")
    print(f"Best score (BO)  : {score_bo.max():.5f}")
