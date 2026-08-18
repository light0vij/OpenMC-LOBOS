"""
config.py — RunConfig for the BO enrichment-layout optimiser  for any NXN (Only 2 enrichments) 
-----------------------------------------------------------------------------------------------

Here if,
  cfg = RunConfig(NO: of side rods =6,  NO: enrichment rods =23)   for 6×6  i.e., 21 BO variables
 or if
  cfg = RunConfig(NO: of side rods =10,  NO: enrichment rods =63)   then 10×10 i.e., 55 BO variables


"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from typing import List, Optional, Tuple



ENR_LOW_DEFAULT  = 1.87   # wt% U-235
ENR_HIGH_DEFAULT = 2.53   # wt% U-235
N_RODS_SIDE_DEFAULT = 6



# Runtime geometry helpers


def make_sym_index(n_rods_side: int) -> List[Tuple[int, int]]:
    """
    Builds the unique (row, col) position list under 1/2-diagonal symmetry.
    Length = n_rods_side * (n_rods_side + 1) / 2
    """
    return [
        (r, c)
        for r in range(n_rods_side)
        for c in range(r + 1)
    ]


def decode_enrichment_vector(
    x_cont:      np.ndarray,
    sym_index:   List[Tuple[int, int]],
    n_rods_side: int,
    n_high:      int,
    enr_low:     float = ENR_LOW_DEFAULT,
    enr_high:    float = ENR_HIGH_DEFAULT,
    apply_mask:  bool  = True,
) -> np.ndarray:
    """
    Convert a continuous vector in [0, 1]^n_sym to a full N×N enrichment grid via argot inventory. 
    

    Works for any assembly size nxn. 

    Algorithm:- 
   
    - Partition the n_sym positions into diagonal (n_rods_side) and off-diagonal (n_sym - n_rods_side).
    - Find the (d, o) pair, where d + 2·o = n_high –––––> Highest total continuous score.
    - Assign high-enrichment to the top-d diagonal and top-o off-diagonal.
    - Mirror lower triangle to full N×N grid.

    Corner masking (apply_mask=True) –––––> Forces extreme low enrichemnt to corner positions.
    """
    x_cont   = np.array(x_cont, dtype=float, copy=True).ravel()
    n_sym    = len(sym_index)
    n_total  = n_rods_side ** 2

    if len(x_cont) != n_sym:
        raise ValueError(
            f"Expected {n_sym} continuous variables for a "
            f"{n_rods_side}×{n_rods_side} assembly, got {len(x_cont)}"
        )

    # Hard corner mask — force corners to always be low enrichment
    if apply_mask:
        last = n_rods_side - 1
        corners = {(0, 0), (0, last), (last, 0), (last, last)}
        for i, (r, c) in enumerate(sym_index):
            if (r, c) in corners or (c, r) in corners:
                x_cont[i] = -np.inf

    diag_idx    = [i for i, (r, c) in enumerate(sym_index) if r == c]
    offdiag_idx = [i for i, (r, c) in enumerate(sym_index) if r != c]

    diag_rank    = sorted(diag_idx,    key=lambda i: -x_cont[i])
    offdiag_rank = sorted(offdiag_idx, key=lambda i: -x_cont[i])

    n_d = len(diag_idx)
    n_o = len(offdiag_idx)

    best_score = -np.inf
    best_d = best_o = 0
    for d in range(n_d + 1):
        rem = n_high - d
        if rem < 0 or rem % 2 != 0:
            continue
        o = rem // 2
        if o > n_o:
            continue
        s = (
            sum(x_cont[diag_rank[k]]    for k in range(d)) +
            sum(x_cont[offdiag_rank[k]] for k in range(o))
        )
        if s > best_score:
            best_score = s
            best_d = d
            best_o = o

    sym_binary = np.zeros(n_sym, dtype=int)
    for k in range(best_d):
        sym_binary[diag_rank[k]] = 1
    for k in range(best_o):
        sym_binary[offdiag_rank[k]] = 1

    grid = np.zeros((n_rods_side, n_rods_side), dtype=int)
    for idx, (r, c) in enumerate(sym_index):
        grid[r, c] = sym_binary[idx]
        grid[c, r] = sym_binary[idx]

    return np.where(grid == 1, enr_high, enr_low)



#**************** RunConfig dataclass **********************####

@dataclass
class RunConfig:
    """
    Parameters for the BO enrichment-layout optimisation.

    To switch the assembly size pass n_rods_side=N, which must be provided by the user. 
    Other geometry parameters like n_vars, sym_index, etc. will be computed automatically. 
    
    """

    
    n_rods_side: int = N_RODS_SIDE_DEFAULT #Depends on assemblies and changes

    
    enr_low:     float = ENR_LOW_DEFAULT
    enr_high:    float = ENR_HIGH_DEFAULT
    n_high_rods: int   = 23   # MUST be set correctly for the chosen n_rods_side

    
    k_target:   float           = 1.25
    ppf_target: Optional[float] = 1.41
    alpha_ppf:  float           = 5.0
    alpha_k:    float           = 10.0

    
    use_corner_masking:    bool  = False
    use_adjacency_penalty: bool  = False
    adj_penalty_weight:    float = 2.0


    
    n_initial_samples: int   = 30
    n_bo_iterations:   int   = 20
    n_candidates:      int   = 80000
    ei_xi:             float = 0.01
    gp_restarts:       int   = 10
    random_seed:       int   = 42

   
    n_particles: int = 5000
    n_inactive:  int = 25
    n_active:    int = 50

    
    n_particles_smoke: int = 500
    n_inactive_smoke:  int = 10
    n_active_smoke:    int = 20

    
    model_name:  str = "bwr_nxn_bo"
    results_dir: str = "results"

    
    n_rods_total:       int                  = field(init=False, default=0)
    n_vars:             int                  = field(init=False, default=0)
    sym_index:          List[Tuple[int, int]] = field(init=False, default_factory=list)
    lower:              np.ndarray           = field(init=False, default=None)
    upper:              np.ndarray           = field(init=False, default=None)
    use_ppf_constraint: bool                 = field(init=False, default=False)

    def __post_init__(self):
        self.n_rods_total = self.n_rods_side ** 2
        self.sym_index    = make_sym_index(self.n_rods_side)
        self.n_vars       = len(self.sym_index)
        self.lower        = np.zeros(self.n_vars)
        self.upper        = np.ones(self.n_vars)
        self.use_ppf_constraint = (self.ppf_target is not None)

        assert 0 < self.n_high_rods < self.n_rods_total, (
            f"n_high_rods={self.n_high_rods} must be in "
            f"(0, {self.n_rods_total}) for a "
            f"{self.n_rods_side}×{self.n_rods_side} assembly"
        )
#*******************************************************************###
#****** Helpers*****************#################

   

    def decode(self, x_cont: np.ndarray) -> np.ndarray:
        
        return decode_enrichment_vector(
            x_cont,
            sym_index   = self.sym_index,
            n_rods_side = self.n_rods_side,
            n_high      = self.n_high_rods,
            enr_low     = self.enr_low,
            enr_high    = self.enr_high,
            apply_mask  = self.use_corner_masking,
        )

    def composite_score(self, keff: float, ppf: float = 0.0) -> float:
        """
        Base objective score (higher the better).

            score = − alpha_k · |k_target − k∞|− alpha_ppf · max(0, PPF − ppf_target)²          *********–––––– Composite Score

        The adjacency penalty (if cfg.use_adjacency_penalty) is applied
        separately in lhs_sampler.py / bo_loop.py after reading adj_high_rods
        from the evaluate_for_bo result dict.
        """
        k_penalty   = self.alpha_k * abs(self.k_target - keff)
        ppf_penalty = 0.0
        if self.use_ppf_constraint and self.ppf_target is not None:
            excess      = max(0.0, ppf - self.ppf_target)
            ppf_penalty = self.alpha_ppf * excess ** 2
        return float(-(k_penalty + ppf_penalty))

   

    def summary(self) -> str:
        avg_enr = (
            self.n_high_rods * self.enr_high
            + (self.n_rods_total - self.n_high_rods) * self.enr_low
        ) / self.n_rods_total
        lines = [
            f"RunConfig — {self.model_name}",
            f"  Assembly           : {self.n_rods_side}×{self.n_rods_side}"
            f" = {self.n_rods_total} pins",
            f"  Design variables   : {self.n_vars}  ",
            f"  ENR_LOW            : {self.enr_low} wt%",
            f"  ENR_HIGH           : {self.enr_high} wt%",
            f"  n_high_rods        : {self.n_high_rods} / {self.n_rods_total}",
            f"  k-inf target       : {self.k_target}  (alpha_k = {self.alpha_k})",
            f"  PPF target         : {self.ppf_target}  (alpha_ppf = {self.alpha_ppf})",
            f"  LHS samples        : {self.n_initial_samples}",
            f"  BO iterations      : {self.n_bo_iterations}",
            f"  Corner masking     : {self.use_corner_masking}",
            f"  Adjacency penalty  : {self.use_adjacency_penalty}"
            f"  (weight={self.adj_penalty_weight})",
            f"  OpenMC particles   : {self.n_particles}  "
            f"({self.n_inactive} inactive + {self.n_active} active)",
            f"  Smoke particles    : {self.n_particles_smoke}  "
            f"({self.n_inactive_smoke} inactive + {self.n_active_smoke} active)",
            f"  Results dir        : {self.results_dir}",
        ]
        return "\n".join(lines)



