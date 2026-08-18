"""
env.py — BWR NxN OpenMC Modelling  
==================================

All geometry (lattice size, mesh, assembly side) is derived from cfg.n_rods_side at runtime. 
Changing BWR_N in the notebook propagates automatically here.

evaluate_for_ga(chrom, cfg) -> dict
evaluate_for_bo(x_cont, cfg) -> dict

"""

from __future__ import annotations

import contextlib
import glob
import io
import os

import numpy as np
import openmc

from .mapping    import ENR_LOW, ENR_HIGH


from .heuristics import count_adjacent_high_rods



# Geometry constants 

PIN_PITCH   = 1.62    # cm
FUEL_RADIUS = 0.53    # cm
CLAD_IR     = 0.54    # cm
CLAD_OR     = 0.61    # cm

T_FUEL = 900.0   # K
T_MOD  = 560.0   # K

RHO_UO2   = 10.40
RHO_WATER =  0.74
RHO_CLAD  =  6.56
RHO_HE    =  0.0015


def _assembly_side(n_rods_side: int) -> float:        ######## (cm)
    return n_rods_side * PIN_PITCH


def get_model_params(n_rods_side: int) -> dict:
    return dict(
        n_rods_side   = n_rods_side,
        pin_pitch     = PIN_PITCH,
        assembly_side = _assembly_side(n_rods_side),
        fuel_radius   = FUEL_RADIUS,
        clad_ir       = CLAD_IR,
        clad_or       = CLAD_OR,
        t_fuel        = T_FUEL,
        t_mod         = T_MOD,
        rho_uo2       = RHO_UO2,
        rho_water     = RHO_WATER,
        rho_clad      = RHO_CLAD,
    )




def clean_openmc_files() -> None:
    '''Deletes stale statepoint.*.h5 and summary.h5 artefacts.'''
    for pattern in ["statepoint.*.h5", "summary.h5"]:
        for path in glob.glob(pattern):
            try:
                os.remove(path)
            except OSError:
                pass


#************************ OPENMC CODE ***********************************************

# Geometry builder — NxN generalised 

def build_assembly_geometry(enr_grid: np.ndarray) -> tuple[
    openmc.Materials, openmc.Geometry, openmc.Tallies, float
]:
    '''
    Build the N×N BWR assembly geometry, materials, and tallies.

    Separated from the run logic so the geometry can be inspected or plotted independently without triggering an OpenMC simulation.

    '''
    n    = enr_grid.shape[0]
    half = _assembly_side(n) / 2.0

    # Materials 
    def _uo2(enr_pct, name):
        m = openmc.Material(name=name)
        m.set_density("g/cm3", RHO_UO2)
        m.add_nuclide("U235", float(enr_pct),          "wo")
        m.add_nuclide("U238", 100.0 - float(enr_pct),  "wo")
        m.add_element("O",    13.45,                    "wo")
        m.temperature = T_FUEL
        return m

    mat_lo  = _uo2(ENR_LOW,  "UO2_low")
    mat_hi  = _uo2(ENR_HIGH, "UO2_high")

    mat_he  = openmc.Material(name="He_gap")
    mat_he.set_density("g/cm3", RHO_HE)
    mat_he.add_element("He", 1.0, "ao")

    mat_zr  = openmc.Material(name="Zircaloy2")
    mat_zr.set_density("g/cm3", RHO_CLAD)
    mat_zr.add_element("Zr", 97.91, "wo")
    mat_zr.add_element("Sn",  1.50, "wo")
    mat_zr.add_element("Fe",  0.15, "wo")
    mat_zr.add_element("Cr",  0.10, "wo")
    mat_zr.add_element("Ni",  0.05, "wo")

    mat_h2o = openmc.Material(name="H2O")
    mat_h2o.set_density("g/cm3", RHO_WATER)
    mat_h2o.add_element("H", 2.0, "ao")
    mat_h2o.add_element("O", 1.0, "ao")
    mat_h2o.add_s_alpha_beta("c_H_in_H2O")
    mat_h2o.temperature = T_MOD

    materials = openmc.Materials([mat_lo, mat_hi, mat_he, mat_zr, mat_h2o])

    # Pin universes 
    s_fuel = openmc.ZCylinder(r=FUEL_RADIUS)
    s_gap  = openmc.ZCylinder(r=CLAD_IR)
    s_clad = openmc.ZCylinder(r=CLAD_OR)

    def _pin_universe(fuel_mat, tag):
        return openmc.Universe(cells=[
            openmc.Cell(fill=fuel_mat, region=-s_fuel,           name=f"{tag}_fuel"),
            openmc.Cell(fill=mat_he,   region=+s_fuel & -s_gap,  name=f"{tag}_gap"),
            openmc.Cell(fill=mat_zr,   region=+s_gap  & -s_clad, name=f"{tag}_clad"),
            openmc.Cell(fill=mat_h2o,  region=+s_clad,           name=f"{tag}_mod"),
        ])

    u_lo = _pin_universe(mat_lo, "lo")
    u_hi = _pin_universe(mat_hi, "hi")

    # Lattice 
    lat            = openmc.RectLattice(name=f"{n}x{n}_BWR_lattice")
    lat.pitch      = (PIN_PITCH, PIN_PITCH)
    lat.lower_left = (-half, -half)
    lat.universes  = [
        [
            u_hi if round(enr_grid[r, c], 2) == round(ENR_HIGH, 2) else u_lo
            for c in range(n)
        ]
        for r in range(n - 1, -1, -1)
    ]

    # Boundary planes & assembly cell 
    bnd    = dict(boundary_type="reflective")
    planes = [
        openmc.XPlane(-half, **bnd), openmc.XPlane(+half, **bnd),
        openmc.YPlane(-half, **bnd), openmc.YPlane(+half, **bnd),
        openmc.ZPlane(-10.0, **bnd), openmc.ZPlane(+10.0, **bnd),
    ]
    asm_region = (
        +planes[0] & -planes[1] & +planes[2] & -planes[3]
        & +planes[4] & -planes[5]
    )
    geometry = openmc.Geometry(
        [openmc.Cell(fill=lat, region=asm_region, name="assembly")]
    )

    # Tallies  
    mesh             = openmc.RegularMesh()
    mesh.dimension   = [n, n, 1]
    mesh.lower_left  = [-half, -half, -10.0]
    mesh.upper_right = [ half,  half,  10.0]
    tally            = openmc.Tally(name="pin_fission")
    tally.filters    = [openmc.MeshFilter(mesh)]
    tally.scores     = ["fission"]
    tallies          = openmc.Tallies([tally])

    return materials, geometry, tallies, half



# Core OpenMC runner 


def build_and_run_openmc(enr_grid: np.ndarray, cfg) -> dict:
    '''
    Build and run an N×N BWR eigenvalue calculation.

    '''
    n = cfg.n_rods_side

    clean_openmc_files()

    
    materials, geometry, tallies, half = build_assembly_geometry(enr_grid)

    # Settings
    settings            = openmc.Settings()
    settings.batches    = cfg.n_inactive + cfg.n_active
    settings.inactive   = cfg.n_inactive
    settings.particles  = cfg.n_particles
    settings.run_mode   = "eigenvalue"
    settings.source     = openmc.IndependentSource(
        space=openmc.stats.Box([-half, -half, -10.0], [half, half, 10.0])
    )
    settings.temperature = {"method": "interpolation", "tolerance": 1000.0}
    
    # Disable summary.h5 output for the optimization loop:
    # 1. Crash Prevention: Avoids 'Failed to open HDF5' lock errors if the OS suspends mid-run.
    # 2. Speed Boost: Eliminates heavy disk I/O overhead.
    settings.output      = {'summary': False} 

    # Run 
    model  = openmc.Model(geometry=geometry, materials=materials,
                          tallies=tallies, settings=settings)
    buf    = io.StringIO()
    with contextlib.redirect_stdout(buf):
        sp_path = model.run(output=False)

    #To check the error. 
    if sp_path is None:
        print("CRITICAL: OpenMC aborted! Here is the hidden error log:")
        print(buf.getvalue())
        raise RuntimeError("OpenMC simulation failed. Check the log above.")
        
    # Extract results 
    with openmc.StatePoint(sp_path) as sp:
        keff     = float(sp.keff.nominal_value)
        keff_std = float(sp.keff.std_dev)
        t        = sp.get_tally(name="pin_fission")
        fission  = t.mean.flatten()[:n * n]
        ppf      = (float(fission.max() / fission.mean())
                    if fission.sum() > 0 else 99.0)

    clean_openmc_files()
    return dict(keff=keff, keff_std=keff_std, ppf=ppf)



#***************************88 Optimiser wrappers — THE DEPENDENCY INJECTION BOUNDARY ***********************#############

#


def evaluate_for_bo(x_cont: np.ndarray, cfg) -> dict:
    '''
    BO wrapper — for LLM 
   
    '''
    enr_grid = cfg.decode(x_cont)
    res      = build_and_run_openmc(enr_grid, cfg)
    res["adj_high_rods"] = count_adjacent_high_rods(enr_grid)
    res["enr_grid"]      = enr_grid
    res["x_cont"]        = np.asarray(x_cont, dtype=float)
    return res