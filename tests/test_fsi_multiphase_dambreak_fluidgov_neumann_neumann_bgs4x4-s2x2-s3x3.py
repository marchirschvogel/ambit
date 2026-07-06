#!/usr/bin/env python3

"""
Two-phase flow FSI simulation of a collapsing water column against an elastic obstacle
Neumann-Neumann formulation (with Lagrange multiplier)
BDF2 time-integration scheme for both fluid and phasefield
Outer BGS4x4(S2x2-S3x3) preconditioner
"""

import ambit_fe

import sys
import numpy as np
from pathlib import Path
import pytest


@pytest.mark.fsi
@pytest.mark.fluid_solid
def test_main():
    basepath = str(Path(__file__).parent.absolute())

    IO_PARAMS = {
        "problem_type": "fsi_multiphase",
        "write_results_every": 1,
        "write_restart_every": -1,
        "indicate_results_by": "time",
        "restart_step": 0,
        "output_path": basepath + "/tmp/",
        "mesh_domain": basepath + "/input/dam-break_domain.xdmf",
        "mesh_boundary": basepath + "/input/dam-break_boundary.xdmf",
        "mesh_encoding": "ASCII",
        "results_to_write": {"solid": ["displacement"],
                             "fluid": ["velocity", "pressure", "density"],
                             "phasefield": ["phase", "potential"],
                             "ale": ["aledisplacement"]},
        "simname": "fsi_multiphase_dambreak",
        "write_counters": True,
        "write_initial_fields": True,
        "report_conservation_properties": True,
    }

    L = 0.146
    eps = 1e-2
    class expr1:
        def __init__(self):
            pass
        def evaluate(self, x):
            d = np.minimum(L-x[0], 2*L-x[1])
            # d = L-x[0]
            val = np.tanh(d / (np.sqrt(2.0) * eps))
            return (
                np.full(x.shape[1], val),
            )

    # print(0.2*np.sqrt(1.0 * (eps)**3 / 72.8e-3))

    CONTROL_PARAMS = {"maxtime": 5.,
                      "dt": 1e-3,
                      "numstep_stop": 3,
                      "initial_fields": [expr1, None],
                      }

    SOLVER_PARAMS = {
        "solve_type": "iterative",  # direct, iterative
        "direct_solver": "mumps",
        # BEGIN: Settings for iterative solver
        "iterative_solver": "fgmres",
        "petsc_options_ksp": {"ksp_gmres_modifiedgramschmidt": True, "ksp_gmres_restart": 1000},
        "block_precond": "BGS_outer",
        "precond_fields": [{"prec": "amg", "blocks": [0]},  # solid-u
                           {"prec": "amg", "blocks": [6]},  # ale-d
                           {"prec": {"s2x2": [{"prec": "amg"},
                                              {"prec": "amg"}]},
                                              "blocks": [4,3]},  # CH-phi,mu
                           {"prec": {"s3x3": [{"prec": "amg"},
                                              {"prec": "amg"},
                                              {"prec": "amg"}]},
                                               "blocks": [1,2,5]},  # fluid-v,p,lm
                           ],
        "indexset_options": {"merge_prec_mat": True},  # currently needed, if index sets do not align with the nested mat structure (e.g. if requesting [1,2,5] blocks in s3x3)
        "tol_lin_rel": 1e-5,
        "tol_lin_abs": 1e-8,
        "lin_norm_type": "unpreconditioned",
        "print_liniter_every": 50,
        "max_liniter": 500,
        "res_lin_monitor": "rel",
        # END: Settings for iterative solver
        "maxiter": 25,
        "tol_res": 1e-4,
        "tol_inc": 1e-3,
        # "divergence_continue": "ptc",
        "k_ptc_initial": 1e2,
        "ptc_field": 1, # on fluid
        "ptc_maxiter": 100,
    }

    TIME_PARAMS_SOLID = {"timint": "genalpha", "rho_inf_genalpha": 0.8, "eval_nonlin_terms": "trapezoidal"}

    TIME_PARAMS_FLUID = {"timint": "bdf2",
                         "theta_ost": 1., # Not used: only for OST scheme
                         "fluid_governing_type": "navierstokes_transient",
                         "continuity_at_midpoint": True} # Not relevant when using BDF2 scheme

    TIME_PARAMS_PF = {"timint": "bdf2",
                      "theta_ost": 1., # Not used: only for OST scheme
                      "potential_at_midpoint": False} # Not relevant when using BDF2 scheme

    FEM_PARAMS_SOLID = {
        "order_disp": 1,
        "order_pres": 1,
        "quad_degree": 5,
        "incompressibility": "no",
    }

    FEM_PARAMS_FLUID = {"order_vel": 1,
                        "order_pres": 1,
                        "quad_degree": 5,
                        "fluid_formulation": "conservative",
                        "mass_formulation": "reduced_mass",  # conservative_mass, reduced_mass
                        "stabilization": {"scheme": "supg_pspg",
                                          "vscale": 1e1, # increasing this cranks up LSIC ("grad/div") stab, while lowers SUPG/PSPG - too high breaks interface!!
                                          "dscales": [1.0, 1.0, 1.0],
                                          "reduced_scheme": True,
                                          "symmetric": False,
                                        }}

    FEM_PARAMS_PF = {"order_phi": 1, "order_mu": 1, "quad_degree": 5, "phi_range": [-1.0, 1.0]}

    FEM_PARAMS_ALE = {"order_disp": 1, "quad_degree": 5}

    COUPLING_PARAMS_FSI = {
        "coupling_fsi": {"interface": [3,4,5]},
        "fsi_system": "neumann_neumann",  # neumann_neumann, neumann_dirichlet
    }

    COUPLING_PARAMS_MULTIPHASE = {"capillary_force_from_korteweg_stress": True,
                                  "clip_phi_range": True,
                                  "smooth_clip": "cubic", # cubic, quintic, cos
                                  "epsilon_clip": 1e-1} # 1e-2

    # fluid1 is air (density x 100), fluid2 is water
    rho1 = 1e2
    rho2 = 1e3
    eta1 = 1.81e-5
    eta2 = 1e-3
    sig = 72.8e-3 # N/m - value for water-air
    M0 = 0.1*eps**2.0  # keep the pre-factor lower than 1 ...

    sigtilde = 3.*sig/(2.*np.sqrt(2.))

    class time_curves:
        def tc1(self, t):
            return 9.81

    class locate_fluid_solid_corner:
        def evaluate(self, x):
            p1 = np.logical_and(np.isclose(x[0], 0.292), np.isclose(x[1], 0.0))
            p2 = np.logical_and(np.isclose(x[0], 0.304), np.isclose(x[1], 0.0))
            return np.logical_or(p1, p2)

    E, nu = 1e6, 0.0
    MATERIALS_SOLID = {"MAT1": {"neohooke_dev": {"mu": E/3.},
                                "ogden_vol": {"kappa": E/(3.*(1.-2.*nu))},
                                "inertia": {"rho0": 2.5e3},
                                "bodyforce": {"dir": [0.0, -1.0, 0.0], "curve": 1, "scale_density": True},
                                "id": 2}}

    MATERIALS_FLUID = {"MAT1": {"newtonian": {"eta1": eta1, "eta2": eta2},
                                "inertia": {"rho1": rho1, "rho2": rho2}, "id": 1,
                                "bodyforce": {"dir": [0.0, -1.0, 0.0], "curve": 1, "scale_density": True}}}

    MATERIALS_PF = {"MAT1": {"mat_cahnhilliard": {"M0": M0,
                                                  "D": sigtilde/(4.*eps),
                                                  "kappa": sigtilde*eps,
                                                  "mobility": "degenerate"}, "id": 1}}

    MATERIALS_ALE = {"MAT1": {"exponential": {"a_0": 1.0, "b_0": 10.0, "kappa": 1e2}, "id": 1}}

    BC_DICT_SOLID = {
        "dirichlet" : [{"id": [9], "dir": "all", "val": 0.0}]
    }

    BC_DICT_FLUID = {
        "dirichlet" : [
                       {"id": [2,6], "dir": "y", "val": 0.0},  # slip
                       {"id": [1,7], "dir": "x", "val": 0.0},  # slip
                       {"id": [locate_fluid_solid_corner()], "dir": "all", "val": 0.0},  # needed if walls are slip
                       ],
        "stabilized_neumann" : [{"id": [8], "beta": 0.2*rho1, "gamma": 1.}]
    }

    theta_wall_b = np.pi/2.
    BC_DICT_PF = { "robin_flux" : [{"id": [8], "phi0": -1.0, "c1": 1e3}]}  # always air at top

    BC_DICT_ALE = {
        "dirichlet" : [{"id": [1,2,6,7,8], "dir": "all", "val": 0.0}]
    }

    BC_DICT_LM = {"dirichlet": [{"id": [locate_fluid_solid_corner()], "dir": "all", "val": 0.0}]}


    # problem setup
    problem = ambit_fe.ambit_main.Ambit(
        IO_PARAMS,
        CONTROL_PARAMS,
        [[TIME_PARAMS_SOLID], [TIME_PARAMS_FLUID], [TIME_PARAMS_PF]],
        SOLVER_PARAMS,
        [[FEM_PARAMS_SOLID], [FEM_PARAMS_FLUID], [FEM_PARAMS_PF], [FEM_PARAMS_ALE]],
        [[MATERIALS_SOLID], [MATERIALS_FLUID], [MATERIALS_PF], [MATERIALS_ALE]],
        [[BC_DICT_SOLID], [BC_DICT_FLUID], [BC_DICT_PF], [BC_DICT_ALE], [BC_DICT_LM]],
        time_curves=time_curves(),
        coupling_params=[COUPLING_PARAMS_FSI,COUPLING_PARAMS_MULTIPHASE],
    )

    # problem solve
    problem.solve_problem()

    # --- results check
    tol = 1.0e-8

    check_node = []
    check_node.append(np.array([0.292, 0.08, 0.]))

    u_corr, v_corr = (
        np.zeros(2 * len(check_node)),
        np.zeros(2 * len(check_node)),
    )

    # correct results
    u_corr[0] = 6.7821621543757800E-06  # x
    u_corr[1] = -6.2847293228190552E-05  # y

    v_corr[0] = 6.5011753141286453E-04  # x
    v_corr[1] = -3.9252357028610879E-02  # y

    check1 = ambit_fe.resultcheck.results_check_node(
        problem.mp.pbs.u,
        check_node,
        u_corr,
        problem.mp.pbs.V_u,
        problem.mp.comm,
        tol=tol,
        nm="u",
        readtol=1e-4,
    )
    check2 = ambit_fe.resultcheck.results_check_node(
        problem.mp.pbf.v,
        check_node,
        v_corr,
        problem.mp.pbf.V_v,
        problem.mp.comm,
        tol=tol,
        nm="v",
        readtol=1e-4,
    )

    success = ambit_fe.resultcheck.success_check([check1, check2], problem.mp.comm)

    if not success:
        raise RuntimeError("Test failed!")



if __name__ == "__main__":
    test_main()
