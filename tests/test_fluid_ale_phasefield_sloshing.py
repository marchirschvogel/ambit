#!/usr/bin/env python3

"""
Two-phase flow in tank that undergoes shear motion
"""

import ambit_fe

import sys
import numpy as np
from pathlib import Path
import pytest


@pytest.mark.fluid_phasefield
def test_main():
    basepath = str(Path(__file__).parent.absolute())

    # reads in restart step from the command line
    try:
        restart_step = int(sys.argv[1])
    except:
        restart_step = 0

    IO_PARAMS = {
        "problem_type": "fluid_ale_phasefield",
        "write_results_every": 1,
        "write_restart_every": 4,
        "indicate_results_by": "time",
        "restart_step": restart_step,
        "output_path": basepath + "/tmp/",
        "mesh_domain": {"type":"rectangle", "celltype":"quadrilateral", "coords_a":[0.0, 0.0], "coords_b":[2.0, 1.0], "meshsize":[32,16]},
        "results_to_write": [["velocity", "pressure", "cauchystress"],["phase", "potential"],["aledisplacement"]],
        "simname": "fluid_ale_phasefield_sloshing",
        "write_initial_fields": True,
        "report_conservation_properties": True,
    }

    h = 2.0/IO_PARAMS["mesh_domain"]["meshsize"][0]
    eps = 1.28*h

    class expr1:
        def __init__(self):
            self.t = 0
            self.eps = 1e0
            self.h_0 = 0.6

            self.x_c = np.asarray([0.5, 0.5, 0.0])

        def evaluate(self, x):

            d = x[1]

            val = 0.5*(1.0 + np.tanh((self.h_0 - d)/(np.sqrt(2.0)*eps)))
            return (
                np.full(x.shape[1], val),
            )

    CONTROL_PARAMS = {"maxtime": 3.0,
                      "dt": 0.004,
                      "numstep_stop": 5,
                      "initial_fields": [expr1, None],
                      }

    SOLVER_PARAMS = {
        "solve_type": "direct",
        "direct_solver": "mumps",
        "maxiter":10,
        "tol_res": [1e-4, 1e-4, 1e-4, 1e-4, 1e-4],
        "tol_inc": [1e-4, 1e16, 1e-4, 1e-4, 1e-4],
        "divergence_continue": "PTC",
        "k_ptc_initial": 100.0,
        "catch_max_inc_value": 1e12,
    }

    TIME_PARAMS_FLUID = {"timint": "ost",
                         "theta_ost": 0.5,
                         "fluid_governing_type": "navierstokes_transient",
                         "eval_nonlin_terms": "midpoint", # midpoint, trapezoidal
                         "continuity_at_midpoint": True} # Should use midpoint if time derivative (drho/dt) is involved...}

    TIME_PARAMS_PF = {"timint": "ost",
                      "theta_ost": 0.5,
                      "eval_nonlin_terms": "midpoint", # midpoint, trapezoidal
                      "potential_at_midpoint": True}


    FEM_PARAMS_FLUID = {"order_vel": 2,
                        "order_pres": 1,
                        "quad_degree": 5,
                        "fluid_formulation": "conservative"}

    FEM_PARAMS_PF = {"order_phi": 2, "order_mu": 2, "quad_degree": 5}

    FEM_PARAMS_ALE = {"order_disp": 2, "quad_degree": 5}

    # all boundary walls
    class locate:
        def evaluate(self, x):
            left_b = np.isclose(x[0], 0.0)
            right_b = np.isclose(x[0], 2.0)
            top_b = np.isclose(x[1], 1.0)
            bottom_b = np.isclose(x[1], 0.0)
            return np.logical_or(np.logical_or(left_b, right_b), np.logical_or(top_b, bottom_b))

    COUPLING_PARAMS_FLUID_ALE = {
        "coupling_ale_fluid": {"locator": locate()}, # no-slip at moving ALE boundary
    }

    # fluid1 water, fluid2 oil
    rho1 = 1000.0
    rho2 = 700.0
    eta1 = 0.001
    eta2 = 0.1
    sig = 100.0
    M0 = 1e-3

    MATERIALS_FLUID = {"MAT1": {"newtonian": {"eta1": eta1, "eta2": eta2},
                                "inertia": {"rho1": rho1, "rho2": rho2}}}

    MATERIALS_PF = {"MAT1": {"mat_cahnhilliard": {"M0": M0, "D": sig/eps, "kappa": sig*eps}}}

    MATERIALS_ALE = {"MAT1": {"linelast": {"Emod": 10.0, "nu": 0.3}}}

    class expr2:
        def __init__(self):
            self.t = 0.0
            self.L = 1.0
            self.A = 0.2

            T=1.0
            self.omega=2.*np.pi/T

        def evaluate(self, x):
            val_t = self.A*np.sin(self.omega*self.t)*x[1]/self.L

            return (np.full(x.shape[1], val_t),
                    np.full(x.shape[1], 0.0))

    class locate_all:
        def evaluate(self, x):
            return np.full(x.shape[1], True, dtype=bool)

    BC_DICT_FLUID = { "bodyforce" : [{"locator": locate_all(), "dir": [0.0, -1.0, 0.0], "val": 0.98, "scale_density": True}] }

    BC_DICT_ALE = {
        "dirichlet": [{"dir": "all", "expression": expr2, "codimension": 2}]
    }


    BC_DICT_PF = { }


    # problem setup
    problem = ambit_fe.ambit_main.Ambit(
        IO_PARAMS,
        CONTROL_PARAMS,
        [TIME_PARAMS_FLUID, TIME_PARAMS_PF],
        SOLVER_PARAMS,
        [FEM_PARAMS_FLUID, FEM_PARAMS_PF, FEM_PARAMS_ALE],
        [MATERIALS_FLUID, MATERIALS_PF, MATERIALS_ALE],
        [BC_DICT_FLUID, BC_DICT_PF, BC_DICT_ALE],
        coupling_params=COUPLING_PARAMS_FLUID_ALE,
    )

    # problem solve
    problem.solve_problem()

    # --- results check
    tol = 1.0e-6

    check_node = []
    check_node.append(np.array([1.0, 0.625, 0.0]))

    v_corr, p_corr = np.zeros(2 * len(check_node)), np.zeros(len(check_node))
    phi_corr, mu_corr = np.zeros(len(check_node)), np.zeros(len(check_node))

    # correct results
    v_corr[0] = 1.3422876769067809E+00  # x
    v_corr[1] = 7.7342515736798442E-04  # y

    p_corr[0] = 1.8156696641083278E+07

    phi_corr[0] = 3.9113503367268654E-01
    mu_corr[0] = -1.4614520536619657E-01

    check1 = ambit_fe.resultcheck.results_check_node(
        problem.mp.pbf.v,
        check_node,
        v_corr,
        problem.mp.pbf.V_v,
        problem.mp.comm,
        tol=tol,
        nm="v",
        readtol=1e-4,
    )
    # check2 = ambit_fe.resultcheck.results_check_node(
    #     problem.mp.pbf.p,
    #     check_node,
    #     p_corr,
    #     problem.mp.pbf.V_p,
    #     problem.mp.comm,
    #     tol=tol,
    #     nm="p",
    #     readtol=1e-4,
    # )
    check3 = ambit_fe.resultcheck.results_check_node(
        problem.mp.pbp.phi,
        check_node,
        phi_corr,
        problem.mp.pbp.V_phi,
        problem.mp.comm,
        tol=tol,
        nm="phi",
        readtol=1e-4,
    )
    check4 = ambit_fe.resultcheck.results_check_node(
        problem.mp.pbp.mu,
        check_node,
        mu_corr,
        problem.mp.pbp.V_mu,
        problem.mp.comm,
        tol=tol,
        nm="mu",
        readtol=1e-4,
    )

    success = ambit_fe.resultcheck.success_check([check1, check3, check4], problem.mp.comm)

    if not success:
        raise RuntimeError("Test failed!")



if __name__ == "__main__":
    test_main()
