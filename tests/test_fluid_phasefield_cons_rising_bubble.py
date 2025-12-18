#!/usr/bin/env python3

"""
Two-phase flow rising bubble example
"""

import ambit_fe

import numpy as np
from pathlib import Path
import pytest


@pytest.mark.fluid_phasefield
@pytest.mark.skip(reason="Not yet ready for testing.")
def test_main():
    basepath = str(Path(__file__).parent.absolute())

    # cases (1,2) from Eikelder et al. (2024)
    case = 2

    IO_PARAMS = {
        "problem_type": "fluid_phasefield",
        "write_results_every": 1,
        "indicate_results_by": "time",
        "output_path": basepath + "/tmp/",
        "mesh_domain": {"type":"rectangle", "celltype":"quadrilateral", "coords_a":[0.0, 0.0], "coords_b":[1.0, 2.0], "meshsize":[32,64]}, # 32,64
        "results_to_write": [["velocity", "pressure", "cauchystress"],["phase", "potential"]],
        "simname": "fluid_phasefield_cons_rising_bubble"+str(case),
        "write_initial_fields": True,
        "report_conservation_properties": True,
    }

    h = 1.0/IO_PARAMS["mesh_domain"]["meshsize"][0]
    eps = 1.28*h

    class expr1:
        def __init__(self):
            self.t = 0
            self.R_0 = 0.25
            self.x_c = np.asarray([0.5, 0.5, 0.0])

        def evaluate(self, x):
            d = np.sqrt( (x[0]-self.x_c[0])**2.0 + (x[1]-self.x_c[1])**2.0 + (x[2]-self.x_c[2])**2.0 )
            val = 0.5*(1.0 + np.tanh((self.R_0 - d)/(np.sqrt(2.0)*eps)))
            return (
                np.full(x.shape[1], val),
            )

    CONTROL_PARAMS = {"maxtime": 3.0,
                      "dt": 0.128*h, # from Eikelder et al. (2024)
                      # "numstep_stop": 5,
                      "initial_fields": [expr1, None],
                      }

    SOLVER_PARAMS = {
        "solve_type": "direct",
        "direct_solver": "mumps",
        "maxiter":10,
        "tol_res": [1e-8, 1e-8, 1e-8, 1e-8],
        "tol_inc": [1e-8, 1e16, 1e-8, 1e-8],
    }

    TIME_PARAMS_FLUID = {"timint": "ost", "theta_ost": 0.5,
                         "fluid_governing_type": "navierstokes_transient",
                         "eval_nonlin_terms": "trapezoidal", # midpoint, trapezoidal
                         "continuity_at_midpoint": True} # Should use midpoint if time derivative (drho/dt) is involved...
    TIME_PARAMS_PF = {"timint": "ost",
                      "theta_ost": 0.5,
                      "eval_nonlin_terms": "trapezoidal", # midpoint, trapezoidal
                      "potential_at_midpoint": False}


    FEM_PARAMS_FLUID = {"order_vel": 2,
                        "order_pres": 1,
                        "quad_degree": 5,
                        "fluid_formulation": "conservative"}

    FEM_PARAMS_PF = {"order_phi": 2, "order_mu": 2, "quad_degree": 5}

    # fluid1 is bubble, fluid2 is surrounding
    # TODO: How is M chosen in paper?
    if case==1:
        rho1 = 100.0
        rho2 = 1000.0
        eta1 = 1.0
        eta2 = 10.0
        sig = 24.5
        M = 1e-3
    elif case==2:
        rho1 = 1.0
        rho2 = 1000.0
        eta1 = 0.1
        eta2 = 1.0
        sig = 1.96
        M = 1e-3
    else:
        raise ValueError("Unknown case.")

    MATERIALS_FLUID = {"MAT1": {"newtonian": {"mu1": eta1, "mu2": eta2},
                                "inertia": {"rho1": rho1, "rho2": rho2}}}


    MATERIALS_PF = {"MAT1": {"mat_cahnhilliard": {"D": sig/eps},
                          "params_cahnhilliard": {"M": M, "lambda": sig*eps}}}

    class locate_top_bottom:
        def evaluate(self, x):
            top_b = np.isclose(x[1], 2.0)
            bottom_b = np.isclose(x[1], 0.0)
            return np.logical_or(top_b, bottom_b)

    class locate_left_right:
        def evaluate(self, x):
            left_b = np.isclose(x[0], 0.0)
            right_b = np.isclose(x[0], 1.0)
            return np.logical_or(left_b, right_b)

    class locate_all:
        def evaluate(self, x):
            return np.full(x.shape[1], True, dtype=bool)

    BC_DICT_FLUID = {
        "dirichlet" : [{"locator": locate_top_bottom(), "dir": "all", "val": 0.0},
                       {"locator": locate_left_right(), "dir": "x", "val": 0.0}],
        "bodyforce" : [{"locator": locate_all(), "dir": [0.0, -1.0, 0.0], "val": 0.98, "scale_density": True}],
    }

    BC_DICT_PF = { }


    # problem setup
    problem = ambit_fe.ambit_main.Ambit(
        IO_PARAMS,
        CONTROL_PARAMS,
        [TIME_PARAMS_FLUID, TIME_PARAMS_PF],
        SOLVER_PARAMS,
        [FEM_PARAMS_FLUID, FEM_PARAMS_PF],
        [MATERIALS_FLUID, MATERIALS_PF],
        [BC_DICT_FLUID, BC_DICT_PF],
        # time_curves=time_curves(),
    )

    # problem solve
    problem.solve_problem()

    # --- results check
    tol = 1.0e-6

    check_node = []
    check_node.append(np.array([0.5, 0.5, 0.0]))

    v_corr, p_corr = np.zeros(2 * len(check_node)), np.zeros(len(check_node))

    # correct results
    v_corr[0] = 2.1786040945059719E+00  # x
    v_corr[1] = -7.2292113543838623E-06  # y

    p_corr[0] = -1.9440344693873112E-07

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
    check2 = ambit_fe.resultcheck.results_check_node(
        problem.mp.pbf.p,
        check_node,
        p_corr,
        problem.mp.pbf.V_p,
        problem.mp.comm,
        tol=tol,
        nm="p",
        readtol=1e-4,
    )

    success = ambit_fe.resultcheck.success_check([check1, check2], problem.mp.comm)

    if not success:
        raise RuntimeError("Test failed!")



if __name__ == "__main__":
    test_main()
