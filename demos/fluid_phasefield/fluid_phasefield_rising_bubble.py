#!/usr/bin/env python3

"""
Two-phase flow rising bubble in gravitational flield
"""

import ambit_fe

import numpy as np
from pathlib import Path


def main():
    basepath = str(Path(__file__).parent.absolute())

    # cases (1,2) from Eikelder et al. (2024)
    case = 1

    IO_PARAMS = {
        "problem_type": "fluid_phasefield",
        "write_results_every": 1,
        "indicate_results_by": "time",
        "output_path": basepath + "/tmp/",
        "mesh_domain": {"type":"rectangle", "celltype":"quadrilateral", "coords_a":[0.0, 0.0], "coords_b":[1.0, 2.0], "meshsize":[128,256]},
        "results_to_write": [["velocity", "pressure", "cauchystress"],["phase", "potential"]],
        "simname": "fluid_phasefield_rising_bubble"+str(case),
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
        "maxiter":25,
        "tol_res": [1e-6, 1e-6, 1e-6, 1e-6],
        "tol_inc": [1e-3, 1e16, 1e-3, 1e-3],
        "divergence_continue": "PTC",
        "k_ptc_initial": 10.0,
        "catch_max_inc_value": 1e12,
    }

    TIME_PARAMS_FLUID = {"timint": "ost", "theta_ost": 0.5,
                         "fluid_governing_type": "navierstokes_transient",
                         "eval_nonlin_terms": "midpoint", # midpoint, trapezoidal
                         "continuity_at_midpoint": True} # Should use midpoint if time derivative (drho/dt) is involved...
    TIME_PARAMS_PF = {"timint": "ost",
                      "theta_ost": 0.5,
                      "eval_nonlin_terms": "midpoint", # midpoint, trapezoidal
                      "potential_at_midpoint": False}


    FEM_PARAMS_FLUID = {"order_vel": 2,
                        "order_pres": 1,
                        "quad_degree": 5,
                        "fluid_formulation": "conservative"}

    FEM_PARAMS_PF = {"order_phi": 1, "order_mu": 1, "quad_degree": 5}

    # fluid1 is bubble, fluid2 is surrounding
    # TODO: How is M chosen in paper?
    if case==1:
        rho1 = 100.0
        rho2 = 1000.0
        eta1 = 1.0
        eta2 = 10.0
        sig = 24.5
        M = 0.5e-3
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


if __name__ == "__main__":
    main()
