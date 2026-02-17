#!/usr/bin/env python3

"""
Two-phase flow rising bubble in gravitational flield
BDF2 time-integration scheme for both fluid and phasefield
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
        "problem_type": "fluid_phasefield",
        "write_results_every": 5,
        "write_restart_every": 4,
        "indicate_results_by": "time",
        "restart_step": restart_step,
        "output_path": basepath + "/tmp/",
        "mesh_domain": {"type":"rectangle", "celltype":"quadrilateral", "coords_a":[0.0, 0.0], "coords_b":[1.0, 2.0], "meshsize":[32,64]},
        "results_to_write": [["velocity", "pressure", "acceleration", "cauchystress", "density"],["phase", "potential"]],
        "simname": "fluid_phasefield_rising_bubble",
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
                      "dt": 0.004,
                      "numstep_stop": 5,
                      "initial_fields": [expr1, None],
                      }

    SOLVER_PARAMS = {
        "solve_type": "direct",
        "direct_solver": "mumps",
        "maxiter":25,
        "tol_res": [1e-6, 1e-6, 1e-6, 1e-6],
        "tol_inc": [1e-3, 1e-3, 1e-3, 1e-3],
    }

    TIME_PARAMS_FLUID = {"timint": "bdf2",
                         "theta_ost": 1.0, # Not used: only for OST scheme
                         "fluid_governing_type": "navierstokes_transient",
                         "continuity_at_midpoint": False} # Not relevant when using BDF2 scheme

    TIME_PARAMS_PF = {"timint": "bdf2",
                      "theta_ost": 1.0, # Not used: only for OST scheme
                      "potential_at_midpoint": False} # Not relevant when using BDF2 scheme

    FEM_PARAMS_FLUID = {"order_vel": 2,
                        "order_pres": 1,
                        "quad_degree": 5,
                        "fluid_formulation": "conservative"}

    FEM_PARAMS_PF = {"order_phi": 1, "order_mu": 1, "quad_degree": 5, "phi_range": [0.0, 1.0]}

    # fluid1 is surrounding, fluid2 is bubble
    rho1 = 1000.0
    rho2 = 100.0
    eta1 = 10.0
    eta2 = 1.0
    sig = 24.5
    M0 = 1e-3

    MATERIALS_FLUID = {"MAT1": {"newtonian": {"eta1": eta1, "eta2": eta2},
                                "inertia": {"rho1": rho1, "rho2": rho2}}}


    MATERIALS_PF = {"MAT1": {"mat_cahnhilliard": {"M0": M0, "D": sig/eps,
                                                  "kappa": sig*eps,
                                                  "mobility": "degenerate"}}}

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

    class locate_center:
        def evaluate(self, x):
            ctr_x = np.isclose(x[0], 0.5)
            ctr_y = np.isclose(x[1], 1.0)
            return np.logical_and(ctr_x, ctr_y)

    BC_DICT_FLUID = {
        "dirichlet" : [{"locator": locate_top_bottom(), "dir": "all", "val": 0.0},
                       {"locator": locate_left_right(), "dir": "x", "val": 0.0}],
        "dirichlet_pres" : [{"locator": locate_center(), "dir": "all", "val": 0.0}], # fix pressure in middle of domain to have a well-defined pressure level
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

    v_corr = np.zeros(2 * len(check_node))
    phi_corr, mu_corr = np.zeros(len(check_node)), np.zeros(len(check_node))

    # correct results
    v_corr[0] = 0.0  # x
    v_corr[1] = 1.5724610618361199E-02 # y

    phi_corr[0] = 9.9986152237601944E-01
    mu_corr[0] = 1.3305871374527716E+00

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
        problem.mp.pbp.phi,
        check_node,
        phi_corr,
        problem.mp.pbp.V_phi,
        problem.mp.comm,
        tol=tol,
        nm="phi",
        readtol=1e-4,
    )
    check3 = ambit_fe.resultcheck.results_check_node(
        problem.mp.pbp.mu,
        check_node,
        mu_corr,
        problem.mp.pbp.V_mu,
        problem.mp.comm,
        tol=tol,
        nm="mu",
        readtol=1e-4,
    )

    success = ambit_fe.resultcheck.success_check([check1, check2, check3], problem.mp.comm)

    if not success:
        raise RuntimeError("Test failed!")



if __name__ == "__main__":
    test_main()
