#!/usr/bin/env python3

"""
solid mechanics, poroelasticity on unit sphere
"""

import ambit_fe

import sys
import numpy as np
from pathlib import Path
import pytest


@pytest.mark.solid
@pytest.mark.skip(reason="Not yet ready for testing.")
def test_main():
    basepath = str(Path(__file__).parent.absolute())

    # reads in restart step from the command line
    try:
        restart_step = int(sys.argv[1])
    except:
        restart_step = 0

    IO_PARAMS = {
        "problem_type": "solid",
        "mesh_domain": {"type": "rectangle", "celltype": "quadrilateral", "coords_a": [0.0, 0.0], "coords_b": [0.127, 0.127], "meshsize": [20,20]},
        "indicate_results_by": "step0",
        "write_results_every": 1,
        "write_restart_every": -1,
        "restart_step": restart_step,
        "output_path": basepath + "/tmp/",
        "results_to_write": {"solid": ["displacement", "porehydpressure"], "scatra": {"concentration"}},
        "simname": "solid_poro_dary_schloegl_nernst_planck_electrostatics",
    }

    CONTROL_PARAMS = {"maxtime": 1.0,
                      "numstep": 100,
                      # "numstep_stop": 5,
                      }

    SOLVER_PARAMS = {
        "solve_type": "direct",
        "direct_solver": "mumps",
        "tol_res": 1.0e-8,
        "tol_inc": 1.0e-8,
    }

    TIME_PARAMS = {
        "timint": "ost",
        "theta_ost": 1.0,
    }

    TIME_PARAMS_SC = [{"timint": "ost", "theta_ost": 1.0},  # Nernst-Planck
                      {"timint": "static"}]  # electrostatics

    FEM_PARAMS = {
        "order_disp": 2,
        "order_phyd": 1,
        "order_pres": 1,
        "quad_degree": 5,
        "incompressibility": "no",
        "poroelasticity": {"model": "darcy_schloegl",
                           "coupled_c_osmotic": "c1",
                           "coupled_c_electric": "c2"},
    }

    FEM_PARAMS_SC = {
        "order_conc": 1,
        "quad_degree": 5,
    }

    class locate_all:
        def evaluate(self, x):
            return np.full(x.shape[1], True, dtype=bool)

    mu, kappa = 110., 330.  # 110 and 330 MPa (=N/mm^2), cf. Antonini et al. (2025)
    MATERIALS = {
        "MAT1": {
            "neohooke_dev": {"mu": mu},
            "ogden_vol": {"kappa": kappa},
            "inertia": {"rho0": 1e-6},
            "MAT_PORO": {"darcy_schloegl": {"k": 1.0e-2, "k_os": 1e-3, "k_el": 1e-3}},
            "id": locate_all(),
        }
    }

    MATERIALS_SC = [{"MAT1": {"mat_diff_coup": {"D": 1e-2, "Dc": 1.0, "cc": "c2"}, "id": locate_all()}},  # Nernst-Planck equation for flux of protons
                    {"MAT1": {"mat_diff": {"D": 1e-2}, "id": locate_all()}}]  # Poisson equation of electrostatics

    # define your load curves here (syntax: tcX refers to curve X, to be used in BC_DICT key 'curve' : [X,0,0], or 'curve' : X)
    class time_curves:
        def tc1(self, t):
            pmax = -1.0
            return pmax * t

    # locators for boundary conditions
    class locate_left:
        def evaluate(self, x):
            return np.isclose(x[0], 0.0)
    class locate_right:
        def evaluate(self, x):
            return np.isclose(x[0], 0.127)

    class locate_bottom:
        def evaluate(self, x):
            return np.isclose(x[1], 0.0)
    class locate_top:
        def evaluate(self, x):
            return np.isclose(x[1], 0.127)

    class expr_dbc_poro:
        def __init__(self):
            self.t = 0.0

        def evaluate(self, x):
            val = 1.0
            return (np.full(x.shape[1], val))


    class expression_neu:
        def __init__(self):
            self.t = 0.0
            self.t_ramp = 0.5
            self.delta = 1.0 # in [0, 1]
            self.lmbda = 0.02 # mm
        def evaluate(self, x):
            g0 = -10.0
            val_t = g0 * 0.5 * (1.0 - np.cos(np.pi * self.t / self.t_ramp)) * (
                self.t < self.t_ramp
            ) + g0 * (self.t >= self.t_ramp)
            val = val_t * (1.+self.delta*np.sin(2.*np.pi*x[1]/self.lmbda))
            return (np.full(x.shape[1], val))

    BC_DICT = {
        "dirichlet": [{"id": [locate_right()], "dir": "x", "val": 0.0},
                      {"id": [locate_bottom(),locate_top()], "dir": "y", "val": 0.0},
                      ],
        "dirichlet_poro": [{"id": [locate_left(),locate_right()], "expression": expr_dbc_poro}],
        "neumann": [{"id": [locate_left()], "dir": "normal_cur", "expression": expression_neu}],
    }

    BC_DICT_SC = {"dirichlet_c1": [{"id": [locate_left()], "val": 2.8},  # mol/l
                                   {"id": [locate_right()], "val": 2.8}],  # mol/l
                  "dirichlet_c2": [{"id": [locate_left()], "val": 125.0},  # mV
                                   {"id": [locate_right()], "val": 0.0}]}  # mV

    # problem setup
    problem = ambit_fe.ambit_main.Ambit(
        IO_PARAMS,
        CONTROL_PARAMS,
        [TIME_PARAMS, TIME_PARAMS_SC],
        SOLVER_PARAMS,
        [FEM_PARAMS, FEM_PARAMS_SC],
        [MATERIALS, MATERIALS_SC],
        [BC_DICT, BC_DICT_SC],
        time_curves=time_curves(),
    )

    # solve time-dependent problem
    problem.solve_problem()

    # # --- results check
    # tol = 1.0e-6
    #
    # check_node, check_node2 = [], []
    # check_node.append(np.array([0.5806386442505, 0.5806386442505002, 0.5707166806795334]))
    # check_node2.append(np.array([0.0, 0.0, 0.0]))
    #
    # u_corr, p_corr = np.zeros(3 * len(check_node)), np.zeros(len(check_node))
    #
    # ## correct results
    # u_corr[0] = -8.1050317190939653E-03  # x
    # u_corr[1] = -8.1050317001130426E-03  # y
    # u_corr[2] = -7.9805588796660921E-03  # z
    #
    # p_corr[0] = 5.5705312265436846E-01
    #
    # check1 = ambit_fe.resultcheck.results_check_node(
    #     problem.mp.u,
    #     check_node,
    #     u_corr,
    #     problem.mp.V_u,
    #     problem.mp.comm,
    #     tol=tol,
    #     nm="u",
    # )
    # check2 = ambit_fe.resultcheck.results_check_node(
    #     problem.mp.phyd,
    #     check_node2,
    #     p_corr,
    #     problem.mp.V_phyd,
    #     problem.mp.comm,
    #     tol=tol,
    #     nm="p",
    # )
    # success = ambit_fe.resultcheck.success_check([check1, check2], problem.mp.comm)
    #
    # if not success:
    #     raise RuntimeError("Test failed!")


if __name__ == "__main__":
    test_main()
