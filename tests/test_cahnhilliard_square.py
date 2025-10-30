#!/usr/bin/env python3

"""
Cahn-Hilliard equation
"""

import ambit_fe

import sys
import numpy as np
from pathlib import Path
import pytest


@pytest.mark.cahnhilliard
def test_main():
    basepath = str(Path(__file__).parent.absolute())

    # reads in restart step from the command line
    try:
        restart_step = int(sys.argv[1])
    except:
        restart_step = 0

    IO_PARAMS = {
        "problem_type": "cahnhilliard",
        "mesh_domain": {"type":"unit_square", "celltype":"triangle", "meshsize":[96,96]},
        "write_results_every": 1,
        "write_restart_every": 3,
        "restart_step": restart_step,
        "restart_io_type": "petscvector",
        "output_path": basepath + "/tmp/",
        "results_to_write": ["phasefield", "potential"],
        "simname": "cahnhilliard_square",
        "initial_fields":[basepath + "/input/phi_init.xdmf",None], # phi is initialized with random field from dolfinx testcase
    }

    CONTROL_PARAMS = {"dt": 5.0e-06, "maxtime": 50*5.0e-06, "numstep_stop":5}

    SOLVER_PARAMS = {
        "solve_type": "direct",
        "tol_res": 1.0e-8,
        "tol_inc": 1.0e-8,
    }

    TIME_PARAMS = {"timint": "ost", "theta_ost": 0.5}

    FEM_PARAMS = {"order_phi": 1, "order_mu": 1, "quad_degree": 5}

    MATERIALS = {"MAT1": {"mat_cahnhilliard": {"D": 100.},
                          "params_cahnhilliard": {"M": 1.0, "lambda": 0.01}}}

    BC_DICT = { }

    # problem setup
    problem = ambit_fe.ambit_main.Ambit(
        IO_PARAMS,
        CONTROL_PARAMS,
        TIME_PARAMS,
        SOLVER_PARAMS,
        FEM_PARAMS,
        MATERIALS,
        BC_DICT,
    )

    # solve time-dependent problem
    problem.solve_problem()

    # --- results check
    tol = 1.0e-6

    check_node = []
    check_node.append(np.array([0.0, 0.0, 0.0]))

    phi_corr = np.zeros(len(check_node))
    mu_corr = np.zeros(len(check_node))

    # correct results
    phi_corr[0] = 6.6490604861067215E-01
    mu_corr[0] = -1.3648309634541389E+01

    check1 = ambit_fe.resultcheck.results_check_node(
        problem.mp.phi,
        check_node,
        phi_corr,
        problem.mp.V_phi,
        problem.mp.comm,
        tol=tol,
        nm="phi",
    )
    check2 = ambit_fe.resultcheck.results_check_node(
        problem.mp.mu,
        check_node,
        mu_corr,
        problem.mp.V_mu,
        problem.mp.comm,
        tol=tol,
        nm="mu",
    )
    success = ambit_fe.resultcheck.success_check([check1,check2], problem.mp.comm)

    if not success:
        raise RuntimeError("Test failed!")


if __name__ == "__main__":
    test_main()
