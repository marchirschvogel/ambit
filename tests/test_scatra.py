#!/usr/bin/env python3

"""
Scalar transport (diffusion)
"""

import ambit_fe

import sys
import numpy as np
from pathlib import Path
import pytest


@pytest.mark.ale
def test_main():
    basepath = str(Path(__file__).parent.absolute())

    # reads in restart step from the command line
    try:
        restart_step = int(sys.argv[1])
    except:
        restart_step = 0

    IO_PARAMS = {
        "problem_type": "scatra",
        "mesh_domain": {"type":"unit_square", "celltype":"triangle", "meshsize":[10,10]},
        "write_results_every": 1,
        "write_restart_every": 8,
        "restart_step": restart_step,
        "output_path": basepath + "/tmp/",
        "results_to_write": ["concentration"],
        "simname": "diffusion",
    }

    CONTROL_PARAMS = {"maxtime": 1.0, "numstep": 10}

    SOLVER_PARAMS = {
        "solve_type": "direct",
        "tol_res": 1.0e-8,
        "tol_inc": 1.0e-0,
    }  # linear problem, so only one solve needed...

    TIME_PARAMS = {"timint": "ost", "theta_ost": 0.5, "eval_nonlin_terms": "midpoint"}

    FEM_PARAMS = {"order_conc": 1, "quad_degree": 2}

    class locate_all:
        def evaluate(self, x):
            return np.full(x.shape[1], True, dtype=bool)

    MATERIALS = {"MAT1": {"mat_diff": {"D": 1.0}, "id": locate_all()}}

    # define your load curves here (syntax: tcX refers to curve X, to be used in BC_DICT key 'curve' : [X,0,0], or 'curve' : X)
    class time_curves:
        def tc1(self, t):
            return t / CONTROL_PARAMS["maxtime"]

    class locate_left:
        def evaluate(self, x):
            return np.isclose(x[0], 0.0)

    class locate_right:
        def evaluate(self, x):
            return np.isclose(x[0], 1.0)

    BC_DICT = {
        "dirichlet": [{"id": [locate_left()], "val": 0.0}],
        "neumann": [{"id": [locate_right()], "curve": 1}],
    }

    # problem setup
    problem = ambit_fe.ambit_main.Ambit(
        IO_PARAMS,
        CONTROL_PARAMS,
        TIME_PARAMS,
        SOLVER_PARAMS,
        FEM_PARAMS,
        MATERIALS,
        BC_DICT,
        time_curves=time_curves(),
    )

    # solve problem
    problem.solve_problem()

    # --- results check
    tol = 1.0e-6

    check_node = []
    check_node.append(np.array([1.0, 1.0, 0.0]))

    c_corr = np.zeros(len(check_node))

    ## correct results
    c_corr[0] = 6.9274409953522031E-01

    check1 = ambit_fe.resultcheck.results_check_node(
        problem.mp.c[0],
        check_node,
        c_corr,
        problem.mp.V_c,
        problem.mp.comm,
        tol=tol,
        nm="c1",
    )
    success = ambit_fe.resultcheck.success_check([check1], problem.mp.comm)

    if not success:
        raise RuntimeError("Test failed!")


if __name__ == "__main__":
    test_main()
