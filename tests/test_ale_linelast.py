#!/usr/bin/env python3

"""
dummy linear-elastic ALE solid
physically irrelevant deformation, just testing the correct functionality of the standalone ALE class, incl. output writing
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
        "problem_type": "ale",
        "mesh_domain": basepath + "/input/block_domain.xdmf",
        "mesh_boundary": basepath + "/input/block_boundary.xdmf",
        "write_results_every": 1,
        "write_restart_every": 1,
        "restart_step": restart_step,
        "restart_io_type": "petscvector",
        "output_path": basepath + "/tmp/",
        "results_to_write": ["aledisplacement", "alevelocity"],
        "simname": "ale_linelast",
    }

    CONTROL_PARAMS = {"maxtime": 1.0, "numstep": 10, "numstep_stop": 5}

    SOLVER_PARAMS = {
        "solve_type": "direct",
        "tol_res": 1.0e-8,
        "tol_inc": 1.0e-0,
    }  # linear problem, so only one solve needed...

    TIME_PARAMS = {"timint": "ost", "theta_ost": 1.0}

    FEM_PARAMS = {"order_disp": 1, "quad_degree": 2}

    MATERIALS = {"MAT1": {"linelast": {"Emod": 10.0, "nu": 0.3}}}

    # define your load curves here (syntax: tcX refers to curve X, to be used in BC_DICT key 'curve' : [X,0,0], or 'curve' : X)
    class time_curves:
        def tc1(self, t):
            return 3.0 * t

    BC_DICT = {
        "neumann": [{"id": [4], "dir": "xyz_ref", "curve": [1, 0, 0]}],
        "dirichlet": [{"id": [1], "dir": "all", "val": 0.0}],
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
    check_node.append(np.array([1.0, 1.0, 1.0]))

    d_corr = np.zeros(3 * len(check_node))

    ## correct results
    d_corr[0] = 1.3995086093073472e-01  # x
    d_corr[1] = -2.7223875668985008e-02  # y
    d_corr[2] = -2.6750427252171242e-02  # z

    check1 = ambit_fe.resultcheck.results_check_node(
        problem.mp.d,
        check_node,
        d_corr,
        problem.mp.V_d,
        problem.mp.comm,
        tol=tol,
        nm="d",
    )
    success = ambit_fe.resultcheck.success_check([check1], problem.mp.comm)

    if not success:
        raise RuntimeError("Test failed!")


if __name__ == "__main__":
    test_main()
