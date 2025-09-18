#!/usr/bin/env python3

"""
solid mechanics, incompressible hollow sphere, testing of inverse mechanics
"""

import ambit_fe

import sys
import numpy as np
from pathlib import Path
import pytest


@pytest.mark.solid
def test_main():
    basepath = str(Path(__file__).parent.absolute())

    # reads in restart step from the command line
    try:
        restart_step = int(sys.argv[1])
    except:
        restart_step = 0

    IO_PARAMS = {
        "problem_type": "solid",
        "mesh_domain": basepath + "/input/sphere-quad_domain.xdmf",
        "mesh_boundary": basepath + "/input/sphere-quad_boundary.xdmf",
        "indicate_results_by": "step0",
        "write_results_every": 1,
        "write_restart_every": -1,
        "restart_step": restart_step,
        "output_path": basepath + "/tmp/",
        "results_to_write": ["displacement", "pressure"],
        "simname": "solid_sphere_inverse",
    }

    CONTROL_PARAMS = {"maxtime": 1.0, "numstep": 10, "numstep_stop": 5}

    SOLVER_PARAMS_SOLID = {
        "solve_type": "direct",
        "tol_res": 1.0e-8,
        "tol_inc": 1.0e-8,
    }

    TIME_PARAMS_SOLID = {
        "timint": "static",
    }

    FEM_PARAMS = {
        "order_disp": 2,
        "order_pres": 1,
        "quad_degree": 5,
        "incompressibility": "full",
        "inverse_mechanics": True,
    }

    MATERIALS = {
        "MAT1": {
            "mooneyrivlin_dev": {"c1": 60.0, "c2": -20.0},
        }
    }

    # define your load curves here (syntax: tcX refers to curve X, to be used in BC_DICT key 'curve' : [X,0,0], or 'curve' : X)
    class time_curves:
        def tc1(self, t):
            pmax = -10.
            return pmax * t


    BC_DICT = {
        "dirichlet": [{"id": [3], "dir": "x", "val": 0.0},
                      {"id": [4], "dir": "y", "val": 0.0},
                      {"id": [5], "dir": "z", "val": 0.0}],
        "neumann": [ # pressure load should be set to ref - this is here the known spatial configuration!
            {"id": [1], "dir": "normal_ref", "curve": 1},
        ],
    }

    # problem setup
    problem = ambit_fe.ambit_main.Ambit(
        IO_PARAMS,
        CONTROL_PARAMS,
        TIME_PARAMS_SOLID,
        SOLVER_PARAMS_SOLID,
        FEM_PARAMS,
        MATERIALS,
        BC_DICT,
        time_curves=time_curves(),
    )

    # solve time-dependent problem
    problem.solve_problem()

    # --- results check
    tol = 1.0e-6

    check_node = []
    check_node.append(np.array([17.419159327515, 17.419159327515, 17.121500420386]))

    u_corr = np.zeros(3 * len(check_node))

    ## correct results
    u_corr[0] = -4.8919037962546347E-01  # x
    u_corr[1] = -4.8919037844099700E-01  # y
    u_corr[2] = -4.8085922434390482E-01  # z

    check1 = ambit_fe.resultcheck.results_check_node(
        problem.mp.u,
        check_node,
        u_corr,
        problem.mp.V_u,
        problem.mp.comm,
        tol=tol,
        nm="u",
    )
    success = ambit_fe.resultcheck.success_check([check1], problem.mp.comm)

    if not success:
        raise RuntimeError("Test failed!")


if __name__ == "__main__":
    test_main()
