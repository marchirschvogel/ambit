#!/usr/bin/env python3

"""
solid mechanics, poroelasticity of sphere
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
        "write_restart_every": 4,
        "restart_step": restart_step,
        "output_path": basepath + "/tmp/",
        "results_to_write": ["displacement", "porepressure"],
        "simname": "solid_sphere_poro",
    }

    CONTROL_PARAMS = {"maxtime": 1.0, "numstep": 10, "numstep_stop": 5}

    SOLVER_PARAMS_SOLID = {
        "solve_type": "direct",
        "direct_solver": "mumps",
        "tol_res": 1.0e-8,
        "tol_inc": 1.0e-8,
    }

    TIME_PARAMS_SOLID = {
        "timint": "ost",
        "theta_ost": 0.5,
    }

    FEM_PARAMS = {
        "order_disp": 2,
        "order_pporo": 1,
        "order_pres": 1,
        "quad_degree": 5,
        "incompressibility": "no",
        "poroelasticity": True,
    }

    MATERIALS = {
        "MAT1": {
            "neohooke_compressible": {"mu": 1.0, "nu": 0.3},
            "inertia": {"rho0": 1e-6},
            "mat_poro": {"k": 1.0e-2},
        }
    }

    # define your load curves here (syntax: tcX refers to curve X, to be used in BC_DICT key 'curve' : [X,0,0], or 'curve' : X)
    class time_curves:
        def tc1(self, t):
            pmax = -1.0
            return pmax * t

    class locate_xsym:
        def evaluate(self, x):
            return np.isclose(x[0], 0.0)
    class locate_ysym:
        def evaluate(self, x):
            return np.isclose(x[1], 0.0)
    class locate_zsym:
        def evaluate(self, x):
            return np.isclose(x[2], 0.0)

    BC_DICT = {
        "dirichlet": [{"id": [locate_xsym()], "dir": "x", "val": 0.0},
                      {"id": [locate_ysym()], "dir": "y", "val": 0.0},
                      {"id": [locate_zsym()], "dir": "z", "val": 0.0}],
        "dirichlet_poro": [{"id": [1], "val": 0.0}],  # draining pressure at surface
        "neumann": [
            {"id": [1], "dir": "normal_cur", "curve": 1},
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

    check_node, check_node2 = [], []
    check_node.append(np.array([17.419159327515, 17.419159327515, 17.121500420386]))
    check_node2.append(np.array([0.0, 0.0, 0.0]))

    u_corr, p_corr = np.zeros(3 * len(check_node)), np.zeros(len(check_node))

    ## correct results
    u_corr[0] = -5.7361733579437170E-02  # x
    u_corr[1] = -5.7361732864131196E-02  # y
    u_corr[2] = -5.7924438049653519E-02  # z

    p_corr[0] = 5.1981544004424063E-01

    check1 = ambit_fe.resultcheck.results_check_node(
        problem.mp.u,
        check_node,
        u_corr,
        problem.mp.V_u,
        problem.mp.comm,
        tol=tol,
        nm="u",
    )
    check2 = ambit_fe.resultcheck.results_check_node(
        problem.mp.pporo,
        check_node2,
        p_corr,
        problem.mp.V_pporo,
        problem.mp.comm,
        tol=tol,
        nm="p",
    )
    success = ambit_fe.resultcheck.success_check([check1, check2], problem.mp.comm)

    if not success:
        raise RuntimeError("Test failed!")


if __name__ == "__main__":
    test_main()
