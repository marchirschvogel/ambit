#!/usr/bin/env python3

"""
solid mechanics coupled to swelling / growth-inducing scalar transport
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
        "mesh_domain": basepath + "/input/sphere-unit-quad_domain.xdmf",
        "mesh_boundary": basepath + "/input/sphere-unit-quad_boundary.xdmf",
        "indicate_results_by": "step0",
        "write_results_every": 1,
        "write_restart_every": 4,
        "restart_step": restart_step,
        "output_path": basepath + "/tmp/",
        "results_to_write": {"solid": ["displacement"], "scatra": ["concentration"]},
        "simname": "solid_sphere_growth_scatra",
    }

    CONTROL_PARAMS = {"maxtime": 1.0, "numstep": 10, "numstep_stop": 5}

    SOLVER_PARAMS = {
        "solve_type": "direct",
        "direct_solver": "mumps",
        "tol_res": 1.0e-8,
        "tol_inc": 1.0e-4,
    }

    TIME_PARAMS = {
        "timint": "ost",
        "theta_ost": 0.5,
    }

    TIME_PARAMS_SC = {
        "timint": "ost",
        "theta_ost": 0.5,
    }

    FEM_PARAMS = {
        "order_disp": 2,
        "order_pres": 1,
        "quad_degree": 5,
        "incompressibility": "no",
        "diffusion": True,  # adds scalar transport problem
    }

    FEM_PARAMS_SC = {
        "order_conc": 1,
        "quad_degree": 5,
    }

    MATERIALS = {
        "MAT1": {
            "neohooke_compressible": {"mu": 1.0, "nu": 0.3},
            "inertia": {"rho0": 1e-6},
            "growth": {
                "growth_dir": "isotropic",
                "growth_trig": "concentration",
                "growth_law_type": "inst",  # inst, rate
                "c0": 0.1,
                "beta": 1e-1,
            },
        }
    }

    MATERIALS_SC = {"MAT1": {"mat_diff": {"D": 1e-2}}}

    # define your load curves here (syntax: tcX refers to curve X, to be used in BC_DICT key 'curve' : [X,0,0], or 'curve' : X)
    class time_curves:
        def tc1(self, t):
            return 0

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
    }

    BC_DICT_SC = {"dirichlet": [{"id": [1], "val": 1.0}]}  # prescribed concentration at outer surface

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

    # --- results check
    tol = 1.0e-6

    check_node, check_node2 = [], []
    check_node.append(np.array([0.5806386442505, 0.5806386442505002, 0.5707166806795334]))
    check_node2.append(np.array([0.0, 0.0, 0.0]))

    u_corr, c_corr = np.zeros(3 * len(check_node)), np.zeros(len(check_node))

    # correct results
    u_corr[0] = 3.4681923558718048E-03  # x
    u_corr[1] = 3.4681923463234757E-03  # y
    u_corr[2] = 3.4196362165880055E-03  # z

    c_corr[0] = 1.2355157863733316E-03

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
        problem.mp.pbscat.c[0],
        check_node2,
        c_corr,
        problem.mp.pbscat.V_c,
        problem.mp.comm,
        tol=tol,
        nm="c",
    )
    success = ambit_fe.resultcheck.success_check([check1, check2], problem.mp.comm)

    if not success:
        raise RuntimeError("Test failed!")


if __name__ == "__main__":
    test_main()
