#!/usr/bin/env python3

"""
Prescribed oscillating ALE box
"""

import ambit_fe

import numpy as np
from pathlib import Path
import pytest


@pytest.mark.ale
def test_main():
    basepath = str(Path(__file__).parent.absolute())

    IO_PARAMS = {
        "problem_type": "ale",
        "write_results_every": 1,
        "indicate_results_by": "step",
        "output_path": basepath + "/tmp/",
        "mesh_domain": {"type":"unit_square", "celltype":"triangle", "meshsize":[10,10]},
        "results_to_write": ["aledisplacement", "alevelocity", "alestress"],
        "simname": "ale_oscillating_box_prescribed",
        "write_initial_fields": True,
    }

    CONTROL_PARAMS = {"maxtime": 1.0,
                      "numstep": 100,
                      "numstep_stop": 20,
                      }

    SOLVER_PARAMS = {
        "solve_type": "direct",
        "direct_solver": "mumps",
        "maxiter":10,
        "tol_res": [1e-6],
        "tol_inc": [1e3],
    }

    TIME_PARAMS = {"timint": "ost", "theta_ost": 1.0}

    FEM_PARAMS = {"order_disp": 2, "quad_degree": 5}

    MATERIALS = {"MAT1": {"linelast": {"Emod": 10.0, "nu": 0.3}}}

    class expr1:
        def __init__(self):
            self.t = 0.0
            self.L = 1.0
            self.A = 0.3

            T=0.25
            self.omega=2.*np.pi/T

        def evaluate(self, x):
            val_t = self.A*np.sin(self.omega*self.t)*x[1]/self.L

            return (np.full(x.shape[1], val_t),
                    np.full(x.shape[1], 0.0))

    BC_DICT = {
        "dirichlet": [{"dir": "all", "expression": expr1, "codimension": 2}]
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
    )

    # problem solve
    problem.solve_problem()

    # --- results check
    tol = 1.0e-6

    check_node = []
    check_node.append(np.array([0.5, 0.5, 0.0]))

    w_corr = np.zeros(2 * len(check_node))

    # correct results
    w_corr[0] = 7.0455318199677031E-01  # x
    w_corr[1] = 0.0  # y

    check1 = ambit_fe.resultcheck.results_check_node(
        problem.mp.w,
        check_node,
        w_corr,
        problem.mp.V_d,
        problem.mp.comm,
        tol=tol,
        nm="w",
        readtol=1e-4,
    )

    success = ambit_fe.resultcheck.success_check([check1], problem.mp.comm)

    if not success:
        raise RuntimeError("Test failed!")



if __name__ == "__main__":
    test_main()
