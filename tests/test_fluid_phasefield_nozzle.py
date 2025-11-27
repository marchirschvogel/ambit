#!/usr/bin/env python3

"""
Two-phase flow in nozzle with pressure boundary condition
"""

import ambit_fe

import numpy as np
from pathlib import Path
import pytest


@pytest.mark.fluid_phasefield
@pytest.mark.skip(reason="Not yet ready for testing.")
def test_main():
    basepath = str(Path(__file__).parent.absolute())

    IO_PARAMS = {
        "problem_type": "fluid_phasefield",
        "write_results_every": 1,
        "indicate_results_by": "step",
        "output_path": basepath + "/tmp/",
        "mesh_domain": basepath + "/input/nozzle_domain.xdmf",
        "mesh_boundary": basepath + "/input/nozzle_boundary.xdmf",
        "mesh_subboundary": basepath + "/input/nozzle_point.xdmf",
        "mesh_encoding": "ASCII",  # HDF5, ASCII
        "results_to_write": [["velocity", "pressure", "cauchystress"],["phase", "potential"]],
        "simname": "fluid_ale_nozzle_rot",
    }

    class expr1:
        def __init__(self):
            self.t = 0

        def evaluate(self, x):
            val = 0.0
            return (
                np.full(x.shape[1], val),
            )

    CONTROL_PARAMS = {"maxtime": 100.0,
                      "numstep": 100,
                      # "numstep_stop": 10,
                      "initial_fields": [expr1, None],
                      }

    SOLVER_PARAMS = {
        "solve_type": "direct",
        "direct_solver": "mumps",
        "maxiter":10,
        "tol_res": [1e-6, 1e-6, 1e-6, 1e-6],
        "tol_inc": [1e-6, 1e-6, 1e-6, 1e-6],
    }

    TIME_PARAMS_FLUID = {"timint": "ost", "theta_ost": 0.5,
                         "fluid_governing_type": "navierstokes_transient"}

    TIME_PARAMS_PF = {"timint": "ost", "theta_ost": 0.5}


    FEM_PARAMS_FLUID = {"order_vel": 1,
                        "order_pres": 1,
                        "quad_degree": 5,
                        'stabilization'  : {'scheme'         : 'supg_pspg',
                                            'vscale'         : 1.0e1,
                                            'dscales'        : [1.,1.,1.],
                                            'symmetric'      : True,
                                            'reduced_scheme' : True,
                                            'vscale_vel_dep' : False}}

    FEM_PARAMS_PF = {"order_phi": 1, "order_mu": 1, "quad_degree": 5}

    COUPLING_PARAMS = {}


    MATERIALS_FLUID = {"MAT1": {"newtonian": {"mu": 1.0e-6},
                                "inertia": {"rho1": 1.0e-6, "rho2": 1.0e-3}}}

    MATERIALS_PF = {"MAT1": {"mat_cahnhilliard": {"D": 100.},
                          "params_cahnhilliard": {"M": 1.0, "lambda": 0.01}}}

    # define your load curves here (syntax: tcX refers to curve X, to be used in BC_DICT key 'curve' : [X,0,0], or 'curve' : X)
    class time_curves:
        def tc1(self, t):
            pmax = 0.001
            t_ramp = CONTROL_PARAMS["maxtime"]
            return 0.5 * (-(pmax)) * (1.0 - np.cos(np.pi * t / t_ramp))

    BC_DICT_FLUID = {
        "neumann" : [{"id" : [1], "dir":"normal_cur", "curve":1}], # inlet
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
        time_curves=time_curves(),
        coupling_params=COUPLING_PARAMS
    )

    # problem solve
    problem.solve_problem()

    # --- results check
    tol = 1.0e-6

    check_node = []
    check_node.append(np.array([250.0, -1.31579, 0.0]))

    v_corr, p_corr = np.zeros(2 * len(check_node)), np.zeros(len(check_node))

    # correct results
    v_corr[0] = 8.5378522799288383E+00  # x
    v_corr[1] = -4.4531337037709484E+00  # y

    p_corr[0] = -1.4697266781934673E-07

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
        problem.mp.pbf.p,
        check_node,
        p_corr,
        problem.mp.pbf.V_p,
        problem.mp.comm,
        tol=tol,
        nm="p",
        readtol=1e-4,
    )

    success = ambit_fe.resultcheck.success_check([check1, check2], problem.mp.comm)

    if not success:
        raise RuntimeError("Test failed!")



if __name__ == "__main__":
    test_main()
