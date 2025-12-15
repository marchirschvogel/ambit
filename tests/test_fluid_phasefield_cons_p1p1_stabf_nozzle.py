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
        "mesh_encoding": "ASCII",  # HDF5, ASCII
        "results_to_write": [["velocity", "pressure", "cauchystress"],["phase", "potential"]],
        "simname": "fluid_phasefield_cons_p1p1_stabf_nozzle",
        "write_initial_fields": True,
    }

    class expr1:
        def __init__(self):
            self.t = 0
            self.eps = 1e0
            self.R_0 = 7.5

            self.x_c = np.asarray([83.3937, -0.084307, 0.0])

        def evaluate(self, x):

            d = np.sqrt( (x[0]-self.x_c[0])**2.0 + (x[1]-self.x_c[1])**2.0 + (x[2]-self.x_c[2])**2.0 )

            val = 0.5*(1.0 + np.tanh((self.R_0 - d)/np.sqrt(2.0)*self.eps))
            return (
                np.full(x.shape[1], val),
            )

    CONTROL_PARAMS = {"maxtime": 1.0,
                      "numstep": 100,
                      "numstep_stop": 5,
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
                                            'symmetric'      : False,
                                            'reduced_scheme' : False,
                                            'vscale_vel_dep' : False},
                        "fluid_formulation": "conservative"}

    FEM_PARAMS_PF = {"order_phi": 2, "order_mu": 2, "quad_degree": 5}

    COUPLING_PARAMS = {}


    MATERIALS_FLUID = {"MAT1": {"newtonian": {"mu1": 1.0e-7, "mu2": 4.0e-6},
                                "inertia": {"rho1": 1.0e-7, "rho2": 1.0e-6}}}

    MATERIALS_PF = {"MAT1": {"mat_cahnhilliard": {"D": 0.1},
                          "params_cahnhilliard": {"M": 1.0, "lambda": 0.01}}}

    # define your load curves here (syntax: tcX refers to curve X, to be used in BC_DICT key 'curve' : [X,0,0], or 'curve' : X)
    class time_curves:
        def tc1(self, t):
            pmax = 5.0
            t_ramp = CONTROL_PARAMS["maxtime"]
            return 0.5 * (-(pmax)) * (1.0 - np.cos(np.pi * t / t_ramp))

    BC_DICT_FLUID = {
        "dirichlet" : [{"id": [2,3,5,6], "dir":"all", "val":0.0}], # no-slip
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
    v_corr[0] = 2.1786040945059719E+00  # x
    v_corr[1] = -7.2292113543838623E-06  # y

    p_corr[0] = -1.9440344693873112E-07

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
