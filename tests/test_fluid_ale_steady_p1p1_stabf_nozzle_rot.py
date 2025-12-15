#!/usr/bin/env python3

"""
Rotating nozzle with pressure boundary condition
- ALE domain receives fixed and moving point DBCs at corners such that a rigid body rotation is induced
- Fluid receives ALE velocities as DBCs on no-slip boundaries
"""

import ambit_fe

import numpy as np
from pathlib import Path
import pytest


@pytest.mark.fluid_ale
def test_main():
    basepath = str(Path(__file__).parent.absolute())

    IO_PARAMS = {
        "problem_type": "fluid_ale",
        "write_results_every": 1,
        "indicate_results_by": "step",
        "output_path": basepath + "/tmp/",
        "mesh_domain": basepath + "/input/nozzle_domain.xdmf",
        "mesh_boundary": basepath + "/input/nozzle_boundary.xdmf",
        "mesh_subboundary": basepath + "/input/nozzle_point.xdmf",
        "mesh_encoding": "ASCII",  # HDF5, ASCII
        "results_to_write": [["velocity", "pressure", "cauchystress"],["aledisplacement", "alevelocity", "alestress"]],
        "simname": "fluid_ale_steady_p1p1_stabf_nozzle_rot",
    }

    CONTROL_PARAMS = {"maxtime": 100.0,
                      "numstep": 100,
                      "numstep_stop": 10,
                      }

    SOLVER_PARAMS = {
        "solve_type": "direct",
        "direct_solver": "mumps",
        "maxiter":10,
        "tol_res": [1e-10, 1e-10, 1e-10],
        "tol_inc": [1e-10, 1e-10, 1e-10],
    }

    TIME_PARAMS_FLUID = {"timint": "ost", "theta_ost": 1.0,
                         "fluid_governing_type": "stokes_steady"}

    FEM_PARAMS_FLUID = {"order_vel": 1,
                        "order_pres": 1,
                        "quad_degree": 5,
                        'stabilization'  : {'scheme'         : 'supg_pspg',
                                            'vscale'         : 1.0e1,
                                            'dscales'        : [1.,1.,1.]},
                        "fluid_formulation": "conservative"} # irrelevant for steady Stokes

    FEM_PARAMS_ALE = {"order_disp": 1, "quad_degree": 5}

    COUPLING_PARAMS = {
        "coupling_ale_fluid": {"surface_ids": [2,3,5,6]}, # no-slip at moving ALE boundary
    }


    MATERIALS_FLUID = {"MAT1": {"newtonian": {"mu": 1.0e-6},
                                "inertia": {"rho": 1.0e-6}}}

    # We need a finite strain capable nonlinear ALE that can undergo a large rotation without straining/shape changing
    MATERIALS_ALE = {"MAT1": {"neohooke": {"mu": 1.0, "nu": 0.1}}}

    # define your load curves here (syntax: tcX refers to curve X, to be used in BC_DICT key 'curve' : [X,0,0], or 'curve' : X)
    class time_curves:
        def tc1(self, t):
            pmax = 0.001
            t_ramp = CONTROL_PARAMS["maxtime"]
            return 0.5 * (-(pmax)) * (1.0 - np.cos(np.pi * t / t_ramp))
        def tc2(self, t):
            l_i = 100.
            l_o = 150.
            h_i = 100.
            h_o = 50.
            val = -(l_i+l_o + h_i/2. + h_o/2.)
            return val*t/CONTROL_PARAMS["maxtime"]


    BC_DICT_FLUID = {
        "neumann" : [{"id" : [1], "dir":"normal_cur", "curve":1}], # inlet
    }


    BC_DICT_ALE = { # point DBCs
        "dirichlet": [
            {"id": [1], "dir": "all", "val": 0., "codimension":0},
            {"id": [2], "dir": "y", "curve": 2, "codimension":0},
        ]
    }

    # problem setup
    problem = ambit_fe.ambit_main.Ambit(
        IO_PARAMS,
        CONTROL_PARAMS,
        TIME_PARAMS_FLUID,
        SOLVER_PARAMS,
        [FEM_PARAMS_FLUID, FEM_PARAMS_ALE],
        [MATERIALS_FLUID, MATERIALS_ALE],
        [BC_DICT_FLUID, BC_DICT_ALE],
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
