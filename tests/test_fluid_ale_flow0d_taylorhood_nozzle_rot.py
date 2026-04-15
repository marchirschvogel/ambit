#!/usr/bin/env python3

"""
Rotating nozzle with pressure boundary condition at inlet and a 0D model at outlet
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
        "problem_type": "fluid_ale_flow0d",
        "write_results_every": 1,
        "indicate_results_by": "step",
        "output_path": basepath + "/tmp/",
        "mesh_domain": basepath + "/input/nozzle_domain.xdmf",
        "mesh_boundary": basepath + "/input/nozzle_boundary.xdmf",
        "mesh_subboundary": basepath + "/input/nozzle_point.xdmf",
        "mesh_encoding": "ASCII",  # HDF5, ASCII
        "results_to_write": [["velocity", "pressure"],["aledisplacement"]],
        "simname": "fluid_ale_flow0d_taylorhood_nozzle_rot",
    }

    CONTROL_PARAMS = {"maxtime": 100.0,
                      "numstep": 100,
                      "numstep_stop": 3,
                      }

    SOLVER_PARAMS = {
        "solve_type": "direct",
        "direct_solver": "mumps",
        "maxiter": 10,
        "tol_res": 1e-8,
        "tol_inc": 1e-8,
        "subsolver_params": {"tol_res": 1.0e-8, "tol_inc": 1.0e-8},
    }

    TIME_PARAMS_FLUID = {"timint": "ost", "theta_ost": 0.67, "eval_nonlin_terms": "trapezoidal",
                         "fluid_governing_type": "stokes_transient"}

    FEM_PARAMS_FLUID = {"order_vel": 2,
                        "order_pres": 1,
                        "quad_degree": 5,
                        "fluid_formulation": "conservative"}

    FEM_PARAMS_ALE = {"order_disp": 2, "quad_degree": 5}

    TIME_PARAMS_FLOW0D = {
        "timint": "ost",
        "theta_ost": 1.0,
        "initial_conditions": {"Q_0": 0.0, "p_0": 0.0},
    }

    MODEL_PARAMS_FLOW0D = {
        "modeltype": "2elwindkessel",
        "parameters": {"C": 1.0e3, "R": 1.0e-2, "p_ref": 0.1},
    }

    COUPLING_PARAMS = {
        "coupling_ale_fluid": {"interface": [2,3,5,6]}, # no-slip at moving ALE boundary
    }

    COUPLING_PARAMS_FLUID_FLOW0D = {
        "interfaces": [[4]],
        "coupling_quantity": ["pressure"],
        "variable_quantity": ["flux"],
        "cq_factor": [1.0],
        "coupling_type": "monolithic_lagrange",
        "print_subiter": False,
    }

    MATERIALS_FLUID = {"MAT1": {"newtonian": {"eta": 1.0e-6},
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
        [TIME_PARAMS_FLUID, TIME_PARAMS_FLOW0D],
        SOLVER_PARAMS,
        [FEM_PARAMS_FLUID, FEM_PARAMS_ALE],
        [MATERIALS_FLUID, MATERIALS_ALE, MODEL_PARAMS_FLOW0D],
        [BC_DICT_FLUID, BC_DICT_ALE],
        time_curves=time_curves(),
        coupling_params=[COUPLING_PARAMS,COUPLING_PARAMS_FLUID_FLOW0D],
    )

    # problem solve
    problem.solve_problem()

    # --- results check
    tol = 1.0e-6

    check_node = []
    check_node.append(np.array([250.0, -1.31579, 0.0]))

    v_corr, p_corr = np.zeros(2 * len(check_node)), np.zeros(len(check_node))

    # correct results
    v_corr[0] = 3.1684127045703714E-01  # x
    v_corr[1] = -5.5190484393401118E-01  # y

    p_corr[0] = -2.6588073697576630E-04

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
