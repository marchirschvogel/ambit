#!/usr/bin/env python3

"""
Pressurized incompressible fluid in ALE formulation: consistency check - no deformation, uniform pressure in domain (value of Neumann BC)
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
        "mesh_domain": {"type":"unit_square", "celltype":"triangle", "meshsize":[10,10]},
        "results_to_write": [["velocity", "pressure"],["aledisplacement"]],
        "simname": "fluid_ale_uniform_pressurized",
    }

    CONTROL_PARAMS = {"maxtime": 1.0,
                      "numstep": 10,
                      }

    SOLVER_PARAMS = {
        "solve_type": "direct",
        "direct_solver": "mumps",
        "maxiter":10,
        "tol_res": [1e-8, 1e-8, 1e-8],
        "tol_inc": [1e-8, 1e-8, 1e-8],
    }

    TIME_PARAMS_FLUID = {"timint": "ost",
                         "theta_ost": 0.5,
                         "eval_nonlin_terms": "trapezoidal",
                         "continuity_at_midpoint": True,
                         "fluid_governing_type": "navierstokes_transient"}

    FEM_PARAMS_FLUID = {"order_vel": 2,
                        "order_pres": 1,
                        "quad_degree": 5,
                        "fluid_formulation": "conservative"}

    FEM_PARAMS_ALE = {"order_disp": 2, "quad_degree": 5}

    # all boundary walls
    class locate:
        def evaluate(self, x):
            left_b = np.isclose(x[0], 0.0)
            right_b = np.isclose(x[0], 1.0)
            top_b = np.isclose(x[1], 1.0)
            bottom_b = np.isclose(x[1], 0.0)
            return np.logical_or(np.logical_or(left_b, right_b), np.logical_or(top_b, bottom_b))

    COUPLING_PARAMS = {
        "coupling_fluid_ale": {"locator": locate()},
    }

    MATERIALS_FLUID = {"MAT1": {"newtonian": {"mu": 0.001}, # Pa s
                                "inertia": {"rho": 1000.0}}} # kg/m^3

    MATERIALS_ALE = {"MAT1": {"neohooke": {"mu": 1.0, "nu": 0.1}}}

    # define your load curves here (syntax: tcX refers to curve X, to be used in BC_DICT key 'curve' : [X,0,0], or 'curve' : X)
    class time_curves:
        def tc1(self, t):
            pmax = 100.0
            t_ramp = CONTROL_PARAMS["maxtime"]
            # return 0.5 * (-(pmax)) * (1.0 - np.cos(np.pi * t / t_ramp))
            return -pmax*t/t_ramp

    BC_DICT_FLUID = {
        "neumann" : [{"locator" : locate(), "dir": "normal_cur", "curve": 1}],
    }


    BC_DICT_ALE = { }

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
    check_node.append(np.array([0.5, 0.5, 0.0]))

    v_corr, p_corr = np.zeros(2 * len(check_node)), np.zeros(len(check_node))

    # correct results
    v_corr[0] = 0.0  # x
    v_corr[1] = 0.0  # y

    p_corr[0] = 100.0

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
