#!/usr/bin/env python3

"""
solid 3D-0D coupling: incompressible hollow solid chamber coupled to 2-element windkessel model
monolithic coupling via multiplier constraint (a more general coupling way that subs out the 0D model to an extra solve)
"""

import ambit_fe

import sys
import numpy as np
from pathlib import Path
import pytest


@pytest.mark.solid_flow0d
def test_main():
    basepath = str(Path(__file__).parent.absolute())

    # reads in restart step from the command line
    try:
        restart_step = int(sys.argv[1])
    except:
        restart_step = 0

    IO_PARAMS = {
        "problem_type": "solid_flow0d",
        "mesh_domain": basepath + "/input/chamber_domain.xdmf",
        "mesh_boundary": basepath + "/input/chamber_boundary.xdmf",
        "write_results_every": -999,
        "write_restart_every": 5,
        "restart_step": restart_step,
        "output_path": basepath + "/tmp/",
        "results_to_write": [""],
        "simname": "test",
    }

    CONTROL_PARAMS = {"maxtime": 1.0, "numstep": 100, "numstep_stop": 10}

    SOLVER_PARAMS = {
        "solve_type": "direct",
        "tol_res": 1.0e-8,
        "tol_inc": 1.0e-8,
        "subsolver_params": {"tol_res": 1.0e-8, "tol_inc": 1.0e-8},
    }

    TIME_PARAMS_SOLID = {
        "timint": "ost",
        "theta_ost": 1.0,
        "rho_inf_genalpha": 0.8,
    }

    TIME_PARAMS_FLOW0D = {
        "timint": "ost",
        "theta_ost": 1.0,
        "initial_conditions": {"Q_0": 0.0},
    }

    MODEL_PARAMS_FLOW0D = {
        "modeltype": "2elwindkessel",
        "parameters": {"C": 10.0, "R": 100.0, "p_ref": 0.0},
    }

    FEM_PARAMS = {
        "order_disp": 2,
        "order_pres": 1,
        "quad_degree": 5,
        "incompressibility": "full",
    }

    COUPLING_PARAMS = {
        "surface_ids": [[3]],
        "coupling_quantity": ["pressure"],
        "variable_quantity": ["flux"],
        "coupling_type": "monolithic_lagrange",
    }

    MATERIALS = {"MAT1": {"neohooke_dev": {"mu": 100.0}, "inertia": {"rho0": 1.0e-6}}}

    # define your load curves here (syntax: tcX refers to curve X, to be used in BC_DICT key 'curve' : [X,0,0], or 'curve' : X)
    class time_curves:
        def tc1(self, t):
            tmax = -50.0
            return tmax * np.sin(2.0 * np.pi * t / CONTROL_PARAMS["maxtime"])

    BC_DICT = {
        "dirichlet": [
            {"id": [1], "dir": "x", "val": 0.0},
            {"id": [3], "dir": "y", "val": 0.0},
            {"id": [3], "dir": "z", "val": 0.0},
        ],
        "neumann": [{"id": [2], "dir": "xyz_ref", "curve": [1, 0, 0]}],
    }

    # problem setup
    problem = ambit_fe.ambit_main.Ambit(
        IO_PARAMS,
        CONTROL_PARAMS,
        [TIME_PARAMS_SOLID, TIME_PARAMS_FLOW0D],
        SOLVER_PARAMS,
        FEM_PARAMS,
        [MATERIALS, MODEL_PARAMS_FLOW0D],
        BC_DICT,
        time_curves=time_curves(),
        coupling_params=COUPLING_PARAMS,
    )

    # solve time-dependent problem
    problem.solve_problem()

    # --- results check
    tol = 1.0e-7

    s_corr = np.zeros(problem.mp.pb0.cardvasc0D.numdof)

    # correct 0D results
    s_corr[0] = 4.1613784331931276e00

    check1 = ambit_fe.resultcheck.results_check_vec_sq(problem.mp.pb0.s, s_corr, problem.mp.comm, tol=tol)
    success = ambit_fe.resultcheck.success_check([check1], problem.mp.comm)

    if not success:
        raise RuntimeError("Test failed!")


if __name__ == "__main__":
    test_main()
