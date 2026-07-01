#!/usr/bin/env python3

"""
FSI simulation in a pipe with axial Neumann loads and a pressure-dependent valve model
"""

import ambit_fe

import sys
import numpy as np
from pathlib import Path
import pytest


@pytest.mark.fsi
@pytest.mark.fluid_solid
def test_main():
    basepath = str(Path(__file__).parent.absolute())

    # reads in restart step from the command line
    try:
        restart_step = int(sys.argv[1])
    except:
        restart_step = 0

    IO_PARAMS = {
        "problem_type": "fsi",
        "duplicate_mesh_domains": [[2], [3]],
        "mesh_domain": basepath + "/input/pipe_fsi_domain.xdmf",
        "mesh_boundary": basepath + "/input/pipe_fsi_boundary.xdmf",
        "mesh_encoding": "HDF5",
        "write_results_every": 1,
        "write_restart_every": -1,
        "restart_step": restart_step,
        "output_path": basepath + "/tmp/",
        "results_to_write": [
            ["displacement"],
            ["velocity", "pressure"],
            ["aledisplacement"],
        ],
        "simname": "fsi_p1p1_pipe_valve_fluidgov_neumann_dirichlet",
    }

    CONTROL_PARAMS = {"maxtime": 1.0, "numstep": 10}

    SOLVER_PARAMS = {
        "solve_type": "direct",
        "direct_solver": "mumps",
        "tol_res": [1.0e-4, 1.0e-8, 1.0e-8, 1.0e-8, 1.0e-8],
        "tol_inc": [1.0e-3, 1.0e-3, 1.0e-3, 1.0e-3, 1.0e-3],
    }

    TIME_PARAMS_SOLID = {"timint": "genalpha", "rho_inf_genalpha": 0.8, "eval_nonlin_terms":"midpoint"}

    TIME_PARAMS_FLUID = {
        "timint": "ost",
        "theta_ost": 1.0,
        "fluid_governing_type": "navierstokes_transient",
    }

    FEM_PARAMS_SOLID = {
        "order_disp": 1,
        "order_pres": 1,
        "quad_degree": 5,
        "incompressibility": "no",
    }

    FEM_PARAMS_FLUID = {
        "order_vel": 1,
        "order_pres": 1,
        "quad_degree": 5,
        "fluid_formulation": "nonconservative",  # nonconservative (default), conservative
        "stabilization": {
            "scheme": "supg_pspg",
            "vscale": 1e3,
            "dscales": [1.0, 1.0, 1.0],
        },
    }

    FEM_PARAMS_ALE = {"order_disp": 1, "quad_degree": 5}

    COUPLING_PARAMS = {
        "coupling_fsi": {"interface": [4,5,6]},
        "fsi_system": "neumann_dirichlet",  # neumann_neumann, neumann_dirichlet
    }

    E = 1e5 # kPa
    nu = 0.3
    MATERIALS_SOLID = {"MAT1": {"neohooke_compressible": {"mu": E/(2.*(1.+nu)), "nu": nu},
                                "inertia": {"rho0": 1.070e-6},
                                "id": 1}}

    MATERIALS_FLUID = {"MAT1": {"newtonian": {"eta": 4.0e-6},
                                "inertia": {"rho": 1.025e-6},
                                "id": 2},
                       "MAT2": {"newtonian": {"eta": 4.0e-6},
                                "inertia": {"rho": 1.025e-6},
                                "id": 3}}

    MATERIALS_ALE = {"MAT1": {"diffusion": {"D": 1.0}, "id": 2},
                     "MAT2": {"diffusion": {"D": 1.0}, "id": 3}}

    # define your load curves here (syntax: tcX refers to curve X, to be used in BC_DICT key 'curve' : [X,0,0], or 'curve' : X)
    class time_curves:
        def tc1(self, t):
            t_ramp = 1.0
            pmax = 1.0
            return 0.5 * (-(pmax)) * (1.0 - np.cos(np.pi * t / t_ramp))

        def tc2(self, t):
            pmax = 0.5
            return -pmax

    BC_DICT_SOLID = {"dirichlet": [{"id": [2,3], "dir": "all", "val": 0.}]}

    class locate_fluid_ring_left_right:
        def evaluate(self, x):
            left_b = np.isclose(x[2], 0.0)
            right_b = np.isclose(x[2], 100.0)

            R0 = 15.0
            rad = np.isclose(np.sqrt(x[0]**2.0 + x[1]**2.0), R0)

            rad_l = np.logical_and(rad, left_b)
            rad_r = np.logical_and(rad, right_b)

            return np.logical_or(rad_l, rad_r)

    BC_DICT_FLUID = {
        "dirichlet": [{"id": [locate_fluid_ring_left_right()], "dir": "all", "val": 0.}],
        "neumann": [
            {"id": [8], "dir": "normal_ref", "curve": 1},
            {"id": [9], "dir": "normal_ref", "curve": 2},
        ],
        "robin_valve": [
            {
                "id": [7],
                "type": "dp_smooth",
                "beta_max": 1e3,
                "beta_min": 1e-3,
                "epsilon": 1e-6,
                "dp_monitor_id": 0,
            }
        ],  # 7 is internal surface (valve)
        "dp_monitor": [{"id": [7], "upstream_domain": 1, "downstream_domain": 2}],
        "flux_monitor": [{"id": [7], "on_subdomain": True, "internal": False, "domain": 1}],
    }

    BC_DICT_ALE = {
        "dirichlet": [
            {"id": [8,9], "dir": "all", "val": 0.0}
        ]
    }

    class locate_left_right:
        def evaluate(self, x):
            left_b = np.isclose(x[2], 0.0)
            right_b = np.isclose(x[2], 100.0)
            return np.logical_or(left_b, right_b)

    BC_DICT_LM = {"dirichlet": [{"id": [locate_left_right()], "dir": "all", "val": 0.0}]}  # only needed for neumann_neumann, and if both fluid and solid carry common DBCs!

    # problem setup
    problem = ambit_fe.ambit_main.Ambit(
        IO_PARAMS,
        CONTROL_PARAMS,
        [TIME_PARAMS_SOLID, TIME_PARAMS_FLUID],
        SOLVER_PARAMS,
        [FEM_PARAMS_SOLID, FEM_PARAMS_FLUID, FEM_PARAMS_ALE],
        [MATERIALS_SOLID, MATERIALS_FLUID, MATERIALS_ALE],
        [BC_DICT_SOLID, BC_DICT_FLUID, BC_DICT_ALE, BC_DICT_LM],
        time_curves=time_curves(),
        coupling_params=COUPLING_PARAMS
    )

    # solve time-dependent problem
    problem.solve_problem()

    # --- results check
    tol = 1.0e-6

    check_node = []
    check_node.append(np.array([4.55947, 0.275408, 49.8859]))

    v_corr = np.zeros(3 * len(check_node))

    # correct results
    v_corr[0] = 3.6449313320432437E+00  # x
    v_corr[1] = -8.0114714050309132E+01  # y
    v_corr[2] = -4.5999666099428185E+02  # z

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

    success = ambit_fe.resultcheck.success_check([check1], problem.mp.comm)

    if not success:
        raise RuntimeError("Test failed!")


if __name__ == "__main__":
    test_main()
