#!/usr/bin/env python3

"""
FSI of arterial segment: stabilized equal-order formulation, compressible solid
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
        "write_results_every": 1,
        "indicate_results_by": "step",
        "write_restart_every": 1,
        "restart_step": restart_step,
        "restart_io_type": "petscvector",  # petscvector, plaintext
        "output_path": basepath + "/tmp/",
        "mesh_domain": basepath + "/input/artseg-fsi-tet-lin_domain.xdmf",
        "mesh_boundary": basepath + "/input/artseg-fsi-tet-lin_boundary.xdmf",
        "results_to_write": [
            ["displacement", "velocity"],
            [
                ["fluiddisplacement", "velocity", "pressure"],
                ["aledisplacement", "alevelocity"],
            ],
        ],
        "domain_ids_solid": [1],
        "domain_ids_fluid": [2],
        "surface_ids_interface": [1],
        "simname": "fsi_p1p1_stab_artseg",
        "write_submeshes":True,
    }

    CONTROL_PARAMS = {"maxtime": 3.0, "numstep": 150, "numstep_stop": 5}

    SOLVER_PARAMS = {
        "solve_type": "direct",
        "direct_solver": "mumps",
        "tol_res": [1e-8, 1e-8, 1e-8, 1e-8, 1e-6],
        "tol_inc": [1e-0, 1e-0, 1e-0, 1e-0, 1e-0],
    }

    TIME_PARAMS_SOLID = {"timint": "genalpha", "rho_inf_genalpha": 1.0}

    TIME_PARAMS_FLUID = {"timint": "genalpha", "rho_inf_genalpha": 1.0}

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
        "stabilization": {
            "scheme": "supg_pspg",
            "vscale": 1e3,
            "dscales": [1.0, 1.0, 1.0],
            "symmetric": True,
            "reduced_scheme": True,
        },
    }

    FEM_PARAMS_ALE = {"order_disp": 1, "quad_degree": 5}

    COUPLING_PARAMS = {
        "coupling_fluid_ale": [{"surface_ids": [1], "type": "strong_dirichlet"}],
        "fsi_governing_type": "solid_governed",  # solid_governed, fluid_governed
        "fsi_system": "neumann_neumann",  # neumann_neumann, neumann_dirichlet
        "remove_mutual_solid_fluid_bcs": False,  # Not yet implemented!
    }

    MATERIALS_SOLID = {
        "MAT1": {
            "neohooke_dev": {"mu": 100.0},
            "sussmanbathe_vol": {"kappa": 500.0},
            "inertia": {"rho0": 1.0e-6},
        }
    }

    MATERIALS_FLUID = {"MAT1": {"newtonian": {"mu": 4.0e-6}, "inertia": {"rho": 1.025e-6}}}

    MATERIALS_ALE = {"MAT1": {"linelast": {"Emod": 2.0, "nu": 0.1}}}

    # define your load curves here (syntax: tcX refers to curve X, to be used in BC_DICT key 'curve' : [X,0,0], or 'curve' : X)
    class time_curves:
        def tc1(self, t):
            t_ramp = 2.0
            p0 = 0.0
            pinfl = 0.1
            return (0.5 * (-(pinfl - p0)) * (1.0 - np.cos(np.pi * t / t_ramp)) + (-p0)) * (t < t_ramp) + (-pinfl) * (
                t >= t_ramp
            )

    BC_DICT_SOLID = {
        "dirichlet": [
            {"id": [2, 4], "dir": "z", "val": 0.0},
            {"id": [6], "dir": "y", "val": 0.0},
            {"id": [8], "dir": "x", "val": 0.0},
        ]
    }

    BC_DICT_FLUID = {
        "neumann": [{"id": [3, 5], "dir": "normal_cur", "curve": 1}],
        "dirichlet": [
            {"id": [7], "dir": "y", "val": 0.0},
            {"id": [9], "dir": "x", "val": 0.0},
        ],
    }

    BC_DICT_ALE = {
        "dirichlet": [
            {"id": [3, 5], "dir": "z", "val": 0.0},
            {"id": [7], "dir": "y", "val": 0.0},
            {"id": [9], "dir": "x", "val": 0.0},
        ]
    }

    # problem setup
    problem = ambit_fe.ambit_main.Ambit(
        IO_PARAMS,
        CONTROL_PARAMS,
        [TIME_PARAMS_SOLID, TIME_PARAMS_FLUID],
        SOLVER_PARAMS,
        [FEM_PARAMS_SOLID, FEM_PARAMS_FLUID, FEM_PARAMS_ALE],
        [MATERIALS_SOLID, MATERIALS_FLUID, MATERIALS_ALE],
        [BC_DICT_SOLID, BC_DICT_FLUID, BC_DICT_ALE],
        time_curves=time_curves(),
        coupling_params=COUPLING_PARAMS,
    )

    # problem solve
    problem.solve_problem()

    # --- results check
    tol = 1.0e-6

    check_node = []
    check_node.append(np.array([7.07107, 7.07107, 2.5]))

    u_corr, v_corr, p_corr = (
        np.zeros(3 * len(check_node)),
        np.zeros(3 * len(check_node)),
        np.zeros(len(check_node)),
    )

    # correct results
    u_corr[0] = 1.4135667472799776e-04  # x
    u_corr[1] = 1.4218659625759257e-04  # y
    u_corr[2] = -1.7638258092165119e-07  # z

    v_corr[0] = 2.8005565683996473e-03  # x
    v_corr[1] = 2.8176524733115717e-03  # y
    v_corr[2] = -3.4904136083075215e-06  # z

    p_corr[0] = 6.1541709918483292e-04

    check1 = ambit_fe.resultcheck.results_check_node(
        problem.mp.pbs.u,
        check_node,
        u_corr,
        problem.mp.pbs.V_u,
        problem.mp.comm,
        tol=tol,
        nm="u",
        readtol=1e-4,
    )
    check2 = ambit_fe.resultcheck.results_check_node(
        problem.mp.pbf.v,
        check_node,
        v_corr,
        problem.mp.pbf.V_v,
        problem.mp.comm,
        tol=tol,
        nm="v",
        readtol=1e-4,
    )
    check3 = ambit_fe.resultcheck.results_check_node(
        problem.mp.pbf.p,
        check_node,
        p_corr,
        problem.mp.pbf.V_p,
        problem.mp.comm,
        tol=tol,
        nm="p",
        readtol=1e-4,
    )

    success = ambit_fe.resultcheck.success_check([check1, check2, check3], problem.mp.comm)

    if not success:
        raise RuntimeError("Test failed!")


if __name__ == "__main__":
    test_main()
