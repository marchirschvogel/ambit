#!/usr/bin/env python3

"""
FSI simulation of a 2D tank with flexible, forced lid
Monolithic Neumann-Neumann formulation (with Lagrange multiplier) - p1p1
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

    IO_PARAMS = {
        "problem_type": "fsi",
        "write_results_every": 1,
        "indicate_results_by": "step",
        "output_path": basepath + "/tmp/",
        "mesh_domain": basepath + "/input/tank2d_domain.xdmf",
        "mesh_boundary": basepath + "/input/tank2d_boundary.xdmf",
        "mesh_encoding": "ASCII",  # HDF5, ASCII
        "results_to_write": [
            ["displacement"],
            [["velocity", "pressure"],
                ["aledisplacement"]],
        ],
        "write_submeshes":True,
        "simname": "tank2d_p1p1_stabr_neumannneumann",
    }

    CONTROL_PARAMS = {"maxtime": 1.0,
                      "dt": 0.0005,
                      "numstep_stop":10,
                      }

    SOLVER_PARAMS = {
        "solve_type": "direct",
        "direct_solver": "mumps",
        "tol_res": [1e-8, 1e-8, 1e-8, 1e-8, 1e-8],
        "tol_inc": [1e-8, 1e-8, 1e-8, 1e5, 1e-8],
    }

    TIME_PARAMS_SOLID = {"timint": "genalpha", "rho_inf_genalpha": 0.8, "eval_nonlin_terms": "midpoint"}
    TIME_PARAMS_FLUID = {"timint": "genalpha", "rho_inf_genalpha": 0.8, "eval_nonlin_terms": "midpoint"}

    FEM_PARAMS_SOLID = {
        "order_disp": 1,
        "order_pres": 1,
        "quad_degree": 5,
        "incompressibility": "no",
    }

    FEM_PARAMS_FLUID = {"order_vel": 1,
                        "order_pres": 1,
                        "quad_degree": 5,
                        "stabilization"  : {"scheme"         : "supg_pspg",
                                            "vscale"         : 1.0e1,
                                            "dscales"        : [1.,1.,1.],
                                            "symmetric"      : True,
                                            "reduced_scheme" : True,
                                            "vscale_vel_dep" : False}}

    FEM_PARAMS_ALE = {"order_disp": 1, "quad_degree": 5}

    COUPLING_PARAMS = {
        "coupling_fluid_ale": {"surface_ids": [3]},
        "fsi_governing_type": "solid_governed", # solid_governed, fluid_governed
        "fsi_system": "neumann_neumann",  # neumann_neumann, neumann_dirichlet
        "remove_mutual_solid_fluid_bcs":False, # TODO: Not working!
    }

    E = 500. # kPa
    nu = 0.3
    MATERIALS_SOLID = {"MAT1": {"neohooke_compressible": {"mu": E/(2.*(1.+nu)), "nu": nu},
                                "inertia": {"rho0": 1.070e-6},
                                "id": 1}}

    MATERIALS_FLUID = {"MAT1": {"newtonian": {"eta": 1.0e-6}, # kPas
                                "inertia": {"rho": 1.0e-6}, # kg/mm^3
                                "id": 2}}

    MATERIALS_ALE = {"MAT1": {"diffusion": {"D": 1.0}, "id": 2}}

    # define your load curves here (syntax: tcX refers to curve X, to be used in BC_DICT key 'curve' : [X,0,0], or 'curve' : X)
    class time_curves:
        def tc1(self, t):
            Tp = 0.5
            pmax = 3.0 # kPa
            return -pmax * np.sin(2.*np.pi*t/Tp)

    BC_DICT_SOLID = {"neumann": [{"id": [1], "dir": "normal_cur", "curve": 1}],
        "dirichlet": [{"id": [2], "dir": "all", "val": 0.}],
        }

    BC_DICT_FLUID = {
        "dirichlet": [{"id":[4], "dir": "all", "val": 0.}],
        'stabilized_neumann' : [{'id' : [5], 'beta' : 0.2e-6, 'gamma' : 1.}]
    }

    BC_DICT_ALE = {
        "dirichlet": [
            {"id": [4,5], "dir": "all", "val": 0.0}
        ]
    }

    problem = ambit_fe.ambit_main.Ambit(
        IO_PARAMS,
        CONTROL_PARAMS,
        [TIME_PARAMS_SOLID, TIME_PARAMS_FLUID],
        SOLVER_PARAMS,
        [FEM_PARAMS_SOLID, FEM_PARAMS_FLUID, FEM_PARAMS_ALE],
        [MATERIALS_SOLID, MATERIALS_FLUID, MATERIALS_ALE],
        [BC_DICT_SOLID, BC_DICT_FLUID, BC_DICT_ALE],
        time_curves=time_curves(),
        coupling_params=COUPLING_PARAMS
    )

    # problem solve
    problem.solve_problem()


    # --- results check
    tol = 1.0e-6

    check_node = []
    check_node.append(np.array([10., 1., 0.]))

    v_corr, p_corr = (
        np.zeros(2 * len(check_node)),
        np.zeros(len(check_node)),
    )

    # correct results
    v_corr[0] = 5.1486298166103154E+01  # x
    v_corr[1] = -2.5581846922962019E+00  # y

    p_corr[0] = -1.4217666576649382E-03

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
