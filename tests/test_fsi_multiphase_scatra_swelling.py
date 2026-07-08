#!/usr/bin/env python3

"""
Multiphase FSI elaso-capillary simulation of an incompressible sessile droplet on a soft solid substrate - cf. example in demos folder for more detailed description
"""

import ambit_fe

import sys
import numpy as np
from pathlib import Path
import pytest


@pytest.mark.fsi
@pytest.mark.fluid_solid
@pytest.mark.skip(reason="Not yet ready for testing.")
def test_main():
    basepath = str(Path(__file__).parent.absolute())

    # reads in restart step from the command line
    try:
        restart_step = int(sys.argv[1])
    except:
        restart_step = 0

    IO_PARAMS = {
        "problem_type": "fsi_multiphase",
        "write_results_every": 1,
        "write_restart_every": 4,
        "indicate_results_by": "step",
        "restart_step": restart_step,
        "output_path": basepath + "/tmp/",
        "mesh_domain": {"type": "rectangle", "celltype": "quadrilateral", "coords_a": [-80.0, 0.0], "coords_b": [160.0, 200.0], "meshsize": [48,40]},
        "results_to_write": {"solid": ["displacement"],
                             "fluid": ["velocity", "pressure", "density"],
                             "phasefield": ["phase", "potential"],
                             "scatra": ["concentration"],
                             "ale": ["aledisplacement"]},
        "write_submeshes": True,
        "write_initial_fields": True,
        "report_conservation_properties": True,
        "simname": "fsi_multiphase_scatra_swelling",
    }

    eps = 1.0

    class expr1:
        def __init__(self):
            self.y0 = 100.
        def evaluate(self, x):
            d = self.y0-x[1]
            val = np.tanh(d / (np.sqrt(2.0) * eps))
            return (
                np.full(x.shape[1], val),
            )

    CONTROL_PARAMS = {"maxtime": 10.0,
                      "dt": 0.1,
                      "numstep_stop": 10,
                      "initial_fields": [expr1, None],
                      }

    SOLVER_PARAMS = {
        "solve_type": "direct",
        "direct_solver": "mumps",   # superlu_dist, mumps
        "tol_res": 1e-8,
        "tol_inc": 1e-8,
    }

    TIME_PARAMS_SOLID = {"timint": "genalpha", "rho_inf_genalpha": 0.8, "eval_nonlin_terms": "midpoint"}
    TIME_PARAMS_SC    = {"timint": "ost", "theta_ost": 1.0}
    TIME_PARAMS_FLUID = {"timint": "bdf2"}
    TIME_PARAMS_PF    = {"timint": "bdf2"}

    FEM_PARAMS_SOLID = {
        "order_disp": 2,
        "order_pres": 1,
        "quad_degree": 5,
        "incompressibility": "no",
        "diffusion": True,
    }

    FEM_PARAMS_SC = {
        "order_conc": 1,
        "quad_degree": 5,
    }

    FEM_PARAMS_FLUID = {"order_vel": 2,
                        "order_pres": 1,
                        "quad_degree": 5,
                        "fluid_formulation": "conservative",
                        "mass_formulation": "reduced_mass"}  # conservative_mass, reduced_mass

    FEM_PARAMS_ALE = {"order_disp": 2,
                      "quad_degree": 5}

    FEM_PARAMS_PF = {"order_phi": 1, "order_mu": 1, "quad_degree": 5}

    class locate_interf:
        def evaluate(self, x):
            return np.isclose(x[0], 0.0)

    COUPLING_PARAMS_FSI = {
        "coupling_fsi": {"interface": [locate_interf()]},
        "fsi_system": "neumann_neumann",
        # phase-scatra coupling
        "coupling_phase_solidscatra": True,
    }

    # Use full Korteweg stress in capillary force contribution - needed for correct inclusion of capillary traction forces at FSI interface!
    COUPLING_PARAMS_MULTIPHASE = {"capillary_force_from_korteweg_stress": True}

    dlt=1e-5
    class locate_solid:
        def evaluate(self, x):
            return (x[0] <= 0.0+dlt)

    class locate_fluid:
        def evaluate(self, x):
            return (x[0] >= 0.0-dlt)

    # locators for boundary conditions
    class locate_right:
        def evaluate(self, x):
            return np.isclose(x[0], 160.0)
    class locate_left:
        def evaluate(self, x):
            return np.isclose(x[0], -80.0)

    class locate_top:
        def evaluate(self, x):
            return np.isclose(x[1], 200.0)

    class locate_bottom:
        def evaluate(self, x):
            return np.isclose(x[1], 0.0)


    E = 1.0
    MATERIALS_SOLID = {"MAT1": {"neohooke_dev": {"mu": E/3.},
                                "inertia": {"rho0": 1.0e-3},
                                "growth": {
                                    "growth_dir": "isotropic",
                                    "growth_trig": "concentration",
                                    "c0": 0.0,
                                    "beta": 1e-1,
                                },
                                "id": locate_solid()}}

    MATERIALS_SC = {"MAT1": {"mat_diff": {"D": 1e-2}, "id": locate_solid()}}

    # fluid1 is oxygen, fluid2 is water
    rho1 = 1e0
    rho2 = 1e3
    eta1 = 1.81e-5
    eta2 = 1e-3
    sig = 72.8e-3 # N/m - value for water-air
    M0 = 0.1*eps**2.0  # keep the pre-factor lower than 1 ...

    sigtilde = 3.*sig/(2.*np.sqrt(2.))

    zeta = 0.0

    MATERIALS_FLUID = {"MAT1": {"newtonian": {"eta1": eta1, "eta2": eta2, "zeta1": zeta, "zeta2": zeta},
                                "inertia": {"rho1": rho1, "rho2": rho2},
                                "id": locate_fluid()}}

    MATERIALS_ALE = {"MAT1": {"diffusion": {"D": 1.0}, "id": locate_fluid()}}

    m = 1e-8 # mobility should be rather low if capillary stress is rather high
    MATERIALS_PF = {"MAT1": {"mat_cahnhilliard": {"mobility": "degenerate",
                                                  "epsilon": 0.0,
                                                  "exponent": 1.0,
                                                  "M0": m*eps**2.0,  # Mobility [length^5/(pressure time)]
                                                  "D": sigtilde/(4.*eps),  # Bulk free-energy parameter [pressure/length^3]
                                                  "kappa": sigtilde*eps},  # Gradient energy coefficient [pressure/length]
                                                  "id": locate_fluid()}}

    BC_DICT_SOLID = {
        "dirichlet": [{"id": [locate_left()], "dir": "all", "val": 0.0},
                      {"id": [locate_top(),locate_bottom()], "dir": "y", "val": 0.0}],
        }

    BC_DICT_SC = { }

    BC_DICT_FLUID = {
        "dirichlet": [{"id": [locate_top(),locate_bottom()], "dir": "y", "val": 0.0}],
    }

    BC_DICT_ALE = {
        "dirichlet": [{"id": [locate_right()], "dir": "all", "val": 0.0},
                      {"id": [locate_top(),locate_bottom()], "dir": "y", "val": 0.0}],
    }

    BC_DICT_PF = { }


    # only for neumann_neumann formulation
    BC_DICT_LM = {"dirichlet": [{"id": [locate_top(),locate_bottom()], "dir": "y", "val": 0.0}]}

    problem = ambit_fe.ambit_main.Ambit(
        IO_PARAMS,
        CONTROL_PARAMS,
        [[TIME_PARAMS_SOLID, TIME_PARAMS_SC], [TIME_PARAMS_FLUID], [TIME_PARAMS_PF]],
        SOLVER_PARAMS,
        [[FEM_PARAMS_SOLID, FEM_PARAMS_SC], [FEM_PARAMS_FLUID], [FEM_PARAMS_PF], [FEM_PARAMS_ALE]],
        [[MATERIALS_SOLID, MATERIALS_SC], [MATERIALS_FLUID], [MATERIALS_PF], [MATERIALS_ALE]],
        [[BC_DICT_SOLID, BC_DICT_SC], [BC_DICT_FLUID], [BC_DICT_PF], [BC_DICT_ALE], [BC_DICT_LM]],
        coupling_params=[COUPLING_PARAMS_FSI, COUPLING_PARAMS_MULTIPHASE],
    )

    # problem solve
    problem.solve_problem()

    # # --- results check
    # tol = 1.0e-6
    #
    # check_node = []
    # check_node.append(np.array([175.0, 0., 0.]))
    #
    # u_corr, v_corr = (
    #     np.zeros(2 * len(check_node)),
    #     np.zeros(2 * len(check_node)),
    # )
    #
    # # correct results
    # u_corr[0] = 9.0328129760261139E-02  # x
    # u_corr[1] = 4.4280664205851694E-01  # y
    #
    # v_corr[0] = 2.3074767133748433E-03  # x
    # v_corr[1] = 1.0848961391266596E-02  # y
    #
    # check1 = ambit_fe.resultcheck.results_check_node(
    #     problem.mp.pbs.u,
    #     check_node,
    #     u_corr,
    #     problem.mp.pbs.V_u,
    #     problem.mp.comm,
    #     tol=tol,
    #     nm="u",
    #     readtol=1e-4,
    # )
    # check2 = ambit_fe.resultcheck.results_check_node(
    #     problem.mp.pbf.v,
    #     check_node,
    #     v_corr,
    #     problem.mp.pbf.V_v,
    #     problem.mp.comm,
    #     tol=tol,
    #     nm="v",
    #     readtol=1e-4,
    # )
    #
    # success = ambit_fe.resultcheck.success_check([check1, check2], problem.mp.comm)
    #
    # if not success:
    #     raise RuntimeError("Test failed!")


if __name__ == "__main__":
    test_main()
