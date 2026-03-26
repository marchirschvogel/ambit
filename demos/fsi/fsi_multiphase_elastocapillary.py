#!/usr/bin/env python3

"""
Multiphase FSI elaso-capillary simulation of a sessile droplet on a soft solid substrate
Example from M. Shokrpour Roudbari and E. H. van Brummelen, "Binary-Fluid-Solid Interaction Based on the Navier-Stokes-Korteweg Equations", Mathematical Models and Methods in Applied Sciences, 2019
Let's assume a ng-µm-µs unit system, hence viscosities are 1 ng/(µm µs) = 10^{3} mPa s
"""

import ambit_fe
import numpy as np
from pathlib import Path


def main():
    basepath = str(Path(__file__).parent.absolute())

    # reads in restart step from the command line
    try:
        restart_step = int(sys.argv[1])
    except:
        restart_step = 0

    IO_PARAMS = {
        "problem_type": "fsi_multiphase",
        "write_results_every": 1,
        "write_restart_every": -1,
        "indicate_results_by": "step",
        "restart_step": restart_step,
        "output_path": basepath + "/tmp/",
        "mesh_domain": {"type":"rectangle", "celltype":"quadrilateral", "coords_a":[0.0, -50.0], "coords_b":[350.0, 300.0], "meshsize":[70,70]}, # should be divisible by 7
        "results_to_write": [
            ["displacement"],
            ["velocity", "pressure", "density"],
            ["phase", "potential"],
            ["aledisplacement"],
        ],
        "write_initial_fields": True,
        "simname": "fsi_multiphase_elastocapillary_II",
    }

    h = 350.0/IO_PARAMS["mesh_domain"]["meshsize"][0] # element edge length
    eps = 1.28*h

    class expr1:
        def __init__(self):
            self.t = 0
            self.R_0 = 178.4 # µm
            self.x_c = np.asarray([0.0, 19.4, 0.0])

        def evaluate(self, x):
            d = np.sqrt( (x[0]-self.x_c[0])**2.0 + (x[1]-self.x_c[1])**2.0 + (x[2]-self.x_c[2])**2.0 )
            # val = 0.5*(1.0 + np.tanh((self.R_0 - d)/(np.sqrt(2.0)*eps)))  # phi in [0,1]
            val = np.tanh((self.R_0 - d)/(np.sqrt(2.0)*eps))  # phi in [-1,1]
            return (
                np.full(x.shape[1], val),
            )

    CONTROL_PARAMS = {"maxtime": 1000.0, # µs
                      "dt": 1.0, # µs
                      # "numstep_stop": 3,
                      "initial_fields": [expr1, None],
                      }

    SOLVER_PARAMS = {
        "solve_type": "direct",
        "direct_solver": "mumps",   # superlu_dist, mumps
        "tol_res": [1e-8, 1e-8, 1e-8, 1e-8, 1e-8, 1e-8, 1e-8],
        "tol_inc": [1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6],
    }

    TIME_PARAMS_SOLID = {"timint": "genalpha", "rho_inf_genalpha": 0.8, "eval_nonlin_terms": "midpoint"}
    TIME_PARAMS_FLUID = {"timint": "bdf2"}
    TIME_PARAMS_PF    = {"timint": "bdf2"}


    FEM_PARAMS_SOLID = {
        "order_disp": 2,
        "order_pres": 1,
        "quad_degree": 5,
        "incompressibility": "no",
    }

    FEM_PARAMS_FLUID = {"order_vel": 2,
                        "order_pres": 1,
                        "quad_degree": 5,
                        "fluid_formulation": "conservative"}

    FEM_PARAMS_ALE = {"order_disp": 2, "quad_degree": 5}

    FEM_PARAMS_PF = {"order_phi": 2, "order_mu": 2, "quad_degree": 5}

    class locate_interf:
        def evaluate(self, x):
            return np.isclose(x[1], 0.0)

    sig_sl = 36e-3
    sig_sa = 31e-3
    COUPLING_PARAMS = {
        "coupling_fluid_ale": {"interface": [locate_interf()]},
        "fsi_system": "neumann_dirichlet",  # neumann_neumann, neumann_dirichlet
        "wetting_condition_interface": {"coeff": sig_sa-sig_sl}, # wetting Robin condition at interface
    }

    class locate_solid:
        def evaluate(self, x):
            return (x[1] <= 0.0)

    class locate_fluid:
        def evaluate(self, x):
            return (x[1] >= 0.0)

    # locators for boundary conditions
    class locate_right:
        def evaluate(self, x):
            return np.isclose(x[0], 350.0)
    class locate_left:
        def evaluate(self, x):
            return np.isclose(x[0], 0.0)

    class locate_top:
        def evaluate(self, x):
            return np.isclose(x[1], 300.0)

    class locate_bottom:
        def evaluate(self, x):
            return np.isclose(x[1], -50.0)

    E = 3.0e-3 # 1 kPa = 10^{-3} ng/(µm µs^2)
    nu = 0.499
    # - devide the solid into two portions that could have different material properties
    MATERIALS_SOLID = {"MAT1": {"stvenantkirchhoff": {"Emod": E, "nu": nu},
                                "inertia": {"rho0": 12.6e-3}, # 1 pg/(µm^3) = 10^{-3} ng/(µm^3)
                                "id": locate_solid()}}
    # MATERIALS_SOLID = {"MAT1": {"neohooke_compressible": {"mu": E/(2.*(1.+nu)), "nu": nu},
    #                             "inertia": {"rho0": 12.6e-3},
    #                             "id": locate_solid()}}

    # fluid1 is vapour (surrounding), fluid2 is liquid (bubble)
    rho1 = 1.26e-3 #0.0816 # pg/(µm^3) = 10^{-3} ng/(µm^3)
    rho2 = 1.26e-3 #0.2408 # pg/(µm^3) = 10^{-3} ng/(µm^3)
    eta1 = 1412e-3 # mPa s = 10^{-3} ng/(µm µs)
    eta2 = 1412e-3 # mPa s = 10^{-3} ng/(µm µs)
    sig = 46e-3 # surface energy density coefficient - mN/m = g/(s^2) = 10^{-3} ng/(µs^2)

    zeta = 0.0

    alpha = (rho1-rho2)/(rho1+rho2)
    sigtilde = sig #3.*sig/(2.*np.sqrt(2.))

    MATERIALS_FLUID = {"MAT1": {"newtonian": {"eta1": eta1, "eta2": eta2, "zeta1": zeta, "zeta2": zeta},
                                "inertia": {"rho1": rho1, "rho2": rho2},
                                "id": locate_fluid()}}

    MATERIALS_ALE = {"MAT1": {"diffusion": {"D": 1.0}, "id": locate_fluid()}}

    MATERIALS_PF = {"MAT1": {"mat_cahnhilliard": {"mobility": "degenerate",
                                                  "epsilon": 0.0,
                                                  "exponent": 1.0,
                                                  "M0": 0.1*eps**2.0,   # Mobility [m^5/(Pa s)]
                                                  "D": sigtilde/(4.*eps),         # Bulk free-energy parameter [Pa/m^3]
                                                  "kappa": sigtilde*eps,     # Gradient energy coefficient [Pa/m]
                                                  "alpha": alpha},   # Pressure factor in diffusive flux
                                                  "id": locate_fluid()}}

    class locate_all:
        def evaluate(self, x):
            return np.full(x.shape[1], True, dtype=bool)

    class locate_corner:
        def evaluate(self, x):
            ctr_x = np.isclose(x[0], 0.0)
            ctr_y = np.isclose(x[1], 0.0)
            return np.logical_and(ctr_x, ctr_y)

    # define your load curves here (syntax: tcX refers to curve X, to be used in BC_DICT key 'curve' : [X,0,0], or 'curve' : X)
    # class time_curves:
    #     def tc1(self, t):
    #         Tp = 0.5
    #         pmax = 3.0 # kPa
    #         return -pmax * np.sin(2.*np.pi*t/Tp)


    BC_DICT_SOLID = {
        "dirichlet": [{"id": [locate_bottom()], "dir": "all", "val": 0.0},
                      {"id": [locate_left(),locate_right()], "dir": "x", "val": 0.0}],
        }

    BC_DICT_FLUID = {
        "dirichlet": [{"id": [locate_top()], "dir": "all", "val": 0.0},
                      {"id": [locate_left(),locate_right()], "dir": "x", "val": 0.0}],
        # "dirichlet_pres" : [{"id": [locate_corner()], "dir": "all", "val": 0.0}],
    }

    BC_DICT_ALE = {
        "dirichlet": [{"id": [locate_top()], "dir": "all", "val": 0.0},
                      {"id": [locate_left(),locate_right()], "dir": "x", "val": 0.0}],
    }

    BC_DICT_PF = { }


    BC_DICT_LM = {"dirichlet": [{"id": [locate_left(),locate_right()], "dir": "x", "val": 0.0}]}

    problem = ambit_fe.ambit_main.Ambit(
        IO_PARAMS,
        CONTROL_PARAMS,
        [TIME_PARAMS_SOLID, TIME_PARAMS_FLUID, TIME_PARAMS_PF],
        SOLVER_PARAMS,
        [FEM_PARAMS_SOLID, FEM_PARAMS_FLUID, FEM_PARAMS_PF, FEM_PARAMS_ALE],
        [MATERIALS_SOLID, MATERIALS_FLUID, MATERIALS_PF, MATERIALS_ALE],
        [BC_DICT_SOLID, BC_DICT_FLUID, BC_DICT_PF, BC_DICT_ALE, BC_DICT_LM],
        # time_curves=time_curves(),
        coupling_params=COUPLING_PARAMS
    )

    # problem solve
    problem.solve_problem()


if __name__ == "__main__":
    main()
