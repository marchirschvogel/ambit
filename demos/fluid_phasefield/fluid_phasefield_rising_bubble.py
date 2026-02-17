#!/usr/bin/env python3

"""
Two-phase flow rising bubble in gravitational field
"""

import ambit_fe

import numpy as np
from pathlib import Path


def main():
    basepath = str(Path(__file__).parent.absolute())

    # cases (1,2) from Eikelder et al. (2024), Brunk and Eikelder (2026)
    case = 1

    IO_PARAMS = {
        "problem_type": "fluid_phasefield",
        "write_results_every": 1,
        "write_restart_every": -1,
        "restart_step": 0,
        "indicate_results_by": "step0",
        "output_path": basepath + "/tmp/",
        "mesh_domain": {"type":"rectangle", "celltype":"quadrilateral", "coords_a":[0.0, 0.0], "coords_b":[1.0, 2.0], "meshsize":[64,128]}, # 32,64   64,128   128,256
        "results_to_write": [["velocity", "pressure", "density"],["phase", "potential"]],
        "simname": "fluid_phasefield_rising_bubble"+str(case)+"_exp1.0_BDF2_eps1.28_-11",
        "write_initial_fields": True,
        "report_conservation_properties": True,
    }

    # add elements in x direction to output name
    IO_PARAMS["simname"] += "_elx"+str(IO_PARAMS["mesh_domain"]["meshsize"][0])

    h = 1.0/IO_PARAMS["mesh_domain"]["meshsize"][0]
    eps = 0.64*h # 1.28*h (Eikelder et al. (2024)), 0.64*h (Brunk and Eikelder (2026))

    class expr1:
        def __init__(self):
            self.t = 0
            self.R_0 = 0.25
            self.x_c = np.asarray([0.5, 0.5, 0.0])

        def evaluate(self, x):
            d = np.sqrt( (x[0]-self.x_c[0])**2.0 + (x[1]-self.x_c[1])**2.0 + (x[2]-self.x_c[2])**2.0 )
            # val = 0.5*(1.0 + np.tanh((self.R_0 - d)/(np.sqrt(2.0)*eps)))  # if phi in [0,1]
            val = np.tanh((self.R_0 - d)/(np.sqrt(2.0)*eps))  # if phi in [-1,1]
            return (
                np.full(x.shape[1], val),
            )

    CONTROL_PARAMS = {"maxtime": 3.0,
                      "dt": 0.128*h, # from Eikelder et al. (2024)
                      # "numstep_stop": 5,
                      "initial_fields": [expr1, None],
                      }

    SOLVER_PARAMS = {
        "solve_type": "direct",
        "direct_solver": "mumps",
        "maxiter": 10,
        "tol_res": [1e-5, 1e-5, 1e-5, 1e-5],
        "tol_inc": [1e-3, 1e-3, 1e-3, 1e-3],
        "ignore_unconverged": True, # badass, but we might have some time steps that stagnate... :-/
    }

    TIME_PARAMS_FLUID = {"timint": "bdf2",
                         "theta_ost": 0.5, # not used (only for OST timint)
                         "fluid_governing_type": "navierstokes_transient",
                         "eval_nonlin_terms": "midpoint", # midpoint, trapezoidal - irrelevant for BDF2 scheme
                         "continuity_at_midpoint": True} # Should use midpoint if time derivative (drho/dt) is involved... irrelevant for BDF2 scheme

    TIME_PARAMS_PF = {"timint": "bdf2",
                      "theta_ost": 0.5, # not used (only for OST timint)
                      "eval_nonlin_terms": "midpoint", # midpoint, trapezoidal - irrelevant for BDF2 scheme
                      "potential_at_midpoint": False} # irrelevant for BDF2 scheme

    FEM_PARAMS_FLUID = {"order_vel": 2,
                        "order_pres": 1,
                        "quad_degree": 9,
                        "fluid_formulation": "conservative"}

    FEM_PARAMS_PF = {"order_phi": 1, "order_mu": 1, "quad_degree": 9,
                     "phi_range" : [-1.0, 1.0]}   # [-1.0, 1.0], [0.0, 1.0]

    # fluid1 is surrounding, fluid2 is bubble
    if case==1:
        rho1 = 1000.0
        rho2 = 100.0
        eta1 = 10.0
        eta2 = 1.0
        sig = 24.5 # surface energy density coefficient
    elif case==2:
        rho1 = 1000.0
        rho2 = 1.0
        eta1 = 10.0   # 10.0, 1.0 TODO: Ambiguity code (10.0) vs. papers (1.0)
        eta2 = 0.1
        sig = 1.96 # surface energy density coefficient
    else:
        raise ValueError("Unknown case.")
    zeta = 0.0

    alpha = (rho1-rho2)/(rho1+rho2) # TODO: Negative in Eikelder et al. (2024), Brunk and Eikelder (2026), but then not working!
    sigtilde = 3.*sig/(2.*np.sqrt(2.))

    MATERIALS_FLUID = {"MAT1": {"newtonian": {"eta1": eta1, "eta2": eta2, "zeta1": zeta, "zeta2": zeta},
                                "inertia": {"rho1": rho1, "rho2": rho2}}}

    MATERIALS_PF = {"MAT1": {"mat_cahnhilliard": {"mobility": "degenerate",
                                                  "epsilon": 0.0,
                                                  "exponent": 1.0,
                                                  "M0": 0.1*eps**2.0,   # Mobility [m^5/(Pa s)]
                                                  "D": sigtilde/(4.*eps),         # Bulk free-energy parameter [Pa/m^3]
                                                  "kappa": sigtilde*eps,     # Gradient energy coefficient [Pa/m]
                                                  "alpha": alpha}}}   # Pressure factor in diffusive flux

    class locate_top_bottom:
        def evaluate(self, x):
            top_b = np.isclose(x[1], 2.0)
            bottom_b = np.isclose(x[1], 0.0)
            return np.logical_or(top_b, bottom_b)

    class locate_left_right:
        def evaluate(self, x):
            left_b = np.isclose(x[0], 0.0)
            right_b = np.isclose(x[0], 1.0)
            return np.logical_or(left_b, right_b)

    class locate_all:
        def evaluate(self, x):
            return np.full(x.shape[1], True, dtype=bool)

    class locate_center:
        def evaluate(self, x):
            ctr_x = np.isclose(x[0], 0.5)
            ctr_y = np.isclose(x[1], 1.0)
            return np.logical_and(ctr_x, ctr_y)

    BC_DICT_FLUID = {
        "dirichlet" : [{"locator": locate_top_bottom(), "dir": "all", "val": 0.0},
                       {"locator": locate_left_right(), "dir": "x", "val": 0.0}],
        "dirichlet_pres" : [{"locator": locate_center(), "dir": "all", "val": 0.0}], # fix pressure in middle of domain to have a well-defined pressure level
        "bodyforce" : [{"locator": locate_all(), "dir": [0.0, -1.0, 0.0], "val": 0.98, "scale_density": True}],
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
        # time_curves=time_curves(),
    )

    # problem solve
    problem.solve_problem()


if __name__ == "__main__":
    main()
