#!/usr/bin/env python3

"""
solid mechanics, poroelasticity on unit sphere
"""

import ambit_fe

import sys
import numpy as np
from pathlib import Path
import pytest


@pytest.mark.solid
@pytest.mark.skip(reason="Not yet ready for testing.")
def test_main():
    basepath = str(Path(__file__).parent.absolute())

    # reads in restart step from the command line
    try:
        restart_step = int(sys.argv[1])
    except:
        restart_step = 0

    IO_PARAMS = {
        "problem_type": "solid",
        "mesh_domain": {"type": "rectangle", "celltype": "quadrilateral", "coords_a": [0.0, 0.0], "coords_b": [0.127, 0.127], "meshsize": [160,160]},
        "indicate_results_by": "step0",
        "write_results_every": 1,
        "write_restart_every": -1,
        "restart_step": restart_step,
        "output_path": basepath + "/tmp/",
        "results_to_write": {"solid": ["displacement", "porehydpressure", "vonmises_cauchystress"], "scatra": {"concentration"}},
        "simname": "solid_poro_dary_schloegl_nernst_planck_electrostatics",
    }

    CONTROL_PARAMS = {"maxtime": 4.0,
                      "numstep": 100,
                      # "numstep_stop": 5,
                      }

    SOLVER_PARAMS = {
        "solve_type": "direct",
        "direct_solver": "mumps",
        "tol_res": 1.0e-8,
        "tol_inc": 1.0e-6,
    }

    TIME_PARAMS = {
        "timint": "ost",
        "theta_ost": 1.0,
    }

    TIME_PARAMS_SC = [{"timint": "ost", "theta_ost": 1.0},  # Nernst-Planck
                      {"timint": "static"}]  # electrostatics

    FEM_PARAMS = {
        "order_disp": 2,
        "order_phyd": 1,
        "order_pres": 1,
        "quad_degree": 5,
        "incompressibility": "no",
        "poroelasticity": {"model": "darcy_schloegl",
                           "coupled_c_osmotic": "c1",
                           "coupled_c_electric": "c2"},
    }

    FEM_PARAMS_SC = {
        "order_conc": 1,
        "quad_degree": 5,
    }

    class locate_all:
        def evaluate(self, x):
            return np.full(x.shape[1], True, dtype=bool)

    # Antonini et al. parameters
    nS_0S = 0.61 #-
    KS0   = 4.77e-14 #mm2
    etaFR = 3.55e-10 #N*s/mm2
    nF_0S = 1.0 - nS_0S #-
    epsF = 100.0 #C/Nmm2
    F = 96485.0 #C/mol
    z_fc = -1.0 #-
    z_proton = 1.0 #-
    teta = 353.0 #K
    MmH = 1.008e-3 #kg/mol
    R = 8314.4 #(N*mm)/(mol*K)
    D_proton = 1.35e-3 #mm2/s
    cmfc_0S = 2.8e-6 #mol/mm3
    cmH20 = 55.5e-6 #mol/mm3


    mu, kappa = 110., 330.  # 110 and 330 MPa (=N/mm^2), cf. Antonini et al. (2025)
    MATERIALS = {
        "MAT1": {
            "neohooke_dev": {"mu": mu},
            "ogden_vol": {"kappa": kappa},
            "inertia": {"rho0": 1e-6},
            "MAT_PORO": {"darcy_schloegl": {"k": KS0/etaFR, "k_os": R*teta, "k_el": z_proton*F}},
            "id": locate_all(),
        }
    }

                     # Proton concentration
    MATERIALS_SC = [{"MAT1": {"diffusion_grad_c":       {"D": D_proton},  # Nernst-Planck equation for flux of protons
                              # "diffusion_c_grad_ccoup": {"D": D_proton*z_proton*F/(R*teta), "cc": "c2"},
                              "id": locate_all()}},
                     # Electric Potential
                    {"MAT1": {"diffusion_grad_c": {"D": 1.0},  # Poisson equation of electrostatics
                              "source": {"type": "coup", "val": F*z_proton/epsF, "cc": "c1"},
                              "id": locate_all()}}]

    # define your load curves here (syntax: tcX refers to curve X, to be used in BC_DICT key 'curve' : [X,0,0], or 'curve' : X)
    class time_curves:
        def tc1(self, t):
            pmax = -1.0
            return pmax * t

    # locators for boundary conditions
    class locate_left:
        def evaluate(self, x):
            return np.isclose(x[0], 0.0)
    class locate_right:
        def evaluate(self, x):
            return np.isclose(x[0], 0.127)

    class locate_bottom:
        def evaluate(self, x):
            return np.isclose(x[1], 0.0)
    class locate_top:
        def evaluate(self, x):
            return np.isclose(x[1], 0.127)

    class expression_neu:
        def __init__(self):
            self.t = 0.0
            self.t_ramp = 1.0

            self.h = 0.127
            self.dy = 0.005
        def evaluate(self, x):
            g0 = -50.0
            val_t = g0 * 0.5 * (1.0 - np.cos(np.pi * self.t / self.t_ramp)) * (
                self.t < self.t_ramp
            ) + g0 * (self.t >= self.t_ramp)

            val = val_t * (x[1]>=0.1*self.h-self.dy/2.)*(x[1]<0.1*self.h+self.dy/2.) + \
                  val_t * (x[1]>=0.3*self.h-self.dy/2.)*(x[1]<0.3*self.h+self.dy/2.) + \
                  val_t * (x[1]>=0.5*self.h-self.dy/2.)*(x[1]<0.5*self.h+self.dy/2.) + \
                  val_t * (x[1]>=0.7*self.h-self.dy/2.)*(x[1]<0.7*self.h+self.dy/2.) + \
                  val_t * (x[1]>=0.9*self.h-self.dy/2.)*(x[1]<0.9*self.h+self.dy/2.)
            return (np.full(x.shape[1], val))

    class expr_dbc_poro:
        def __init__(self):
            self.t = 0.0
            self.t_off = 1.0
            self.t_ramp = 0.5
            self.p0 = 0.0
            self.p1 = 0.1
        def evaluate(self, x):
            val = self.p0*(self.t < self.t_off) + (self.p0 + (self.p1-self.p0) * (self.t-self.t_off)/self.t_ramp) * (self.t >= self.t_off)*(self.t < self.t_off+self.t_ramp) + self.p1*(self.t >= self.t_off+self.t_ramp)
            return (np.full(x.shape[1], val))

    class expr_dbc_cm:
        def __init__(self):
            self.t = 0.0
            self.t_off = 2.0
            self.t_ramp = 0.5
            self.cm0 = 2.8e-6
            self.cm1 = 2.9e-6
        def evaluate(self, x):
            val = self.cm0*(self.t < self.t_off) + (self.cm0 + (self.cm1-self.cm0) * (self.t-self.t_off)/self.t_ramp) * (self.t >= self.t_off)*(self.t < self.t_off+self.t_ramp) + self.cm1*(self.t >= self.t_off+self.t_ramp)
            return (np.full(x.shape[1], val))


    class expr_dbc_phie:
        def __init__(self):
            self.t = 0.0
            self.t_off = 2.0
            self.t_ramp = 0.5
            self.phie0 = 0.0
            self.phie1 = 247.0
        def evaluate(self, x):
            val = self.phie0*(self.t < self.t_off) + (self.phie0 + (self.phie1-self.phie0) * (self.t-self.t_off)/self.t_ramp) * (self.t >= self.t_off)*(self.t < self.t_off+self.t_ramp) + self.phie1*(self.t >= self.t_off+self.t_ramp)
            return (np.full(x.shape[1], val))



    BC_DICT = {
        "dirichlet": [{"id": [locate_right()], "dir": "x", "val": 0.0},
                      {"id": [locate_bottom(),locate_top()], "dir": "y", "val": 0.0},
                      ],
        "dirichlet_poro": [{"id": [locate_left(),locate_right()], "expression": expr_dbc_poro}],
        "neumann": [{"id": [locate_left()], "dir": "normal_cur", "expression": expression_neu}],
    }

    BC_DICT_SC = {"dirichlet_c1": [
                                   {"id": [locate_left()], "expression": expr_dbc_cm},
                                   {"id": [locate_right()], "val": 2.8e-6},  # mol/mm^3
                                   ],
                  "dirichlet_c2": [
                                   {"id": [locate_left()], "expression": expr_dbc_phie},  # mV
                                   {"id": [locate_right()], "val": 0.0},  # mV
                                   ]
                  }

    # problem setup
    problem = ambit_fe.ambit_main.Ambit(
        IO_PARAMS,
        CONTROL_PARAMS,
        [TIME_PARAMS, TIME_PARAMS_SC],
        SOLVER_PARAMS,
        [FEM_PARAMS, FEM_PARAMS_SC],
        [MATERIALS, MATERIALS_SC],
        [BC_DICT, BC_DICT_SC],
        time_curves=time_curves(),
    )

    # solve time-dependent problem
    problem.solve_problem()

    # # --- results check
    # tol = 1.0e-6
    #
    # check_node, check_node2 = [], []
    # check_node.append(np.array([0.5806386442505, 0.5806386442505002, 0.5707166806795334]))
    # check_node2.append(np.array([0.0, 0.0, 0.0]))
    #
    # u_corr, p_corr = np.zeros(3 * len(check_node)), np.zeros(len(check_node))
    #
    # ## correct results
    # u_corr[0] = -8.1050317190939653E-03  # x
    # u_corr[1] = -8.1050317001130426E-03  # y
    # u_corr[2] = -7.9805588796660921E-03  # z
    #
    # p_corr[0] = 5.5705312265436846E-01
    #
    # check1 = ambit_fe.resultcheck.results_check_node(
    #     problem.mp.u,
    #     check_node,
    #     u_corr,
    #     problem.mp.V_u,
    #     problem.mp.comm,
    #     tol=tol,
    #     nm="u",
    # )
    # check2 = ambit_fe.resultcheck.results_check_node(
    #     problem.mp.phyd,
    #     check_node2,
    #     p_corr,
    #     problem.mp.V_phyd,
    #     problem.mp.comm,
    #     tol=tol,
    #     nm="p",
    # )
    # success = ambit_fe.resultcheck.success_check([check1, check2], problem.mp.comm)
    #
    # if not success:
    #     raise RuntimeError("Test failed!")


if __name__ == "__main__":
    test_main()
