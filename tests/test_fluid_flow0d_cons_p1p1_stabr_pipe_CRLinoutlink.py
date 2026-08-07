#!/usr/bin/env python3

"""
Transient incompressible Navier-Stokes flow in a pipe with two separate regions connected via a 0D model
- stabilized P1P1 elements for velocity and pressure (reduced SUPF/PSPG, G2 scheme)
- 2 material domains in fluid with different densities and viscosities
- Generalized-alpha time-integration for fluid
- Conservative Navier-Stokes
"""

import ambit_fe

import sys
import numpy as np
from pathlib import Path
import pytest


@pytest.mark.fluid_flow0d
def test_main():
    basepath = str(Path(__file__).parent.absolute())

    # reads in restart step from the command line
    try:
        restart_step = int(sys.argv[1])
    except:
        restart_step = 0

    IO_PARAMS = {
        "problem_type": "fluid_flow0d",
        "duplicate_mesh_domains": [[1], [2]],
        "mesh_domain": basepath + "/input/pipe_fluid_domain.xdmf",
        "mesh_boundary": basepath + "/input/pipe_fluid_boundary.xdmf",
        "mesh_encoding": "HDF5",
        "write_results_every": 1,
        "write_restart_every": 9,
        "restart_step": restart_step,
        "output_path": basepath + "/tmp/",
        "results_to_write": ["velocity", "pressure"],
        "simname": "fluid_flow0d_cons_p1p1_stabr_pipe_CRLinoutlink",
    }

    CONTROL_PARAMS = {"maxtime": 0.2,
                      "numstep": 100,
                      "numstep_stop": 10}

    SOLVER_PARAMS = {
        "solve_type": "direct",
        "direct_solver": "mumps",
        "tol_res": 1.0e-8,
        "tol_inc": 1.0e-8,
        "subsolver_params": {"tol_res" : 1.0e-6, "tol_inc" : 1.0e-6},
    }

    TIME_PARAMS = {
        "timint": "genalpha",
        "rho_inf_genalpha": 0.8,
        "fluid_governing_type": "navierstokes_transient",
        "eval_nonlin_terms": "midpoint",
        "discretely_conservative": True,  # both time scheme variants equal for constant density and Eulerian frame (would matter in ALE frame)!
    }

    TIME_PARAMS_FLOW0D   = {"timint"                : "ost",
                            "theta_ost"             : 0.67,
                            "initial_backwardeuler" : True,
                            "initial_conditions"    : {"q_in_0" : 0.0, "q_d_0" : 0.0, "p_d_0" : 0.0, "q_out_0" : 0.0}}

    MODEL_PARAMS_FLOW0D  = {"modeltype"             : "CRLinoutlink",
                            "parameters"            : {"C_in" : 1e1, "R_in" : 100e-6, "L_in" : 1e-7, "C_out" : 0.01, "R_out" : 1e-6, "L_out" : 1e-8}}

    FEM_PARAMS = {
        "order_vel": 1,
        "order_pres": 1,
        "quad_degree": 5,
        "fluid_formulation": "conservative",  # nonconservative (default), conservative
        "stabilization": {
            "scheme": "supg_pspg",
            "vscale": 1e3,
            "reduced_scheme": True,
            "dscales": [1.0, 1.0, 1.0],
        },
    }

    COUPLING_PARAMS_FLUID_FLOW0D = {"interfaces"    : [[4],[6]],
                            "coupling_quantity"     : ["pressure"]*2,
                            "variable_quantity"     : ["flux"]*2,
                            "cq_factor"             : [1.,-1.], # out-flow positive, in-flow negative
                            "coupling_type"         : "monolithic_lagrange",
                            "print_subiter"         : False,
                            "condense_0d_model"     : "no"}

    MATERIALS = {
        "MAT1": {"newtonian": {"eta": 1.0e-3}, "inertia": {"rho": 1.0e-3}},
        "MAT2": {"newtonian": {"eta": 1.0e-6}, "inertia": {"rho": 1.0e-6}},
    }


    class expression1:
        def __init__(self):

            self.t = 0.0

            self.T = 0.4
            self.vmax = 1e3

            self.r = 15.0 # pipe radius

        def evaluate(self, x):

            vel_inflow_xy = (x[0]**2. - self.r**2.)*(x[1]**2. - self.r**2.) / (self.r**4.) # parabolic inflow profile

            val_t = 0.5*self.vmax*(1.-np.cos(2.*np.pi*self.t/self.T)) * vel_inflow_xy

            return ( np.full(x.shape[1], 0.0),
                     np.full(x.shape[1], 0.0),
                     np.full(x.shape[1], val_t) )

    BC_DICT = {
        # ids: 1,5: lateral wall - 2: inflow, 5: axial outflow, 6: top outflow, 7: bottom outflow, 3: valve
        "dirichlet" : [{"id" : [2], "dir" : "all", "expression" : expression1}, # inflow
                                            {"id" : [1,5], "dir" : "all", "val" : 0.}, # lateral wall
                                            {"id" : [3], "dir" : "all", "val" : 0.}], # inner (valve) plane
                             "stabilized_neumann" : [{"id" : [4,6,7], "beta" : 0.2e-6, "gamma" : 1.}],
                             "neumann" : [{"id" : [7], "dir" : "normal_cur", "val" : 0.1}],  # constant Neumann pressure that produces an initial acceleration != 0 for testing purposes (relevant in gen-alpha for rho_inf < 1)
        "dp_monitor": [{"id": [3], "upstream_domain": 2, "downstream_domain": 1}],
        "flux_monitor": [{"id": [3], "on_subdomain": True, "internal": False, "domain": 2}],
    }

    # problem setup
    problem = ambit_fe.ambit_main.Ambit(
        IO_PARAMS,
        CONTROL_PARAMS,
        [[TIME_PARAMS], TIME_PARAMS_FLOW0D],
        SOLVER_PARAMS,
        [FEM_PARAMS],
        [[MATERIALS], MODEL_PARAMS_FLOW0D],
        [BC_DICT],
        coupling_params=COUPLING_PARAMS_FLUID_FLOW0D,
    )

    # solve time-dependent problem
    problem.solve_problem()

    # --- results check
    tol = 1.0e-6

    check_node = []
    check_node.append(np.array([0.0096063, 15.0, 67.0447]))

    v_corr = np.zeros(3 * len(check_node))

    # correct results
    v_corr[0] = 3.4097479289218411E+00  # x
    v_corr[1] = 9.1986656828994683E+01  # y
    v_corr[2] = -2.4095044371396881E+01  # z

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
