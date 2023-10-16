#!/usr/bin/env python3

"""
steel block falling for 1 s due to Earth gravity, acting in negative z direction
"""

import ambit_fe

import sys
import numpy as np
from pathlib import Path
import pytest


@pytest.mark.solid
def test_main():

    basepath = str(Path(__file__).parent.absolute())

    # reads in restart step from the command line
    try: restart_step = int(sys.argv[1])
    except: restart_step = 0

    IO_PARAMS         = {'problem_type'          : 'solid',
                         'mesh_domain'           : basepath+'/input/block2_domain.xdmf',
                         'mesh_boundary'         : basepath+'/input/block2_boundary.xdmf',
                         'write_results_every'   : 1,
                         'write_restart_every'   : 1,
                         'restart_step'          : restart_step,
                         'output_path'           : basepath+'/tmp/',
                         'results_to_write'      : ['displacement','velocity','acceleration'],
                         'simname'               : 'solid_bodyforce_gravity'}

    SOLVER_PARAMS     = {'solve_type'            : 'direct',
                         'tol_res'               : 1.0e-3,
                         'tol_inc'               : 1.0e-10,
                         'maxiter'               : 25,
                         'divergence_continue'   : None}

    TIME_PARAMS       = {'maxtime'               : 1.0,
                         'numstep'               : 10,
                         'timint'                : 'genalpha',
                         'rho_inf_genalpha'      : 1.0}

    FEM_PARAMS        = {'order_disp'            : 1,
                         'quad_degree'           : 2,
                         'incompressible_2field' : False}

    MATERIALS         = {'MAT1' : {'stvenantkirchhoff' : {'Emod' : 210e9, 'nu' : 0.3}, # Pa
                                   'inertia'           : {'rho0' : 7850.}}} # kg/m^3


    # define your load curves here (syntax: tcX refers to curve X, to be used in BC_DICT key 'curve' : [X,0,0], or 'curve' : X)
    class time_curves():

        def tc1(self, t):
            return MATERIALS['MAT1']['inertia']['rho0'] * 9.81 # 9.81 m/s^2


    BC_DICT           = { 'bodyforce' : [{'id' : [1], 'dir' : [0.,0.,-1.], 'curve' : 1}] }


    # problem setup
    problem = ambit_fe.ambit_main.Ambit(IO_PARAMS, TIME_PARAMS, SOLVER_PARAMS, FEM_PARAMS, MATERIALS, BC_DICT, time_curves=time_curves())

    # solve time-dependent problem
    problem.solve_problem()


    # --- results check
    tol_u, tol_v, tol_a = 1.0e-8, 1e-4, 1e-4

    check_node = []
    check_node.append(np.array([-1.0000000000000000e+00, 1.0000000000000000e+00, 1.0000000000000000e+01]))

    u_corr, v_corr, a_corr = np.zeros(3*len(check_node)), np.zeros(3*len(check_node)), np.zeros(3*len(check_node))

    ## correct results
    u_corr[0] = 0.0 # x
    u_corr[1] = 0.0 # y
    u_corr[2] = -4.905 # z - should be -0.5*g*t^2
    
    v_corr[0] = 0.0 # x
    v_corr[1] = 0.0 # y
    v_corr[2] = -9.81 # z - should be -g*t

    a_corr[0] = 0.0 # x
    a_corr[1] = 0.0 # y
    a_corr[2] = -9.81 # z - should be -g

    check1 = ambit_fe.resultcheck.results_check_node(problem.mp.u, check_node, u_corr, problem.mp.V_u, problem.mp.comm, tol=tol_u, nm='u')
    check2 = ambit_fe.resultcheck.results_check_node(problem.mp.io.v_proj, check_node, v_corr, problem.mp.V_u, problem.mp.comm, tol=tol_v, nm='v')
    check3 = ambit_fe.resultcheck.results_check_node(problem.mp.io.a_proj, check_node, a_corr, problem.mp.V_u, problem.mp.comm, tol=tol_a, nm='a')
    success = ambit_fe.resultcheck.success_check([check1,check2,check3], problem.mp.comm)

    if not success:
        raise RuntimeError("Test failed!")



if __name__ == "__main__":

    test_main()
