#!/usr/bin/env python3

"""
tests solver divergence continue action with PTC on detection of large residual values/nans
"""

import ambit_fe

import sys
import numpy as np
from pathlib import Path
import pytest


@pytest.mark.solid
def test_main():

    basepath = str(Path(__file__).parent.absolute())

    IO_PARAMS         = {'problem_type'          : 'solid',
                         'mesh_domain'           : basepath+'/input/block2_domain.xdmf',
                         'mesh_boundary'         : basepath+'/input/block2_boundary.xdmf',
                         'write_results_every'   : -999,
                         'output_path'           : basepath+'/tmp/',
                         'results_to_write'      : ['displacement'],
                         'simname'               : 'solid_divcont_ptc'}

    SOLVER_PARAMS     = {'solve_type'            : 'direct',
                         'divergence_continue'   : 'PTC',
                         'k_ptc_initial'         : 10.,
                         'tol_res'               : 1.0e-8,
                         'tol_inc'               : 1.0e-8,
                         'maxiter'               : 25}

    TIME_PARAMS       = {'maxtime'               : 1.0,
                         'numstep'               : 10,
                         'numstep_stop'          : 1,
                         'timint'                : 'genalpha',
                         'rho_inf_genalpha'      : 1.0}

    FEM_PARAMS        = {'order_disp'            : 1,
                         'quad_degree'           : 1}

    MATERIALS         = {'MAT1' : {'exponential_dev'  : {'a_0' : 10., 'b_0' : 120.},
                                   'sussmanbathe_vol' : {'kappa' : 1.0e3},
                                   'inertia'          : {'rho0' : 1.0e-6}}}


    # define your load curves here (syntax: tcX refers to curve X, to be used in BC_DICT key 'curve' : [X,0,0], or 'curve' : X)
    class time_curves():

        def tc1(self, t):
            return -60.*t

    BC_DICT           = { 'dirichlet' : [{'id' : [2], 'dir' : 'all', 'val' : 0.}],
                          'neumann' :   [{'id' : [3], 'dir' : 'xyz_ref', 'curve' : [0,1,1]}] }


    # problem setup
    problem = ambit_fe.ambit_main.Ambit(IO_PARAMS, TIME_PARAMS, SOLVER_PARAMS, FEM_PARAMS, MATERIALS, BC_DICT, time_curves=time_curves())

    # solve time-dependent problem
    problem.solve_problem()


    # --- results check
    tol = 1.0e-6

    check_node = []
    check_node.append(np.array([1.0, 1.0, 10.0]))

    u_corr = np.zeros(3*len(check_node))

    ## correct results
    u_corr[0] = 6.2386802737805072E-02 # x
    u_corr[1] = -2.2693049759162623E-01 # y
    u_corr[2] = -1.7719786920313996E-01 # z

    check1 = ambit_fe.resultcheck.results_check_node(problem.mp.u, check_node, u_corr, problem.mp.V_u, problem.mp.comm, tol=tol, nm='u')
    success = ambit_fe.resultcheck.success_check([check1], problem.mp.comm)

    if not success:
        raise RuntimeError("Test failed!")



if __name__ == "__main__":

    test_main()
