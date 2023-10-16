#!/usr/bin/env python3

# tests:
# - MULF prestressing (Gee et al. 2010) in displacement formulation (Schein & Gee 2021)

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
                         'results_to_write'      : [''],
                         'simname'               : 'solid_robin_static_prestress'}

    SOLVER_PARAMS     = {'solve_type'            : 'direct',
                         'tol_res'               : 1.0e-8,
                         'tol_inc'               : 1.0e-8}

    TIME_PARAMS       = {'maxtime'               : 1.0,
                         'numstep'               : 1,
                         'timint'                : 'static'}

    FEM_PARAMS        = {'order_disp'            : 1,
                         'order_pres'            : 1,
                         'quad_degree'           : 2,
                         'incompressible_2field' : False,
                         'prestress_initial'     : True,
                         'prestress_numstep'     : 1}

    MATERIALS         = {'MAT1' : {'stvenantkirchhoff' : {'Emod' : 1000., 'nu' : 0.3}}}


    # define your load curves here (syntax: tcX refers to curve X, to be used in BC_DICT key 'curve' : [X,0,0], or 'curve' : X)
    class time_curves():

        def tc1(self, t): # prestress
            return 3.*t

        def tc2(self, t): # poststress
            return 3.

    BC_DICT           = { 'dirichlet' : [{'id' : [1,2,3], 'dir' : 'z', 'val' : 0.}],
                          'neumann_prestress' : [{'id' : [3], 'dir' : 'xyz_ref', 'curve' : [1,0,0]}],
                          'neumann' : [{'id' : [3], 'dir' : 'xyz_ref', 'curve' : [2,0,0]}],
                          'robin' : [{'type' : 'spring', 'id' : [1,2], 'dir' : 'normal_ref', 'stiff' : 5.0}] }


    # problem setup
    problem = ambit_fe.ambit_main.Ambit(IO_PARAMS, TIME_PARAMS, SOLVER_PARAMS, FEM_PARAMS, MATERIALS, BC_DICT, time_curves=time_curves())

    # solve time-dependent problem
    problem.solve_problem()


    # --- results check
    tol = 1.0e-6

    check_node = []
    check_node.append(np.array([-1.0000000000000000e+00, -1.0000000000000000e+00, 1.0000000000000000e+01]))

    u_corr = np.zeros(3*len(check_node))

    ## correct results
    u_corr[0] = 0.0 # x
    u_corr[1] = 0.0 # y
    u_corr[2] = 0.0 # z

    check1 = ambit_fe.resultcheck.results_check_node(problem.mp.u, check_node, u_corr, problem.mp.V_u, problem.mp.comm, tol=tol, nm='u')
    success = ambit_fe.resultcheck.success_check([check1], problem.mp.comm)

    if not success:
        raise RuntimeError("Test failed!")



if __name__ == "__main__":

    test_main()
