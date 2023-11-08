#!/usr/bin/env python3

"""
- quasi-static solid mechanics
- compressible NeoHookean material
- force applied over an edge integral
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
    try: restart_step = int(sys.argv[1])
    except: restart_step = 0

    IO_PARAMS         = {'problem_type'          : 'solid',
                         'USE_MIXED_DOLFINX_BRANCH' : True,
                         'mesh_domain'           : basepath+'/input/block_domain.xdmf',
                         'mesh_boundary'         : basepath+'/input/block_boundary.xdmf',
                         'mesh_edge'             : basepath+'/input/block_edge.xdmf',
                         'write_results_every'   : 1,
                         'write_restart_every'   : -1,
                         'indicate_results_by'   : 'step',
                         'restart_step'          : restart_step,
                         'output_path'           : basepath+'/tmp/',
                         'results_to_write'      : ['displacement'],
                         'simname'               : 'solid_pinch_edgeEEEII'}

    SOLVER_PARAMS     = {'solve_type'            : 'direct',
                         'tol_res'               : 1.0e-8,
                         'tol_inc'               : 1.0e-8,
                         'maxiter'               : 25,
                         'divergence_continue'   : None}

    TIME_PARAMS       = {'maxtime'               : 1.0,
                         'numstep'               : 10,
                         'numstep_stop'          : 10,
                         'timint'                : 'static'}

    FEM_PARAMS        = {'order_disp'            : 1,
                         'quad_degree'           : 2,
                         'incompressible_2field' : False}

    #MATERIALS         = {'MAT1' : {'neohooke_compressible' : {'mu' : 10., 'nu' : 0.1}}}
    MATERIALS         = {'MAT1' : {'neohooke_dev' : {'mu' : 10.},
                                   'sussmanbathe_vol' : {'kappa' : 500.}}}


    # define your load curves here (syntax: tcX refers to curve X, to be used in BC_DICT key 'curve' : [X,0,0], or 'curve' : X)
    class time_curves():

        def tc1(self, t):
            return -8.*t

    BC_DICT           = { 'neumann' : [{'id' : [6], 'dir' : 'xyz_ref', 'curve' : [0,0,1]}],
                          'dirichlet' : [{'id' : [3], 'dir' : 'all', 'val' : 0.0},
                                         {'id' : [1], 'dir' : 'all', 'val' : 0.0, 'codimension' : 1}]}#,
                          #'robin' : [{'id' : [1], 'dir' : 'xyz_ref', 'type' : 'spring', 'stiff' : 1e3, 'codimension' : 1}]}


    # problem setup
    problem = ambit_fe.ambit_main.Ambit(IO_PARAMS, TIME_PARAMS, SOLVER_PARAMS, FEM_PARAMS, MATERIALS, BC_DICT, time_curves=time_curves())

    # solve time-dependent problem
    problem.solve_problem()


    # --- results check
    tol = 1.0e-6

    check_node = []
    check_node.append(np.array([0.5, 0.5, 1.0]))

    u_corr = np.zeros(3*len(check_node))

    ## correct results
    u_corr[0] = 6.00095441680302044e-01 # x
    u_corr[1] = -1.0862313365225019e-07 # y
    u_corr[2] = -0.000897803340365617 # z

    check1 = ambit_fe.resultcheck.results_check_node(problem.mp.u, check_node, u_corr, problem.mp.V_u, problem.mp.comm, tol=tol, nm='u')
    success = ambit_fe.resultcheck.success_check([check1], problem.mp.comm)

    if not success:
        raise RuntimeError("Test failed!")



if __name__ == "__main__":

    test_main()
