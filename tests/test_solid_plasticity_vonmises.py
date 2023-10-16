#!/usr/bin/env python3

"""
tests:
- solid mechanics: finite strain plasticity
TODO:  NOT YET FULLY IMPLEMENTED!
"""

import ambit_fe

import sys, traceback
import numpy as np
from pathlib import Path
import pytest


@pytest.mark.solid
@pytest.mark.skip(reason="Not yet ready for testing.")
def main():

    basepath = str(Path(__file__).parent.absolute())

    IO_PARAMS         = {'problem_type'          : 'solid',
                         'mesh_domain'           : basepath+'/input/block_domain.xdmf',
                         'mesh_boundary'         : basepath+'/input/block_boundary.xdmf',
                         'write_results_every'   : 1,
                         'output_path'           : basepath+'/tmp/',
                         'results_to_write'      : ['displacement'],
                         'simname'               : 'solid_plasticity_vonmises'}

    SOLVER_PARAMS     = {'solve_type'            : 'direct',
                         'tol_res'               : 1.0e-8,
                         'tol_inc'               : 1.0e-8,
                         'maxiter'               : 25,
                         'divergence_continue'   : None}

    TIME_PARAMS       = {'maxtime'               : 1.0,
                         'numstep'               : 10,
                         'numstep_stop'          : 10,
                         'timint'                : 'static',
                         'rho_inf_genalpha'      : 1.0}

    FEM_PARAMS        = {'order_disp'            : 1,
                         'quad_degree'           : 1,
                         'incompressible_2field' : False}

    MATERIALS         = {'MAT1' : {'neohooke_dev' : {'mu' : 10.},
                                   'ogden_vol'    : {'kappa' : 10./(1.-2.*0.49)},
                                   'plastic'      : {'type' : 'vonmises', 'yield_stress' : 1.0}}}


    # define your load curves here (syntax: tcX refers to curve X, to be used in BC_DICT key 'curve' : [X,0,0], or 'curve' : X)
    class time_curves():

        def tc1(self, t):
            umax = -0.25
            return 0.5*umax*(1.-np.cos(2.*np.pi*t))

    BC_DICT           = { 'dirichlet' : [{'id' : [1], 'dir' : 'all', 'val' : 0.0}, {'id' : [4], 'dir' : 'x', 'curve' : 1}] }


    # problem setup
    problem = ambit_fe.ambit_main.Ambit(IO_PARAMS, TIME_PARAMS, SOLVER_PARAMS, FEM_PARAMS, MATERIALS, BC_DICT, time_curves=time_curves())

    # solve time-dependent problem
    problem.solve_problem()


    ## --- results check
    #tol = 1.0e-6

    #check_node = []
    #check_node.append(np.array([-1.0000000000000000e+00, -1.0000000000000000e+00, 1.0000000000000000e+01]))

    #u_corr = np.zeros(3*len(check_node))

    ### correct results
    #u_corr[0] = 6.00095441680302044e-01 # x
    #u_corr[1] = -1.0862313365225019e-07 # y
    #u_corr[2] = -0.000897803340365617 # z

    #check1 = ambit_fe.resultcheck.results_check_node(problem.mp.u, check_node, u_corr, problem.mp.V_u, problem.mp.comm, tol=tol, nm='u')
    #success = ambit_fe.resultcheck.success_check([check1], problem.mp.comm)

    #if not success:
        #raise RuntimeError("Test failed!")



if __name__ == "__main__":

    test_main()
