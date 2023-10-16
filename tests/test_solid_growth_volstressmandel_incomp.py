#!/usr/bin/env python3

"""
solid mechanics: stress-mediated volumetric growth of an incompressible block (p2p1 interpolation)
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
                         'mesh_domain'           : basepath+'/input/block_domain.xdmf',
                         'mesh_boundary'         : basepath+'/input/block_boundary.xdmf',
                         'write_results_every'   : -999,
                         'output_path'           : basepath+'/tmp/',
                         'results_to_write'      : ['displacement','pressure','theta','cauchystress'],
                         'simname'               : 'test_solid_growth_volstressmandel_incomp'}

    SOLVER_PARAMS     = {'solve_type'            : 'direct',
                         'tol_res'               : 1.0e-8,
                         'tol_inc'               : 1.0e-8}

    TIME_PARAMS       = {'maxtime'               : 1.0,
                         'numstep'               : 5,
                         'timint'                : 'static'}

    FEM_PARAMS        = {'order_disp'            : 2,
                         'order_pres'            : 1,
                         'quad_degree'           : 5,
                         'incompressible_2field' : True}

    MATERIALS         = {'MAT1' : {'neohooke_dev' : {'mu' : 10.},
                                   'growth'       : {'growth_dir' : 'isotropic', # isotropic, fiber, crossfiber, radial
                                                     'growth_trig' : 'volstress', # fibstretch, volstress, prescribed
                                                     'growth_thres' : -25.0,
                                                     'thetamax' : 3.0,
                                                     'thetamin' : 1.0,
                                                     'tau_gr' : 20.0,
                                                     'gamma_gr' : 2.0,
                                                     'tau_gr_rev' : 10000.0,
                                                     'gamma_gr_rev' : 2.0}}}


    # define your load curves here (syntax: tcX refers to curve X, to be used in BC_DICT key 'curve' : [X,0,0], or 'curve' : X)
    class time_curves():

        def tc1(self, t):
            pmax = -10.
            return pmax*t/TIME_PARAMS['maxtime']


    BC_DICT           = { 'dirichlet' : [{'id' : [1], 'dir' : 'x', 'val' : 0.},
                                         {'id' : [2], 'dir' : 'y', 'val' : 0.},
                                         {'id' : [3], 'dir' : 'z', 'val' : 0.}],
                            # hydrostatic Neumann on all faces
                            'neumann' : [{'id' : [1,2,3,4,5,6], 'dir' : 'normal_cur', 'curve' : 1}] }


    # problem setup
    problem = ambit_fe.ambit_main.Ambit(IO_PARAMS, TIME_PARAMS, SOLVER_PARAMS, FEM_PARAMS, MATERIALS, BC_DICT, time_curves=time_curves())

    # solve time-dependent problem
    problem.solve_problem()


    # --- results check
    tol = 1.0e-6

    check_node = []
    check_node.append(np.array([1.00000000e+00, 1.00000000e+00, 1.00000000e+00]))

    u_corr, p_corr = np.zeros(3*len(check_node)), np.zeros(len(check_node))

    # correct results
    u_corr[0] = 1.8222579446973952E-01 # x
    u_corr[1] = 1.8222579446973952E-01 # y
    u_corr[2] = 1.8222579446973952E-01 # z

    p_corr[0] = 1.6523471374159470E+01

    check1 = ambit_fe.resultcheck.results_check_node(problem.mp.u, check_node, u_corr, problem.mp.V_u, problem.mp.comm, tol=tol, nm='u')
    check2 = ambit_fe.resultcheck.results_check_node(problem.mp.p, check_node, p_corr, problem.mp.V_p, problem.mp.comm, tol=tol, nm='p')

    success = ambit_fe.resultcheck.success_check([check1,check2], problem.mp.comm)

    if not success:
        raise RuntimeError("Test failed!")



if __name__ == "__main__":

    test_main()
