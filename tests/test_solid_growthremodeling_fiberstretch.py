#!/usr/bin/env python3

"""
solid mechanics with growth
block that grows in fiber direction triggered by fiber stretch and remodels to softer material
TODO: Somehow, this does not converge quadratically at the end (seems irrespective of remodeling,
but likely to be attributed to the growth in fiber direction) ---> check linearization terms!

only one hex element in this testcase - cannot be run on multiple cores!
"""

import ambit_fe

import sys
import numpy as np
from pathlib import Path
import pytest


@pytest.mark.solid
def test_main():

    basepath = str(Path(__file__).parent.absolute())

    IO_PARAMS            = {'problem_type'          : 'solid',
                            'mesh_domain'           : basepath+'/input/blockhex_domain.xdmf',
                            'mesh_boundary'         : basepath+'/input/blockhex_boundary.xdmf',
                            'fiber_data'            : [np.array([1.0,0.0,0.0]),np.array([0.0,1.0,0.0])],
                            'write_results_every'   : -999,
                            'output_path'           : basepath+'/tmp/',
                            'results_to_write'      : ['displacement','theta','fiberstretch','fiberstretch_e','phi_remod'],
                            'simname'               : 'solid_growthremodeling_fiberstretch'}

    SOLVER_PARAMS_SOLID  = {'solve_type'            : 'direct',
                            'tol_res'               : 1.0e-8,
                            'tol_inc'               : 1.0e-8}

    TIME_PARAMS_SOLID    = {'maxtime'               : 1.0,
                            'numstep'               : 20,
                            'timint'                : 'static'}

    FEM_PARAMS           = {'order_disp'            : 1,
                            'order_pres'            : 1,
                            'quad_degree'           : 3,
                            'incompressible_2field' : False}

    MATERIALS            = {'MAT1' : {'neohooke_dev'     : {'mu' : 10.},
                                      'ogden_vol'        : {'kappa' : 10./(1.-2.*0.49)},
                                      'growth'           : {'growth_dir' : 'isotropic', # isotropic, fiber, crossfiber, radial
                                                            'growth_trig' : 'fibstretch', # fibstretch, volstress, prescribed
                                                            'growth_thres' : 1.15,
                                                            'thetamax' : 3.0,
                                                            'thetamin' : 1.0,
                                                            'tau_gr' : 1.0,
                                                            'gamma_gr' : 1.72,
                                                            'tau_gr_rev' : 10000.0,
                                                            'gamma_gr_rev' : 1.0,
                                                            'remodeling_mat' : {'neohooke_dev' : {'mu' : 3.},
                                                                                'ogden_vol'    : {'kappa' : 3./(1.-2.*0.49)}}}}}

    # define your load curves here (syntax: tcX refers to curve X, to be used in BC_DICT key 'curve' : [X,0,0], or 'curve' : X)
    class time_curves():

        def tc1(self, t):
            pmax = 10.0
            return pmax*t/TIME_PARAMS_SOLID['maxtime']



    BC_DICT           = { 'dirichlet' : [{'id' : [1], 'dir' : 'x', 'val' : 0.},
                                         {'id' : [2], 'dir' : 'y', 'val' : 0.},
                                         {'id' : [3], 'dir' : 'z', 'val' : 0.}],
                            'neumann' : [{'id' : [4], 'dir' : 'xyz_ref', 'curve' : [1,0,0]}] }


    # problem setup
    problem = ambit_fe.ambit_main.Ambit(IO_PARAMS, TIME_PARAMS_SOLID, SOLVER_PARAMS_SOLID, FEM_PARAMS, MATERIALS, BC_DICT, time_curves=time_curves())

    # solve time-dependent problem
    problem.solve_problem()


    # --- results check
    tol = 1.0e-6

    check_node = []
    check_node.append(np.array([1.0, 1.0, 1.0]))

    u_corr = np.zeros(3*len(check_node))

    ## correct results
    u_corr[0] = 1.0812823521095760E+00 # x
    u_corr[1] = -1.4360291810029382E-01 # y
    u_corr[2] = -1.4360291810029457E-01 # z

    check1 = ambit_fe.resultcheck.results_check_node(problem.mp.u, check_node, u_corr, problem.mp.V_u, problem.mp.comm, tol=tol, nm='u')
    success = ambit_fe.resultcheck.success_check([check1], problem.mp.comm)

    if not success:
        raise RuntimeError("Test failed!")



if __name__ == "__main__":

    test_main()
