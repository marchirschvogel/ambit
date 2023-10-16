#!/usr/bin/env python3

"""
solid mechanics: prescribed growth of an idealized LV geometry to theta = 2
"""

import ambit_fe

import sys
import numpy as np
from pathlib import Path
import pytest


@pytest.mark.solid
def test_main():

    basepath = str(Path(__file__).parent.absolute())


    IO_PARAMS            = {'problem_type'           : 'solid',
                            'mesh_domain'            : basepath+'/input/lv_domain.xdmf',
                            'mesh_boundary'          : basepath+'/input/lv_boundary.xdmf',
                            'write_results_every'    : -999,
                            'output_path'            : basepath+'/tmp/',
                            'results_to_write'       : ['displacement','theta'],
                            'simname'                : 'solid_growth_prescribed_iso_lv'}

    FEM_PARAMS           = {'order_disp'             : 1,
                            'order_pres'             : 1,
                            'quad_degree'            : 1,
                            'incompressible_2field'  : True}

    SOLVER_PARAMS_SOLID  = {'solve_type'             : 'direct',
                            'tol_res'                : 1.0e-8,
                            'tol_inc'                : 1.0e-8}


    TIME_PARAMS_SOLID    = {'maxtime'                : 1.0,
                            'numstep'                : 10,
                            'timint'                 : 'static'}

    MATERIALS            = {'MAT1' : {'neohooke_dev' : {'mu' : 100.},
                                      'growth'       : {'growth_dir' : 'isotropic',
                                                        'growth_trig' : 'prescribed',
                                                        'prescribed_curve' : 1}}}


    # define your load curves here (syntax: tcX refers to curve X, to be used in BC_DICT key 'curve' : [X,0,0], or 'curve' : X)
    class time_curves():

        def tc1(self, t):
            thetamax = 2.0
            gr = thetamax-1.
            return 1.0 + gr*t/TIME_PARAMS_SOLID['maxtime']


    BC_DICT              = { 'dirichlet' : [{'id' : [2], 'dir' : 'all', 'val' : 0.}]}

    # problem setup
    problem = ambit_fe.ambit_main.Ambit(IO_PARAMS, TIME_PARAMS_SOLID, SOLVER_PARAMS_SOLID, FEM_PARAMS, MATERIALS, BC_DICT, time_curves=time_curves())

    # solve time-dependent problem
    problem.solve_problem()


    # --- results check
    tol = 1.0e-6

    check_node = []
    check_node.append(np.array([3.475149154663086, -3.17646312713623, -74.3183364868164]))

    u_corr, p_corr = np.zeros(3*len(check_node)), np.zeros(len(check_node))

    ## correct results (apex node)
    u_corr[0] = -9.3743445617845182E+00 # x
    u_corr[1] = 1.0877463102123736E+01 # y
    u_corr[2] = -1.0954897338860498E+02 # z

    p_corr[0] = -9.3006817518134621E+00

    check1 = ambit_fe.resultcheck.results_check_node(problem.mp.u, check_node, u_corr, problem.mp.V_u, problem.mp.comm, tol=tol, nm='u')
    check2 = ambit_fe.resultcheck.results_check_node(problem.mp.p, check_node, p_corr, problem.mp.V_p, problem.mp.comm, tol=tol, nm='p')

    success = ambit_fe.resultcheck.success_check([check1,check2], problem.mp.comm)

    if not success:
        raise RuntimeError("Test failed!")



if __name__ == "__main__":

    test_main()
