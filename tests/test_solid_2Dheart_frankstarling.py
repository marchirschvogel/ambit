#!/usr/bin/env python3

"""
solid mechanics, 2D biventricular generic heart, testing of:
- Mooney-Rivlin material
- active stress with Frank-Starling law
- Robin BC in normal direction (spring)
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
                            'mesh_domain'           : basepath+'/input/heart2D_domain.xdmf',
                            'mesh_boundary'         : basepath+'/input/heart2D_boundary.xdmf',
                            'fiber_data'            : [basepath+'/input/fib_fiber_nodal_2D.txt'],
                            'order_fib_input'       : 1,
                            'write_results_every'   : -999,
                            'output_path'           : basepath+'/tmp/',
                            'results_to_write'      : ['displacement','pressure','fiberstretch'],
                            'simname'               : 'solid_2Dheart_frankstarling'}

    SOLVER_PARAMS_SOLID  = {'solve_type'            : 'direct',
                            'tol_res'               : 1.0e-8,
                            'tol_inc'               : 1.0e-8,
                            'ptc'                   : False}

    TIME_PARAMS_SOLID    = {'maxtime'               : 1.0,
                            'numstep'               : 10,
                            'numstep_stop'          : 5,
                            'timint'                : 'genalpha',
                            'theta_ost'             : 1.0,
                            'rho_inf_genalpha'      : 0.8}

    FEM_PARAMS           = {'order_disp'            : 2,
                            'order_pres'            : 1,
                            'quad_degree'           : 5,
                            'incompressible_2field' : True}

    MATERIALS            = {'MAT1' : {'mooneyrivlin_dev'  : {'c1' : 60., 'c2' : -20.},
                                      'active_fiber'      : {'sigma0' : 100.0, 'alpha_max' : 15.0, 'alpha_min' : -20.0, 'activation_curve' : 3, 'frankstarling' : True, 'amp_min' : 1., 'amp_max' : 1.7, 'lam_threslo' : 1.01, 'lam_maxlo' : 1.15, 'lam_threshi' : 999., 'lam_maxhi' : 9999.},
                                      'inertia'           : {'rho0' : 1.0e-5},
                                      'visco_green'       : {'eta' : 0.001}}}



    # define your load curves here (syntax: tcX refers to curve X, to be used in BC_DICT key 'curve' : [X,0,0], or 'curve' : X)
    class time_curves():

        def tc1(self, t):
            pmax = -16.
            if t <= 0.2:
                return pmax*t/0.2
            else:
                return pmax

        def tc2(self, t):
            pmax = -4.
            if t <= 0.2:
                return pmax*t/0.2
            else:
                return pmax

        def tc3(self, t):

            K = 5.
            t_contr, t_relax = 0.2, 1000.

            alpha_max = MATERIALS['MAT1']['active_fiber']['alpha_max']
            alpha_min = MATERIALS['MAT1']['active_fiber']['alpha_min']

            c1 = t_contr + alpha_max/(K*(alpha_max-alpha_min))
            c2 = t_relax - alpha_max/(K*(alpha_max-alpha_min))

            # Diss Hirschvogel eq. 2.101
            return (K*(t-c1)+1.)*((K*(t-c1)+1.)>0.) - K*(t-c1)*((K*(t-c1))>0.) - K*(t-c2)*((K*(t-c2))>0.) + (K*(t-c2)-1.)*((K*(t-c2)-1.)>0.)


    BC_DICT              = { 'dirichlet' : [{'dir' : '2dimZ', 'val' : 0.}],
                            'neumann' : [{'id' : [1], 'dir' : 'normal_cur', 'curve' : 1},
                                         {'id' : [2], 'dir' : 'normal_cur', 'curve' : 2}],
                            'robin' : [{'type' : 'spring', 'id' : [3], 'dir' : 'normal_ref', 'stiff' : 0.075}] }

    # problem setup
    problem = ambit_fe.ambit_main.Ambit(IO_PARAMS, TIME_PARAMS_SOLID, SOLVER_PARAMS_SOLID, FEM_PARAMS, MATERIALS, BC_DICT, time_curves=time_curves())

    # solve time-dependent problem
    problem.solve_problem()


    # --- results check
    tol = 1.0e-6

    check_node = []
    check_node.append(np.array([-21.089852094479845, -26.26308841783208, 9.227760327944651e-16]))

    u_corr = np.zeros(3*len(check_node))

    ## correct results
    u_corr[0] = 4.8992618227875901E+00 # x
    u_corr[1] = 2.5321717354565503E+00 # y
    u_corr[2] = 0.0 # z

    check1 = ambit_fe.resultcheck.results_check_node(problem.mp.u, check_node, u_corr, problem.mp.V_u, problem.mp.comm, tol=tol, nm='u')
    success = ambit_fe.resultcheck.success_check([check1], problem.mp.comm)

    if not success:
        raise RuntimeError("Test failed!")



if __name__ == "__main__":

    test_main()
