#!/usr/bin/env python3

"""
solid 3D-0D coupling: hollow solid chambers linked via CRLinoutlink 0D model (two CRL models in series that link in- to outflux 3D model surface)
one active block contracts and fills the other
"""

import ambit_fe

import sys
import numpy as np
from pathlib import Path
import pytest


@pytest.mark.solid_flow0d
def test_main():

    basepath = str(Path(__file__).parent.absolute())

    IO_PARAMS            = {'problem_type'          : 'solid_flow0d',
                            'mesh_domain'           : basepath+'/input/chambers2hex_domain.xdmf',
                            'mesh_boundary'         : basepath+'/input/chambers2hex_boundary.xdmf',
                            'fiber_data'            : [np.array([1.0,0.0,0.0])],
                            'write_results_every'   : 1,
                            'output_path'           : basepath+'/tmp/',
                            'results_to_write'      : ['displacement'],
                            'simname'               : 'CRLinoutlink'}

    SOLVER_PARAMS        = {'solve_type'            : 'direct',
                            'tol_res'               : 1.0e-8,
                            'tol_inc'               : 1.0e-8,
                            'subsolver_params'      : {'tol_res' : 1.0e-8, 'tol_inc' : 1.0e-8}}

    TIME_PARAMS_SOLID    = {'maxtime'               : 0.3,
                            'numstep'               : 10,
                            #'numstep_stop'          : 10,
                            'timint'                : 'ost',
                            'theta_ost'             : 1.0}

    TIME_PARAMS_FLOW0D   = {'timint'                : 'ost',
                            'theta_ost'             : 1.0,
                            'initial_conditions'    : {'q_in_0' : 0.0, 'q_d_0' : 0.0, 'p_d_0' : 0.0, 'q_out_0' : 0.0}}

    MODEL_PARAMS_FLOW0D  = {'modeltype'             : 'CRLinoutlink',
                            'parameters'            : {'C_in' : 0.001, 'R_in' : 1e-6, 'L_in' : 0., 'C_out' :  0.001, 'R_out' : 1e-6, 'L_out' : 0.}}

    FEM_PARAMS           = {'order_disp'            : 1,
                            'order_pres'            : 1,
                            'quad_degree'           : 5,
                            'incompressible_2field' : False}

    COUPLING_PARAMS      = {'surface_ids'           : [[2],[4]],
                            'coupling_quantity'     : ['pressure']*2,
                            'variable_quantity'     : ['flux']*2,
                            'cq_factor'             : [1.,-1.], # out-flow positive, in-flow negative
                            'coupling_type'         : 'monolithic_lagrange',
                            'print_subiter'         : True}

    MATERIALS            = {'MAT1' : {'neohooke_dev' : {'mu' : 100.}, 'ogden_vol' : {'kappa' : 100./(1.-2.*0.49)}, 'inertia' : {'rho0' : 1.0e-6}, 'active_fiber' : {'sigma0' : 50.0, 'alpha_max' : 15.0, 'alpha_min' : -20.0, 'activation_curve' : 1}},
                            'MAT2' : {'neohooke_dev' : {'mu' : 100.}, 'ogden_vol' : {'kappa' : 100./(1.-2.*0.49)}, 'inertia' : {'rho0' : 1.0e-6}}}

    # define your load curves here (syntax: tcX refers to curve X, to be used in BC_DICT key 'curve' : [X,0,0], or 'curve' : X)
    class time_curves():

        def tc1(self, t):

            K = 5.
            t_contr, t_relax = 0.0, 0.3

            alpha_max = MATERIALS['MAT1']['active_fiber']['alpha_max']
            alpha_min = MATERIALS['MAT1']['active_fiber']['alpha_min']

            c1 = t_contr + alpha_max/(K*(alpha_max-alpha_min))
            c2 = t_relax - alpha_max/(K*(alpha_max-alpha_min))

            # Diss Hirschvogel eq. 2.101
            return (K*(t-c1)+1.)*((K*(t-c1)+1.)>0.) - K*(t-c1)*((K*(t-c1))>0.) - K*(t-c2)*((K*(t-c2))>0.) + (K*(t-c2)-1.)*((K*(t-c2)-1.)>0.)


    BC_DICT           = { 'dirichlet' : [{'id' : [1,3], 'dir' : 'x', 'val' : 0.},
                                         {'id' : [2,4], 'dir' : 'y', 'val' : 0.},
                                         {'id' : [2,4], 'dir' : 'z', 'val' : 0.}]}


    # problem setup
    problem = ambit_fe.ambit_main.Ambit(IO_PARAMS, [TIME_PARAMS_SOLID, TIME_PARAMS_FLOW0D], SOLVER_PARAMS, FEM_PARAMS, [MATERIALS, MODEL_PARAMS_FLOW0D], BC_DICT, time_curves=time_curves(), coupling_params=COUPLING_PARAMS)

    # solve time-dependent problem
    problem.solve_problem()


    # --- results check
    tol = 1.0e-7

    s_corr = np.zeros(problem.mp.pb0.cardvasc0D.numdof)

    # correct 0D results
    s_corr[0] = 5.9984206989156643E+00
    s_corr[1] = 5.9984750503031279E+00
    s_corr[2] = 2.3053217861216080E+01
    s_corr[3] = 5.9984815253587698E+00

    check1 = ambit_fe.resultcheck.results_check_vec_sq(problem.mp.pb0.s, s_corr, problem.mp.comm, tol=tol)
    success = ambit_fe.resultcheck.success_check([check1], problem.mp.comm)

    if not success:
        raise RuntimeError("Test failed!")



if __name__ == "__main__":

    test_main()
