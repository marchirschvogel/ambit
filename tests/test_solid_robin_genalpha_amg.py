#!/usr/bin/env python3

"""
- solid mechanics
- elastic St. Venant-Kirchhoff material
- Generalized-alpha time integration
- Robin conditions
- own read-/write function for restarts (only working for nodal fields!)
- iterative solution using Hypre AMG
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
                         'write_results_every'   : -999,
                         'write_restart_every'   : 8,
                         'restart_step'          : restart_step,
                         'restart_io_type'       : 'rawtxt', # petscvector, rawtxt
                         'output_path'           : basepath+'/tmp/',
                         'results_to_write'      : [''],
                         'simname'               : 'solid_robin_genalpha'}

    CONTROL_PARAMS    = {'maxtime'               : 1.0,
                         'numstep'               : 10}

    SOLVER_PARAMS     = {'solve_type'            : 'iterative',
                         'petsc_options_ksp'     : {'ksp_gmres_modifiedgramschmidt':True, 'ksp_gmres_restart':100},
                         'precond_fields'        : [{'prec':'amg', 'petsc_options':{'pc_hypre_boomeramg_strong_threshold':0.7}}],
                         'tol_lin_rel'           : 1.0e-5,
                         'tol_lin_abs'           : 1.0e-9,
                         'print_liniter_every'   : 10,
                         'tol_res'               : 1.0e-8,
                         'tol_inc'               : 1.0e-8,
                         'maxiter'               : 25,
                         'divergence_continue'   : None}

    TIME_PARAMS       = {'timint'                : 'genalpha',
                         'eval_nonlin_terms'     : 'midpoint',
                         'rho_inf_genalpha'      : 1.0}

    FEM_PARAMS        = {'order_disp'            : 1,
                         'quad_degree'           : 2,
                         'incompressibility'     : 'no'}

    MATERIALS         = {'MAT1' : {'stvenantkirchhoff' : {'Emod' : 1000., 'nu' : 0.3},
                                   'inertia'           : {'rho0' : 1.0e-6}}}


    class expression1:
        def __init__(self):
            self.t = 0.0

        def evaluate(self, x):
            val = 3.*self.t
            return ( np.full(x.shape[1], val),
                     np.full(x.shape[1], 0.0),
                     np.full(x.shape[1], 0.0) )

    BC_DICT           = { 'neumann' : [{'id' : [3], 'dir' : 'xyz_ref', 'expression' : expression1}],
                            'robin' : [{'type' : 'spring', 'id' : [1,2], 'dir' : 'normal_ref', 'stiff' : 5.0}] }


    # problem setup
    problem = ambit_fe.ambit_main.Ambit(IO_PARAMS, CONTROL_PARAMS, TIME_PARAMS, SOLVER_PARAMS, FEM_PARAMS, MATERIALS, BC_DICT)

    # solve time-dependent problem
    problem.solve_problem()


    # --- results check
    tol = 1.0e-6

    check_node = []
    check_node.append(np.array([-1.0000000000000000e+00, -1.0000000000000000e+00, 1.0000000000000000e+01]))

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
