#!/usr/bin/env python3

"""
- solid mechanics
- ramping of Dirichlet BCs using three different methods (time curve, expression, file with time curve)
- One-Step-Theta time integration
- compressible NeoHookean material
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
                         'mesh_domain'           : basepath+'/input/blocks3_domain.xdmf',
                         'mesh_boundary'         : basepath+'/input/blocks3_boundary.xdmf',
                         'write_results_every'   : -1,
                         'write_restart_every'   : -1,
                         'restart_step'          : restart_step,
                         'restart_io_type'       : 'petscvector', # petscvector, rawtxt
                         'output_path'           : basepath+'/tmp/',
                         'results_to_write'      : ['displacement'],
                         'simname'               : 'solid_ost_dbc_ramp'}

    SOLVER_PARAMS     = {'solve_type'            : 'direct',
                         'tol_res'               : 1.0e-8,
                         'tol_inc'               : 1.0e-8}

    TIME_PARAMS       = {'maxtime'               : 1.0,
                         'numstep'               : 10,
                         'timint'                : 'ost',
                         'eval_nonlin_terms'     : 'midpoint',
                         'theta_ost'             : 0.5}

    FEM_PARAMS        = {'order_disp'            : 1,
                         'quad_degree'           : 2,
                         'incompressible_2field' : False}

    MATERIALS         = {'MAT1' : {'neohooke_compressible' : {'mu' : 100., 'nu' : 0.3},
                                   'inertia'           : {'rho0' : 1.0e-4}},
                         'MAT2' : {'neohooke_compressible' : {'mu' : 100., 'nu' : 0.3},
                                   'inertia'           : {'rho0' : 1.0e-4}},
                         'MAT3' : {'neohooke_compressible' : {'mu' : 100., 'nu' : 0.3},
                                   'inertia'           : {'rho0' : 1.0e-4}}}


    # define your load curves here (syntax: tcX refers to curve X, to be used in BC_DICT key 'curve' : [X,0,0], or 'curve' : X)
    class time_curves:

        def tc1(self, t):
            return 0.5*t

    # expression which can be time- and space-dependent
    class expression1:
        def __init__(self):
            self.t = 0.0

        def evaluate(self, x):
            val_t = 0.5*self.t
            return ( np.full(x.shape[1], val_t),
                     np.full(x.shape[1], val_t),
                     np.full(x.shape[1], val_t))


    # three different options to apply the time-varying DBC: all yield same result
    BC_DICT           = { 'dirichlet' : [{'id' : [1,7,13], 'dir' : 'all', 'val' : 0.},
                                         {'id' : [4], 'dir' : 'all', 'curve' : [1,1,1]}, # ramp by time curve
                                         {'id' : [10], 'dir' : 'all', 'expression' : expression1}, # set by expression
                                         {'id' : [16], 'dir' : 'all', 'file' : basepath+'/input/solid_blocks3_dbc.txt', 'ramp_curve' : 1}] } # set by file and ramp by curve


    # problem setup
    problem = ambit_fe.ambit_main.Ambit(IO_PARAMS, TIME_PARAMS, SOLVER_PARAMS, FEM_PARAMS, MATERIALS, BC_DICT, time_curves=time_curves())

    # solve time-dependent problem
    problem.solve_problem()


    # --- results check
    tol = 1.0e-6

    check_node = []
    check_node.append(np.array([1.0, 1.0, 1.0]))
    check_node.append(np.array([1.0, 1.0, 2.25]))
    check_node.append(np.array([1.0, 1.0, 3.5]))
    
    u_corr = np.zeros(3*len(check_node))

    ## correct results
    u_corr[0] = 0.5 # x
    u_corr[1] = 0.5 # y
    u_corr[2] = 0.5 # z
    u_corr[3] = 0.5 # x
    u_corr[4] = 0.5 # y
    u_corr[5] = 0.5 # z
    u_corr[6] = 0.5 # x
    u_corr[7] = 0.5 # y
    u_corr[8] = 0.5 # z

    check1 = ambit_fe.resultcheck.results_check_node(problem.mp.u, check_node, u_corr, problem.mp.V_u, problem.mp.comm, tol=tol, nm='u')
    success = ambit_fe.resultcheck.success_check([check1], problem.mp.comm)

    if not success:
        raise RuntimeError("Test failed!")



if __name__ == "__main__":

    test_main()
