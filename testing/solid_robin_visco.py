#!/usr/bin/env python3

import ambit

import sys, traceback
import numpy as np
from pathlib import Path

import resultcheck

# purely viscous block testing basic rate-dependent solid material

def main():
    
    basepath = str(Path(__file__).parent.absolute())

    IO_PARAMS         = {'problem_type'          : 'solid',
                         'mesh_domain'           : [basepath+'/input/blockhex_domain.xdmf'],
                         'mesh_boundary'         : [basepath+'/input/blockhex_boundary.xdmf'],
                         'write_results_every'   : -999,
                         'output_path'           : '/home/shared/work/codes/fem_scripts/tests/tmp/',
                         'results_to_write'      : ['displacement'],
                         'simname'               : 'solid_robin_visc'}

    SOLVER_PARAMS     = {'solve_type'            : 'direct', # direct, iterative
                         'tol_res'               : 1.0e-8,
                         'tol_inc'               : 1.0e-5,
                         'maxiter'               : 25,
                         'divergence_continue'   : None}

    TIME_PARAMS       = {'maxtime'               : 1.0,
                         'numstep'               : 10,
                         'numstep_stop'          : 5,
                         'timint'                : 'genalpha', # genalpha, ost, static
                         'rho_inf_genalpha'      : 1.0}
    
    FEM_PARAMS        = {'order_disp'            : 1,
                         'quad_degree'           : 2,
                         'incompressible_2field' : False} # True, False

    MATERIALS         = {'MAT1' : {'visco'       : {'eta' : 2.0},
                                   'inertia'     : {'rho0' : 1.0e-6}}}


    # define your load curves here (syntax: tcX refers to curve X, to be used in BC_DICT key 'curve' : [X,0,0], or 'curve' : X)
    class time_curves():
        
        def tc1(self, t):
            return 3.*t

    BC_DICT           = { 'neumann' : [{'type' : 'pk1', 'id' : [3], 'dir' : 'xyz', 'curve' : [0,0,1]}],
                            'robin' : [{'type' : 'spring', 'id' : [1,6], 'dir' : 'normal', 'stiff' : 5.0}] }


    # problem setup
    problem = ambit.Ambit(IO_PARAMS, TIME_PARAMS, SOLVER_PARAMS, FEM_PARAMS, MATERIALS, BC_DICT, time_curves=time_curves())
    
    # solve time-dependent problem
    problem.solve_problem()

    
    # --- results check
    tol = 1.0e-6

    check_node = []
    check_node.append(np.array([0.0, 0.0, 0.0]))

    u_corr = np.zeros(3*len(check_node))
    
    ## correct results
    u_corr[0] = -9.3310481940784017E-17 # x
    u_corr[1] = -3.8042412652908981E-11 # y
    u_corr[2] = 6.4458861009353818E-01 # z

    check1 = resultcheck.results_check_node(problem.mp.u, check_node, u_corr, problem.mp.V_u, problem.mp.comm, tol=tol, nm='u')
    success = resultcheck.success_check([check1], problem.mp.comm)
    
    return success



if __name__ == "__main__":
    
    success = False
    
    try:
        success = main()
    except:
        print(traceback.format_exc())
    
    if success:
        sys.exit(0)
    else:
        sys.exit(1)
