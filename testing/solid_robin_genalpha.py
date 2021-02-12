#!/usr/bin/env python3

import ambit

import sys, traceback
import numpy as np
from pathlib import Path

import results_check


def main():
    
    basepath = str(Path(__file__).parent.absolute())

    IO_PARAMS         = {'problem_type'          : 'solid', # solid, fluid, flow0d, solid_flow0d, fluid_flow0d
                         'mesh_domain'           : ''+basepath+'/input/block2_domain.xdmf',
                         'mesh_boundary'         : ''+basepath+'/input/block2_boundary.xdmf',
                         'write_results_every'   : -999,
                         'output_path'           : ''+basepath+'/tmp/',
                         'results_to_write'      : [''],
                         'simname'               : 'solid_robin_genalpha'}

    SOLVER_PARAMS     = {'solve_type'            : 'direct', # direct, iterative
                         'tol_res'               : 1.0e-8,
                         'tol_inc'               : 1.0e-8,
                         'maxiter'               : 25,
                         'divergence_continue'   : None}

    TIME_PARAMS       = {'maxtime'               : 1.0,
                         'numstep'               : 10,
                         'numstep_stop'          : 10,
                         'timint'                : 'genalpha', # genalpha, ost, static
                         'rho_inf_genalpha'      : 1.0,
                         'avg_genalpga'          : 'trlike'} # trlike, midlike
    
    FEM_PARAMS        = {'order_disp'            : 1,
                         'quad_degree'           : 1,
                         'incompressible_2field' : False} # True, False

    MATERIALS         = {'MAT1' : {'stvenantkirchhoff' : {'Emod' : 1000., 'nu' : 0.3},
                                   'inertia'           : {'rho0' : 1.0e-6}}}


    # define your load curves here (syntax: tcX refers to curve X, to be used in BC_DICT key 'curve' : [X,0,0], or 'curve' : X)
    class time_curves():
        
        def tc1(self, t):
            return 3.*t

    BC_DICT           = { 'neumann' : [{'type' : 'pk1', 'id' : 3, 'dir' : 'xyz', 'curve' : [1,0,0]}],
                            'robin' : [{'type' : 'spring', 'id' : 1, 'dir' : 'normal', 'stiff' : 5.0},
                                       {'type' : 'spring', 'id' : 2, 'dir' : 'normal', 'stiff' : 5.0}] }


    # problem setup
    problem = ambit.Ambit(IO_PARAMS, TIME_PARAMS, SOLVER_PARAMS, FEM_PARAMS, MATERIALS, BC_DICT, time_curves=time_curves())
    
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

    check1 = results_check.results_check_node(problem.mp.u, check_node, u_corr, problem.mp.V_u, problem.mp.comm, tol=tol, nm='u')
    success = results_check.success_check([check1], problem.mp.comm)
    
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
