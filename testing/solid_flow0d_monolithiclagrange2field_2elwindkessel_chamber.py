#!/usr/bin/env python3

import ambit

import sys, time, traceback
import numpy as np
from pathlib import Path

import results_check


def main():
    
    basepath = str(Path(__file__).parent.absolute())

    IO_PARAMS            = {'problem_type'          : 'solid_flow0d', # solid, fluid, flow0d, solid_flow0d, fluid_flow0d
                            'mesh_domain'           : ''+basepath+'/input/chamber_domain.xdmf',
                            'mesh_boundary'         : ''+basepath+'/input/chamber_boundary.xdmf',
                            'write_results_every'   : -999,
                            'output_path'           : ''+basepath+'/tmp/',
                            'results_to_write'      : [''],
                            'simname'               : 'test'}

    SOLVER_PARAMS_SOLID  = {'solve_type'            : 'direct', # direct, iterative
                            'tol_res'               : 1.0e-8,
                            'tol_inc'               : 1.0e-8}
    
    SOLVER_PARAMS_FLOW0D = {'tol_res'               : 1.0e-8,
                            'tol_inc'               : 1.0e-8}

    TIME_PARAMS_SOLID    = {'maxtime'               : 1.0,
                            'numstep'               : 100,
                            'numstep_stop'          : 10,
                            'timint'                : 'ost', # genalpha, ost, static
                            'theta_ost'             : 1.0,
                            'rho_inf_genalpha'      : 0.8}
    
    TIME_PARAMS_FLOW0D   = {'timint'                : 'ost', # ost
                            'theta_ost'             : 1.0,
                            'initial_conditions'    : {'Q_0' : 0.0}}

    MODEL_PARAMS_FLOW0D  = {'modeltype'             : '2elwindkessel',
                            'parameters'            : {'C' : 10.0, 'R' : 100.0, 'p_ref' : 0.0}}

    FEM_PARAMS           = {'order_disp'            : 1,
                            'order_pres'            : 1,
                            'quad_degree'           : 1,
                            'incompressible_2field' : True} # True, False
    
    COUPLING_PARAMS      = {'surface_ids'           : [[3]],
                            'coupling_quantity'     : 'pressure',
                            'coupling_type'         : 'monolithic_lagrange'}

    MATERIALS            = {'MAT1' : {'neohooke_dev' : {'mu' : 100.}, 'inertia' : {'rho0' : 1.0e-6}}}

    # define your load curves here (syntax: tcX refers to curve X, to be used in BC_DICT key 'curve' : [X,0,0], or 'curve' : X)
    class time_curves():
        
        def tc1(self, t):
            tmax = -50.
            return tmax*np.sin(2.*np.pi*t/TIME_PARAMS_SOLID['maxtime'])


    BC_DICT           = { 'dirichlet' : [{'id' : [1], 'dir' : 'x', 'val' : 0.},
                                         {'id' : [3], 'dir' : 'y', 'val' : 0.},
                                         {'id' : [3], 'dir' : 'z', 'val' : 0.}],
                            'neumann' : [{'type' : 'pk1', 'id' : [2], 'dir' : 'xyz', 'curve' : [1,0,0]}]}


    # problem setup
    problem = ambit.Ambit(IO_PARAMS, [TIME_PARAMS_SOLID, TIME_PARAMS_FLOW0D], [SOLVER_PARAMS_SOLID, SOLVER_PARAMS_FLOW0D], FEM_PARAMS, [MATERIALS, MODEL_PARAMS_FLOW0D], BC_DICT, time_curves=time_curves(), coupling_params=COUPLING_PARAMS)
    
    # solve time-dependent problem
    problem.solve_problem()


    # --- results check
    tol = 1.0e-7
        
    s_corr = np.zeros(problem.mp.pbf.cardvasc0D.numdof)

    # correct 0D results
    s_corr[0] = 9.0742660503706130E-01
    
    check1 = results_check.results_check_vec(problem.mp.pbf.s, s_corr, problem.mp.comm, tol=tol)
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
