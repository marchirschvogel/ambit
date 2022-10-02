#!/usr/bin/env python3

import ambit

import sys, traceback
import numpy as np
from pathlib import Path

import resultcheck


def main():
    
    basepath = str(Path(__file__).parent.absolute())

    IO_PARAMS            = {'problem_type'          : 'solid_flow0d', # solid, fluid, flow0d, solid_flow0d, fluid_flow0d
                            'mesh_domain'           : basepath+'/input/chamber_domain.xdmf',
                            'mesh_boundary'         : basepath+'/input/chamber_boundary.xdmf',
                            'write_results_every'   : -999,
                            'output_path'           : basepath+'/tmp/',
                            'results_to_write'      : [''],
                            'simname'               : 'test'}

    SOLVER_PARAMS_SOLID  = {'solve_type'            : 'direct', # direct, iterative
                            'tol_res'               : 1.0e-8,
                            'tol_inc'               : 1.0e-8}
    
    SOLVER_PARAMS_FLOW0D = {'tol_res'               : 1.0e-8,
                            'tol_inc'               : 1.0e-8}

    TIME_PARAMS_SOLID    = {'maxtime'               : 1.0,
                            'numstep'               : 20,
                            'numstep_stop'          : 10,
                            'timint'                : 'genalpha', # genalpha, ost, static
                            'theta_ost'             : 1.0,
                            'rho_inf_genalpha'      : 0.8}
    
    TIME_PARAMS_FLOW0D   = {'timint'                : 'ost', # ost
                            'theta_ost'             : 0.5,
                            'initial_conditions'    : {'p_0' : 0.0, 'q_0' : 0.0, 's_0' : 0.0}}

    MODEL_PARAMS_FLOW0D  = {'modeltype'             : '4elwindkesselLsZ',
                            'parameters'            : {'R' : 1.0e3, 'C' : 0.0, 'Z' : 10.0, 'L' : 5.0, 'p_ref' : 0.0}}

    FEM_PARAMS           = {'order_disp'            : 1,
                            'order_pres'            : 1,
                            'quad_degree'           : 1,
                            'incompressible_2field' : False} # True, False
    
    COUPLING_PARAMS      = {'surface_ids'           : [[3]],
                            'coupling_quantity'     : ['volume'],
                            'coupling_type'         : 'monolithic_direct'}

    MATERIALS            = {'MAT1' : {'neohooke_dev' : {'mu' : 100.}, 'ogden_vol' : {'kappa' : 100./(1.-2.*0.49)}, 'inertia' : {'rho0' : 1.0e-6}}}

    # define your load curves here (syntax: tcX refers to curve X, to be used in BC_DICT key 'curve' : [X,0,0], or 'curve' : X)
    class time_curves():
        
        def tc1(self, t):
            pmax = -10.
            return pmax*t/TIME_PARAMS_SOLID['maxtime']


    BC_DICT           = { 'dirichlet' : [{'id' : [1], 'dir' : 'x', 'val' : 0.},
                                         {'id' : [3], 'dir' : 'y', 'val' : 0.},
                                         {'id' : [3], 'dir' : 'z', 'val' : 0.}],
                            'neumann' : [{'type' : 'true', 'id' : [2], 'dir' : 'normal', 'curve' : 1}]}


    # problem setup
    problem = ambit.Ambit(IO_PARAMS, [TIME_PARAMS_SOLID, TIME_PARAMS_FLOW0D], [SOLVER_PARAMS_SOLID, SOLVER_PARAMS_FLOW0D], FEM_PARAMS, [MATERIALS, MODEL_PARAMS_FLOW0D], BC_DICT, time_curves=time_curves(), coupling_params=COUPLING_PARAMS)
    
    # solve time-dependent problem
    problem.solve_problem()


    # --- results check
    tol = 1.0e-7
        
    s_corr = np.zeros(problem.mp.pbf.cardvasc0D.numdof)

    # correct 0D results
    s_corr[0] = 9.2733644380642666E+00
    s_corr[1] = -9.1004836216937203E-03
    s_corr[2] = -1.6375196030721982E-02
    
    check1 = resultcheck.results_check_vec(problem.mp.pbf.s, s_corr, problem.mp.comm, tol=tol)
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
