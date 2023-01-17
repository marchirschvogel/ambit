#!/usr/bin/env python3

import ambit

import sys, traceback
import numpy as np
from pathlib import Path

import resultcheck


def main():
    
    basepath = str(Path(__file__).parent.absolute())

    IO_PARAMS         = {'problem_type'          : 'flow0d', # solid, fluid, flow0d, solid_flow0d, fluid_flow0d
                         'write_results_every'   : -999,
                         'output_path'           : basepath+'/tmp/',
                         'simname'               : 'test'}

    SOLVER_PARAMS     = {'solve_type'            : 'direct', # direct
                         'tol_res'               : 1.0e-8,
                         'tol_inc'               : 1.0e-8}

    TIME_PARAMS       = {'maxtime'               : 1.0,
                         'numstep'               : 100,
                         'numstep_stop'          : 100,
                         'timint'                : 'ost', # ost
                         'theta_ost'             : 0.5,
                         'initial_conditions'    : init()}
    
    MODEL_PARAMS      = {'modeltype'             : '4elwindkesselLsZ',
                         'parameters'            : param(),
                         'excitation_curve'      : 1}
    

    # define your time curves here (syntax: tcX refers to curve X)
    class time_curves():
        
        def tc1(self, t):
            return 0.5*(1.-np.cos(2.*np.pi*(t)/0.1)) + 1.0


    # problem setup
    problem = ambit.Ambit(IO_PARAMS, TIME_PARAMS, SOLVER_PARAMS, constitutive_params=MODEL_PARAMS, time_curves=time_curves())
    
    # solve time-dependent problem
    problem.solve_problem()


    # --- results check
    tol = 1.0e-7

    s_corr = np.zeros(problem.mp.cardvasc0D.numdof)

    # correct results
    s_corr[0] = 1.0608252198133588E+00
    s_corr[1] = 0.0
    s_corr[2] = 0.0
    
    check1 = resultcheck.results_check_vec(problem.mp.s, s_corr, problem.mp.comm, tol=tol)
    success = resultcheck.success_check([check1], problem.mp.comm)

    return success



def init():
    
    return {'p_0' : 10.0,
            'q_0' : 0.0,
            's_0' : 0.0}


def param():
    
    return {'R' : 100e-6,
            'C' : 2000.0,
            'Z' : 5.0e-6,
            'L' : 6.0e-7,
            'p_ref' : 1.0}




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
