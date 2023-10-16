#!/usr/bin/env python3

"""
0D signalling network model for cardiac hypertrophy: not yet fully tested!
"""

import ambit_fe

import sys
import numpy as np
from pathlib import Path
import pytest


@pytest.mark.signet
@pytest.mark.skip(reason="Not yet ready for testing.")
def main():

    basepath = str(Path(__file__).parent.absolute())

    IO_PARAMS         = {'problem_type'          : 'signet',
                         'write_results_every'   : 1,
                         'output_path'           : basepath+'/tmp/',
                         'simname'               : 'test'}

    SOLVER_PARAMS     = {'solve_type'            : 'direct',
                         'tol_res'               : 1.0e-8,
                         'tol_inc'               : 1.0e-8}

    TIME_PARAMS       = {'maxtime'               : 1.0,
                         'numstep'               : 100,
                         'numstep_stop'          : 100,
                         'timint'                : 'ost',
                         'theta_ost'             : 0.5,
                         'initial_conditions'    : init()}

    MODEL_PARAMS      = {'modeltype'             : 'hypertrophy',
                         'parameters'            : param(),
                         'excitation_curve'      : 1}


    # define your time curves here (syntax: tcX refers to curve X)
    class time_curves():

        def tc1(self, t):
            return 0.5*(1.-np.cos(2.*np.pi*(t)/0.1)) + 1.0

    # problem setup
    problem = ambit_fe.ambit_main.Ambit(IO_PARAMS, TIME_PARAMS, SOLVER_PARAMS, constitutive_params=MODEL_PARAMS, time_curves=time_curves())

    # solve time-dependent problem
    problem.solve_problem()


    ## --- results check
    #tol = 1.0e-7

    #s_corr = np.zeros(problem.mp.cardvasc0D.numdof)

    ## correct results
    #s_corr[0] = 1.1484429140599208E+00
    #s_corr[1] = -7.3138173135468898E-01
    #s_corr[2] = 0.0
    #s_corr[3] = 0.0

    #check1 = ambit_fe.resultcheck.results_check_vec(problem.mp.s, s_corr, problem.mp.comm, tol=tol)
    #success = ambit_fe.resultcheck.success_check([check1], problem.mp.comm)

    #if not success:
        #raise RuntimeError("Test failed!")



def init():

    return {'var1_0' : 0.0}


def param():

    return {'p1' : 1.}



if __name__ == "__main__":

    test_main()
