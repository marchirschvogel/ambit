#!/usr/bin/env python3

"""
2-element Windkessel, 3 decoupled models
"""

import ambit_fe

import sys
import numpy as np
from pathlib import Path
import pytest


@pytest.mark.flow0d
def test_main():

    basepath = str(Path(__file__).parent.absolute())

    IO_PARAMS         = {'problem_type'          : 'flow0d',
                         'write_results_every'   : -999,
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

    MODEL_PARAMS      = {'modeltype'             : '2elwindkessel',
                         'parameters'            : param(),
                         'excitation_curve'      : [1,1,1]}


    # define your time curves here (syntax: tcX refers to curve X)
    class time_curves:

        def tc1(self, t):
            return 0.5*(1.-np.cos(2.*np.pi*(t)/0.1)) + 1.0

    # problem setup
    problem = ambit_fe.ambit_main.Ambit(IO_PARAMS, TIME_PARAMS, SOLVER_PARAMS, constitutive_params=MODEL_PARAMS, time_curves=time_curves())

    # solve time-dependent problem
    problem.solve_problem()


    # --- results check
    tol = 1.0e-7

    s_corr = np.zeros(problem.mp.cardvasc0D.numdof)

    # correct results
    s_corr[0] = 1.0608252198133676E+00
    s_corr[1] = 1.0608252198133676E+00
    s_corr[2] = 1.0608252198133676E+00

    check1 = ambit_fe.resultcheck.results_check_vec_sq(problem.mp.s, s_corr, problem.mp.comm, tol=tol)
    success = ambit_fe.resultcheck.success_check([check1], problem.mp.comm)

    if not success:
        raise RuntimeError("Test failed!")



def init():

    return {'p1_0' : 10.0,
            'p2_0' : 10.0,
            'p3_0' : 10.0}


def param():

    return {'R1' : 100e-6,
            'C1' : 2000.0,
            'p_ref1' : 1.0,
            'R2' : 100e-6,
            'C2' : 2000.0,
            'p_ref2' : 1.0,
            'R3' : 100e-6,
            'C3' : 2000.0,
            'p_ref3' : 1.0,
            'num_models' : 3}




if __name__ == "__main__":

    test_main()
