#!/usr/bin/env python3

"""
solid 3D-0D coupling: incompressible hollow solid chamber coupled to 4-element windkessel model (inertance parallel to impedance, LpZ),
- monolithic coupling via direct monolithic integration of 0D model into system
- direct solve
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
                            'mesh_domain'           : basepath+'/input/chamber_domain.xdmf',
                            'mesh_boundary'         : basepath+'/input/chamber_boundary.xdmf',
                            'write_results_every'   : -999,
                            'output_path'           : basepath+'/tmp/',
                            'results_to_write'      : [''],
                            'simname'               : 'test',
                            'ode_parallel'          : True}

    SOLVER_PARAMS        = {'solve_type'            : 'direct',
                            'tol_res'               : 1.0e-8,
                            'tol_inc'               : 1.0e-8}

    TIME_PARAMS_SOLID    = {'maxtime'               : 1.0,
                            'numstep'               : 20,
                            'numstep_stop'          : 10,
                            'timint'                : 'genalpha',
                            'theta_ost'             : 1.0,
                            'rho_inf_genalpha'      : 0.8}

    TIME_PARAMS_FLOW0D   = {'timint'                : 'ost', # ost
                            'theta_ost'             : 0.5,
                            'initial_conditions'    : {'p_0' : 0.0, 'g_0' : 0.0, 'q_0' : 0.0, 's_0' : 0.0}}

    MODEL_PARAMS_FLOW0D  = {'modeltype'             : '4elwindkesselLpZ',
                            'parameters'            : {'R' : 1.0e3, 'C' : 0.0, 'Z' : 10.0, 'L' : 5.0, 'p_ref' : 0.0}}

    FEM_PARAMS           = {'order_disp'            : 2,
                            'order_pres'            : 1,
                            'quad_degree'           : 5,
                            'incompressible_2field' : True}

    COUPLING_PARAMS      = {'surface_ids'           : [[3]],
                            'coupling_quantity'     : ['volume'],
                            'coupling_type'         : 'monolithic_direct'}

    MATERIALS            = {'MAT1' : {'neohooke_dev' : {'mu' : 100.}, 'inertia' : {'rho0' : 1.0e-6}}}

    # define your load curves here (syntax: tcX refers to curve X, to be used in BC_DICT key 'curve' : [X,0,0], or 'curve' : X)
    class time_curves():

        def tc1(self, t):
            pmax = -10.
            return pmax*t/TIME_PARAMS_SOLID['maxtime']


    BC_DICT           = { 'dirichlet' : [{'id' : [1], 'dir' : 'x', 'val' : 0.},
                                         {'id' : [3], 'dir' : 'y', 'val' : 0.},
                                         {'id' : [3], 'dir' : 'z', 'val' : 0.}],
                            'neumann' : [{'id' : [2], 'dir' : 'normal_cur', 'curve' : 1}]}


    # problem setup
    problem = ambit_fe.ambit_main.Ambit(IO_PARAMS, [TIME_PARAMS_SOLID, TIME_PARAMS_FLOW0D], SOLVER_PARAMS, FEM_PARAMS, [MATERIALS, MODEL_PARAMS_FLOW0D], BC_DICT, time_curves=time_curves(), coupling_params=COUPLING_PARAMS)

    # solve time-dependent problem
    problem.solve_problem()


    # --- results check
    tol = 1.0e-7

    s_corr = np.zeros(problem.mp.pb0.cardvasc0D.numdof)

    # correct 0D results
    s_corr[0] = 6.6112734346791608E+00
    s_corr[1] = -5.0536832486004135E-01
    s_corr[2] = -6.5698331560523982E-03
    s_corr[3] = 4.1830472040250060E-04

    check1 = ambit_fe.resultcheck.results_check_vec(problem.mp.pb0.s, s_corr, problem.mp.comm, tol=tol)
    success = ambit_fe.resultcheck.success_check([check1], problem.mp.comm)

    if not success:
        raise RuntimeError("Test failed!")



if __name__ == "__main__":

    test_main()
