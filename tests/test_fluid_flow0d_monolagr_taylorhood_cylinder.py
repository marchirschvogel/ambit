#!/usr/bin/env python3

"""
monolithic 3D-0D coupling of incompressible Navier-Stokes with 2-element Windkessel model
"""

import ambit_fe

import sys
import numpy as np
from pathlib import Path
import pytest


@pytest.mark.fluid_flow0d
def test_main():

    basepath = str(Path(__file__).parent.absolute())

    # reads in restart step from the command line
    try: restart_step = int(sys.argv[1])
    except: restart_step = 0

    IO_PARAMS            = {'problem_type'          : 'fluid_flow0d',
                            'mesh_domain'           : basepath+'/input/cylinder-quad_domain.xdmf',
                            'mesh_boundary'         : basepath+'/input/cylinder-quad_boundary.xdmf',
                            'write_results_every'   : -1,
                            'write_restart_every'   : 1,
                            'restart_step'          : restart_step,
                            'output_path'           : basepath+'/tmp/',
                            'results_to_write'      : [],
                            'simname'               : 'fluid_flow0d_monolagr_taylorhood_cylinder'}

    SOLVER_PARAMS       =  {'solve_type'            : 'direct',
                            'direct_solver'         : 'superlu_dist', # no idea why, but mumps does not seem to like this system in parallel...
                            'tol_res'               : 1.0e-8,
                            'tol_inc'               : 1.0e-8,
                            'subsolver_params'      : {'tol_res' : 1.0e-8, 'tol_inc' : 1.0e-8}}

    TIME_PARAMS_FLUID   =  {'maxtime'               : 1.0,
                            'numstep'               : 10,
                            'numstep_stop'          : 2,
                            'timint'                : 'ost',
                            'theta_ost'             : 1.0}

    TIME_PARAMS_FLOW0D   = {'timint'                : 'ost',
                            'theta_ost'             : 1.0,
                            'initial_conditions'    : {'Q_0' : 0.0, 'p_0' : 0.0}}

    MODEL_PARAMS_FLOW0D  = {'modeltype'             : '2elwindkessel',
                            'parameters'            : {'C' : 1.0e3, 'R' : 1.0e-2, 'p_ref' : 0.1}}

    FEM_PARAMS           = {'order_vel'             : 2,
                            'order_pres'            : 1,
                            'quad_degree'           : 5}

    COUPLING_PARAMS      = {'surface_ids'           : [[3]],
                            'coupling_quantity'     : ['pressure'],
                            'variable_quantity'     : ['flux'],
                            'print_subiter'         : True}

    MATERIALS         = { 'MAT1' : {'newtonian' : {'mu' : 4.0e-6},
                                    'inertia' : {'rho' : 1.025e-6}} }


    # define your load curves here (syntax: tcX refers to curve X, to be used in BC_DICT key 'curve' : [X,0,0], or 'curve' : X)
    class time_curves():

        def tc1(self, t):
            return -0.001*np.sin(2.*np.pi*t/TIME_PARAMS_FLUID['maxtime'])


    BC_DICT           = { 'dirichlet' : [{'id' : [1], 'dir' : 'all', 'val' : 0.}], # lateral surf
                          'neumann' : [{'id' : [4], 'dir' : 'xyz_ref', 'curve' : [0,0,1]}]} # inflow; 2,3 are outflows


    # problem setup
    problem = ambit_fe.ambit_main.Ambit(IO_PARAMS, [TIME_PARAMS_FLUID, TIME_PARAMS_FLOW0D], SOLVER_PARAMS, FEM_PARAMS, [MATERIALS, MODEL_PARAMS_FLOW0D], BC_DICT, time_curves=time_curves(), coupling_params=COUPLING_PARAMS)


    # solve time-dependent problem
    problem.solve_problem()


    # --- results check
    tol = 1.0e-6

    check_node = []
    check_node.append(np.array([0.0170342, 2.99995, 13.4645]))

    v_corr, p_corr = np.zeros(3*len(check_node)), np.zeros(len(check_node))

    # correct results
    v_corr[0] = -1.7208654619616248E-02 # x
    v_corr[1] = -5.9053964448523599E-01 # y
    v_corr[2] = -3.6933688903433675E+00 # z

    p_corr[0] = -2.9127423264346895E-04

    check1 = ambit_fe.resultcheck.results_check_node(problem.mp.pbf.v, check_node, v_corr, problem.mp.pbf.V_v, problem.mp.comm, tol=tol, nm='v', readtol=1e-4)
    check2 = ambit_fe.resultcheck.results_check_node(problem.mp.pbf.p, check_node, p_corr, problem.mp.pbf.V_p, problem.mp.comm, tol=tol, nm='p', readtol=1e-4)

    success = ambit_fe.resultcheck.success_check([check1,check2], problem.mp.comm)

    if not success:
        raise RuntimeError("Test failed!")



if __name__ == "__main__":

    test_main()
