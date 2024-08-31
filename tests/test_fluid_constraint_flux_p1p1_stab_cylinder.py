#!/usr/bin/env python3

"""
transient incompressible Navier-Stokes flow in a cylinder with in-flux constraint at one boundary
- stabilized P1P1 elements for velocity and pressure
- trapezoidal midpoint time stepping scheme
- 2 material domains in fluid (w/ same parameters though)
"""

import ambit_fe

import sys
import numpy as np
from pathlib import Path
import pytest


@pytest.mark.fluid
@pytest.mark.fluid_constraint
def test_main():
    
    basepath = str(Path(__file__).parent.absolute())
    
    # reads in restart step from the command line
    try: restart_step = int(sys.argv[1])
    except: restart_step = 0
    
    IO_PARAMS           = {'problem_type'          : 'fluid_constraint',
                           'mesh_domain'           : basepath+'/input/cylinder_domain.xdmf',
                           'mesh_boundary'         : basepath+'/input/cylinder_boundary.xdmf',
                           'write_results_every'   : 1,
                           'write_restart_every'   : 1,
                           'restart_step'          : restart_step,
                           'output_path'           : basepath+'/tmp/',
                           'results_to_write'      : ['velocity','pressure'],
                           'simname'               : 'fluid_constraint_flux_p1p1_stab_cylinder'}

    SOLVER_PARAMS       = {'solve_type'            : 'direct',
                           'direct_solver'         : 'mumps',
                           'tol_res'               : 1.0e-8,
                           'tol_inc'               : 1.0e-8}

    TIME_PARAMS_FLUID   = {'maxtime'               : 1.0,
                           'numstep'               : 10,
                           'numstep_stop'          : 2,
                           'timint'                : 'ost',
                           'theta_ost'             : 0.5,
                           'eval_nonlin_terms'     : 'trapezoidal',
                           'fluid_governing_type'  : 'navierstokes_transient'}
    
    FEM_PARAMS          = {'order_vel'             : 1,
                           'order_pres'            : 1,
                           'quad_degree'           : 5,
                           'fluid_formulation'     : 'nonconservative', # nonconservative (default), conservative
                           'stabilization'         : {'scheme' : 'supg_pspg', 'vscale' : 1e3, 'dscales' : [1.,1.,1.], 'reduced_scheme' : True}}

    CONSTRAINT_PARAMS    = {'constraint_physics'   : [{'id' : [4], 'type' : 'flux', 'prescribed_curve' : 1}],
                            'multiplier_physics'   : [{'id' : [4], 'type' : 'pressure'}]}

    MATERIALS           = { 'MAT1' : {'newtonian' : {'mu' : 4.0e-6},
                                      'inertia' : {'rho' : 1.025e-6}},
                            'MAT2' : {'newtonian' : {'mu' : 4.0e-6},
                                      'inertia' : {'rho' : 1.025e-6}} }


    # define your load curves here (syntax: tcX refers to curve X, to be used in BC_DICT key 'curve' : [X,0,0], or 'curve' : X)
    class time_curves:

        def tc1(self, t):
            qini = 0.
            qmax = -1e4
            return (qmax-qini)*t/TIME_PARAMS_FLUID['maxtime'] + qini


    BC_DICT           = { 'dirichlet' : [{'id' : [1], 'dir' : 'all', 'val' : 0.}] } # lateral surf


    # problem setup
    problem = ambit_fe.ambit_main.Ambit(IO_PARAMS, TIME_PARAMS_FLUID, SOLVER_PARAMS, FEM_PARAMS, MATERIALS, BC_DICT, time_curves=time_curves(), coupling_params=CONSTRAINT_PARAMS)

    # solve time-dependent problem
    problem.solve_problem()


    # --- results check
    tol = 1.0e-6

    check_node = []
    check_node.append(np.array([0.0170342, 2.99995, 13.4645]))

    v_corr, p_corr = np.zeros(3*len(check_node)), np.zeros(len(check_node))

    # correct results
    v_corr[0] = -5.7966875760150955E+00 # x
    v_corr[1] = 6.3978144245133038E+01 # y
    v_corr[2] = 7.1588985130116392E+00 # z

    p_corr[0] = 3.3288329107831207E-03

    check1 = ambit_fe.resultcheck.results_check_node(problem.mp.pbf.v, check_node, v_corr, problem.mp.pbf.V_v, problem.mp.comm, tol=tol, nm='v', readtol=1e-4)
    check2 = ambit_fe.resultcheck.results_check_node(problem.mp.pbf.p, check_node, p_corr, problem.mp.pbf.V_p, problem.mp.comm, tol=tol, nm='p', readtol=1e-4)

    success = ambit_fe.resultcheck.success_check([check1,check2], problem.mp.comm)

    if not success:
        raise RuntimeError("Test failed!")



if __name__ == "__main__":
    
    test_main()
