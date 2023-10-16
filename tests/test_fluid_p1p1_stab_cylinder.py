#!/usr/bin/env python3

"""
steady incompressible Navier-Stokes flow in a cylinder with axial Neumann and two outflows
- stabilized P1P1 elements for velocity and pressure
- Backward-Euler time stepping scheme
- 2 material domains in fluid (w/ same parameters though)
"""

import ambit_fe

import sys
import numpy as np
from pathlib import Path
import pytest


@pytest.mark.fluid
def test_main():
    
    basepath = str(Path(__file__).parent.absolute())
    

    IO_PARAMS           = {'problem_type'          : 'fluid',
                           'mesh_domain'           : basepath+'/input/cylinder_domain.xdmf',
                           'mesh_boundary'         : basepath+'/input/cylinder_boundary.xdmf',
                           'write_results_every'   : 1,
                           'output_path'           : basepath+'/tmp/',
                           'results_to_write'      : ['velocity','pressure'],
                           'simname'               : 'fluid_p1p1_stab_cylinder'}

    SOLVER_PARAMS       = {'solve_type'            : 'direct',
                           'direct_solver'         : 'mumps',
                           'tol_res'               : 1.0e-8,
                           'tol_inc'               : 1.0e-8}

    TIME_PARAMS_FLUID   = {'maxtime'               : 1.0,
                           'numstep'               : 10,
                           'numstep_stop'          : 2,
                           'timint'                : 'ost',
                           'theta_ost'             : 1.0,
                           'fluid_governing_type'  : 'navierstokes_steady'}
    
    FEM_PARAMS          = {'order_vel'             : 1,
                           'order_pres'            : 1,
                           'quad_degree'           : 5,
                           'fluid_formulation'     : 'nonconservative', # nonconservative (default), conservative
                           'stabilization'         : {'scheme' : 'supg_pspg2', 'vscale' : 1e3, 'dscales' : [1.,1.,1.]}}


    MATERIALS           = { 'MAT1' : {'newtonian' : {'mu' : 4.0e-6},
                                      'inertia' : {'rho' : 1.025e-6}},
                            'MAT2' : {'newtonian' : {'mu' : 4.0e-6},
                                      'inertia' : {'rho' : 1.025e-6}} }


    # define your load curves here (syntax: tcX refers to curve X, to be used in BC_DICT key 'curve' : [X,0,0], or 'curve' : X)
    class time_curves():
        
        def tc1(self, t):
            return -0.001*np.sin(2.*np.pi*t/TIME_PARAMS_FLUID['maxtime'])


    BC_DICT           = { 'dirichlet' : [{'id' : [1], 'dir' : 'all', 'val' : 0.}], # lateral surf
                          'neumann' : [{'id' : [4], 'dir' : 'xyz_ref', 'curve' : [0,0,1]}]} # inflow; 2,3 are outflows


    # problem setup
    problem = ambit_fe.ambit_main.Ambit(IO_PARAMS, TIME_PARAMS_FLUID, SOLVER_PARAMS, FEM_PARAMS, MATERIALS, BC_DICT, time_curves=time_curves())

    # solve time-dependent problem
    problem.solve_problem()


    # --- results check
    tol = 1.0e-6

    check_node = []
    check_node.append(np.array([0.0170342, 2.99995, 13.4645]))

    v_corr, p_corr = np.zeros(3*len(check_node)), np.zeros(len(check_node))

    # correct results
    v_corr[0] = 3.0401864063179573E-01 # x
    v_corr[1] = -3.7615292073863640E+00 # y
    v_corr[2] = -7.5782395441818418E-02 # z

    p_corr[0] = -2.1474832854517510E-04

    check1 = ambit_fe.resultcheck.results_check_node(problem.mp.v, check_node, v_corr, problem.mp.V_v, problem.mp.comm, tol=tol, nm='v', readtol=1e-4)
    check2 = ambit_fe.resultcheck.results_check_node(problem.mp.p, check_node, p_corr, problem.mp.V_p, problem.mp.comm, tol=tol, nm='p', readtol=1e-4)

    success = ambit_fe.resultcheck.success_check([check1,check2], problem.mp.comm)

    if not success:
        raise RuntimeError("Test failed!")



if __name__ == "__main__":
    
    test_main()
