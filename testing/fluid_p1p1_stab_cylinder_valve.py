#!/usr/bin/env python3

# tests:
# - transient incompressible Navier-Stokes flow in a cylinder with axial Neumann and two outflows
# - Stabilized P1P1 elements for velocity and pressure
# - Backward-Euler time stepping scheme
# - 2 material domains in fluid (w/ same parameters though)
# - closed internal valve, requiring duplicate pressure nodes at that internal surface
# - works currently only with mixed dolfinx branch (USE_MIXED_DOLFINX_BRANCH)

import ambit

import sys, traceback
import numpy as np
from pathlib import Path

import resultcheck


def main():
    
    basepath = str(Path(__file__).parent.absolute())
    

    IO_PARAMS           = {'problem_type'          : 'fluid',
                           'USE_MIXED_DOLFINX_BRANCH' : True,
                           'duplicate_mesh_domains': [1,2],
                           'mesh_domain'           : basepath+'/input/cylinder_domain.xdmf',
                           'mesh_boundary'         : basepath+'/input/cylinder_boundary.xdmf',
                           'write_results_every'   : 1,
                           'output_path'           : basepath+'/tmp/',
                           'results_to_write'      : ['velocity','pressure'],
                           'simname'               : 'fluid_p1p1_stab_cylinder_valve'}

    SOLVER_PARAMS       = {'solve_type'            : 'direct',
                           'direct_solver'         : 'mumps',
                           'tol_res'               : 1.0e-8,
                           'tol_inc'               : 1.0e-8}

    TIME_PARAMS_FLUID   = {'maxtime'               : 1.0,
                           'numstep'               : 100,
                           'numstep_stop'          : 3,
                           'timint'                : 'ost',
                           'theta_ost'             : 1.0,
                           'fluid_governing_type'  : 'navierstokes_transient'}
    
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
            t_ramp = 1.0
            pmax = 1.5
            return 0.5*(-(pmax))*(1.-np.cos(np.pi*t/t_ramp))


    BC_DICT        = { 'dirichlet' : [{'id' : [1,5], 'dir' : 'all', 'val' : 0.}], # 5 is internal surface (valve)
                       'neumann' : [{'id' : [2], 'dir' : 'normal_cur', 'curve' : 1}] }


    # problem setup
    problem = ambit.Ambit(IO_PARAMS, TIME_PARAMS_FLUID, SOLVER_PARAMS, FEM_PARAMS, MATERIALS, BC_DICT, time_curves=time_curves())

    # solve time-dependent problem
    problem.solve_problem()


    # --- results check
    tol = 1.0e-6

    check_node = []
    check_node.append(np.array([0.0170342, 2.99995, 13.4645]))

    v_corr, p_corr = np.zeros(3*len(check_node)), np.zeros(len(check_node))

    # correct results
    v_corr[0] = -1.3770531845487710E+00 # x
    v_corr[1] = 1.3042877526484933E+01 # y
    v_corr[2] = -1.2323336148783317E+00 # z

    check1 = resultcheck.results_check_node(problem.mp.v, check_node, v_corr, problem.mp.V_v, problem.mp.comm, tol=tol, nm='v', readtol=1e-4)

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
