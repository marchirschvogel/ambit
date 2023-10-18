#!/usr/bin/env python3

"""
transient incompressible Navier-Stokes flow in a cylinder with axial Neumann and two outflows
- stabilized P1P1 elements for velocity and pressure (full SUPF/PSPG scheme)
- Backward-Euler time stepping scheme
- 2 material domains in fluid (w/ same parameters though)
- internal valve, requiring duplicate pressure nodes at that internal surface
- works currently only with mixed dolfinx branch (USE_MIXED_DOLFINX_BRANCH)
- checkpoint writing of domain-wise discontinuous pressure field
"""

import ambit_fe

import sys
import numpy as np
from pathlib import Path
import pytest


@pytest.mark.fluid
def test_main():
    
    basepath = str(Path(__file__).parent.absolute())

    # reads in restart step from the command line
    try: restart_step = int(sys.argv[1])
    except: restart_step = 0

    IO_PARAMS           = {'problem_type'          : 'fluid',
                           'USE_MIXED_DOLFINX_BRANCH' : True,
                           'duplicate_mesh_domains': [1,2],
                           'mesh_domain'           : basepath+'/input/cylinder_domain.xdmf',
                           'mesh_boundary'         : basepath+'/input/cylinder_boundary.xdmf',
                           'write_results_every'   : 1,
                           'write_restart_every'   : 9,
                           'restart_step'          : restart_step,
                           'output_path'           : basepath+'/tmp/',
                           'results_to_write'      : ['velocity','pressure'],
                           'simname'               : 'fluid_p1p1_stab_cylinder_valve'}

    SOLVER_PARAMS       = {'solve_type'            : 'direct',
                           'direct_solver'         : 'mumps',
                           'tol_res'               : 1.0e-8,
                           'tol_inc'               : 1.0e-8}

    TIME_PARAMS_FLUID   = {'maxtime'               : 1.0,
                           'numstep'               : 10,
                           #'numstep_stop'          : 3,
                           'timint'                : 'ost',
                           'theta_ost'             : 1.0,
                           'fluid_governing_type'  : 'navierstokes_transient'}
    
    FEM_PARAMS          = {'order_vel'             : 1,
                           'order_pres'            : 1,
                           'quad_degree'           : 5,
                           'fluid_formulation'     : 'nonconservative', # nonconservative (default), conservative
                           'stabilization'         : {'scheme' : 'supg_pspg', 'vscale' : 1e3, 'dscales' : [1.,1.,1.]}}


    MATERIALS           = { 'MAT1' : {'newtonian' : {'mu' : 4.0e-6},
                                      'inertia' : {'rho' : 1.025e-6}},
                            'MAT2' : {'newtonian' : {'mu' : 4.0e-6},
                                      'inertia' : {'rho' : 1.025e-6}} }


    # define your load curves here (syntax: tcX refers to curve X, to be used in BC_DICT key 'curve' : [X,0,0], or 'curve' : X)
    class time_curves():
        
        def tc1(self, t):
            t_ramp = 1.0
            pmax = 1.0
            return 0.5*(-(pmax))*(1.-np.cos(np.pi*t/t_ramp))

        def tc2(self, t):
            pmax = 0.5
            return -pmax


    BC_DICT        = { 'dirichlet'   : [{'id' : [1], 'dir' : 'all', 'val' : 0.}],
                       'neumann'     : [{'id' : [2], 'dir' : 'normal_ref', 'curve' : 1},
                                        {'id' : [4], 'dir' : 'normal_ref', 'curve' : 2}],
                       'robin_valve' : [{'id' : [5], 'type' : 'dp_smooth', 'beta_max' : 1e3, 'beta_min' : 1e-3, 'epsilon' : 1e-6, 'dp_monitor_id' : 0}], # 5 is internal surface (valve)
                       'dp_monitor'  : [{'id' : [5], 'upstream_domain' : 2, 'downstream_domain' : 1}], 
                       'flux_monitor': [{'id' : [5], 'on_subdomain' : True, 'internal' : False, 'domain' : 2}] }


    # problem setup
    problem = ambit_fe.ambit_main.Ambit(IO_PARAMS, TIME_PARAMS_FLUID, SOLVER_PARAMS, FEM_PARAMS, MATERIALS, BC_DICT, time_curves=time_curves())

    # solve time-dependent problem
    problem.solve_problem()


    # --- results check
    tol = 1.0e-6

    check_node = []
    check_node.append(np.array([0.0170342, 2.99995, 13.4645]))

    v_corr = np.zeros(3*len(check_node))

    # correct results
    v_corr[0] = 2.8679355190833773E+00 # x
    v_corr[1] = 1.1644703461381966E+03 # y
    v_corr[2] = -9.9701079153832620E+02 # z

    check1 = ambit_fe.resultcheck.results_check_node(problem.mp.v, check_node, v_corr, problem.mp.V_v, problem.mp.comm, tol=tol, nm='v', readtol=1e-4)

    success = ambit_fe.resultcheck.success_check([check1], problem.mp.comm)

    if not success:
        raise RuntimeError("Test failed!")



if __name__ == "__main__":
    
    test_main()
