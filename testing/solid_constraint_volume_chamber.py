#!/usr/bin/env python3

import ambit

import sys, time, traceback
import numpy as np
from pathlib import Path

import results_check


def main():
    
    basepath = str(Path(__file__).parent.absolute())

    IO_PARAMS            = {'problem_type'          : 'solid_constraint', # solid, fluid, flow0d, solid_flow0d, fluid_flow0d
                            'mesh_domain'           : ''+basepath+'/input/chamber_domain.xdmf',
                            'mesh_boundary'         : ''+basepath+'/input/chamber_boundary.xdmf',
                            'write_results_every'   : -999,
                            'output_path'           : ''+basepath+'/tmp/',
                            'results_to_write'      : ['displacement','pressure'],
                            'simname'               : 'solid_constraint_volume_chamber'}

    SOLVER_PARAMS_SOLID  = {'solve_type'            : 'direct', # direct, iterative
                            'tol_res'               : 1.0e-8,
                            'tol_inc'               : 1.0e-8}

    SOLVER_PARAMS_CONSTR = {'tol_res'               : 1.0e-8,
                            'tol_inc'               : 1.0e-8}

    TIME_PARAMS_SOLID    = {'maxtime'               : 1.0,
                            'numstep'               : 10,
                            'numstep_stop'          : 5,
                            'timint'                : 'ost',
                            'theta_ost'             : 1.0}

    FEM_PARAMS           = {'order_disp'            : 1,
                            'order_pres'            : 1,
                            'quad_degree'           : 1,
                            'incompressible_2field' : True} # True, False
    
    CONSTRAINT_PARAMS    = {'surface_ids'           : [[3]],
                            'constraint_quantity'   : 'volume',
                            'prescribed_curve'      : [1]}

    MATERIALS            = {'MAT1' : {'neohooke_dev' : {'mu' : 100., 'rho0' : 1.0e-6}, 'inertia' : {'rho0' : 1.0e-6}}}

    # define your load curves here (syntax: tcX refers to curve X, to be used in BC_DICT key 'curve' : [X,0,0], or 'curve' : X)
    class time_curves():
        
        def tc1(self, t):
            vini = 1.
            vmax = 2.0
            return (vmax-vini)*t/TIME_PARAMS_SOLID['maxtime'] + vini



    BC_DICT           = { 'dirichlet' : [{'id' : [1], 'dir' : 'x', 'val' : 0.},
                                         {'id' : [3], 'dir' : 'y', 'val' : 0.},
                                         {'id' : [3], 'dir' : 'z', 'val' : 0.}]}


    # problem setup
    problem = ambit.Ambit(IO_PARAMS, TIME_PARAMS_SOLID, [SOLVER_PARAMS_SOLID, SOLVER_PARAMS_CONSTR], FEM_PARAMS, MATERIALS, BC_DICT, time_curves=time_curves(), coupling_params=CONSTRAINT_PARAMS)
    
    # solve time-dependent problem
    problem.solve_problem()

    # --- results check
    tol = 1.0e-6

    check_node = []
    check_node.append(np.array([1.5, 0.75, 0.75]))

    u_corr = np.zeros(3*len(check_node))
    
    ## correct results
    u_corr[0] = 7.1440094913591246E-01 # x
    u_corr[1] = -1.1768897247463314E-02 # y
    u_corr[2] = 4.5878411920493023E-03 # z

    check1 = results_check.results_check_node(problem.mp.pbs.u, check_node, u_corr, problem.mp.pbs.V_u, problem.mp.comm, tol=tol, nm='u')
    success = results_check.success_check([check1], problem.mp.comm)
    
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
