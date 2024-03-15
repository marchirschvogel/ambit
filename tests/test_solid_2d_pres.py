#!/usr/bin/env python3

"""
solid mechanics
2D U-shaped material under pressure at interior face
Gmsh msh format read-in
Compressible Neo-Hooke material
"""

import ambit_fe

import sys
import numpy as np
from pathlib import Path
import pytest


@pytest.mark.solid
def test_main():

    basepath = str(Path(__file__).parent.absolute())

    IO_PARAMS         = {'problem_type'          : 'solid',
                         'mesh_domain'           : basepath+'/input/ushape_2d.msh',
                         'mesh_boundary'         : None,
                         'mesh_dim'              : 2, # apparently needed for gmsh read-in...
                         'meshfile_format'       : 'gmsh',
                         'write_results_every'   : 1,
                         'output_path'           : basepath+'/tmp/',
                         'results_to_write'      : ['displacement'],
                         'simname'               : 'solid_2d_pres'}

    SOLVER_PARAMS     = {'solve_type'            : 'direct',
                         'tol_res'               : 1.0e-8,
                         'tol_inc'               : 1.0e-5}

    TIME_PARAMS       = {'maxtime'               : 1.0,
                         'numstep'               : 5,
                         'timint'                : 'static'}

    FEM_PARAMS        = {'order_disp'            : 1,
                         'quad_degree'           : 3}

    MATERIALS         = {'MAT1' : {'neohooke_compressible' : {'mu':10., 'nu':0.3}}}



    # define your load curves here (syntax: tcX refers to curve X, to be used in BC_DICT key 'curve' : [X,0,0], or 'curve' : X)
    class time_curves:

        def tc1(self, t):
            pmax = 0.003
            return -pmax*t/TIME_PARAMS['maxtime']


    BC_DICT           = { 'dirichlet' : [{'id' : [1], 'dir' : 'all', 'val' : 0.}], # id 1: left foot
                            'neumann' : [{'id' : [2,3,4], 'dir' : 'normal_cur', 'curve' : 1}] } # ids 2,3,4: right-inner, arch, left-inner edge


    # problem setup
    problem = ambit_fe.ambit_main.Ambit(IO_PARAMS, TIME_PARAMS, SOLVER_PARAMS, FEM_PARAMS, MATERIALS, BC_DICT, time_curves=time_curves())

    # solve time-dependent problem
    problem.solve_problem()


    # --- results check
    tol = 1.0e-6

    check_node = []
    check_node.append(np.array([0.5, -2.0, 0.0]))

    u_corr = np.zeros(2*len(check_node))

    ## correct results
    u_corr[0] = 4.9056740227796608E-01 # x
    u_corr[1] = 2.4138674507501251E-01 # y

    check1 = ambit_fe.resultcheck.results_check_node(problem.mp.u, check_node, u_corr, problem.mp.V_u, problem.mp.comm, tol=tol, nm='u')
    success = ambit_fe.resultcheck.success_check([check1], problem.mp.comm)

    if not success:
        raise RuntimeError("Test failed!")



if __name__ == "__main__":

    test_main()
