#!/usr/bin/env python3

"""
surface membrane model, simple deformation modes - tests correct quasi-static membrane stress response
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
                         'mesh_domain'           : basepath+'/input/blocks5hex_domain.xdmf',
                         'mesh_boundary'         : basepath+'/input/blocks5hex_boundary.xdmf',
                         'write_results_every'   : 1,
                         'output_path'           : basepath+'/tmp/',
                         'results_to_write'      : ['displacement','cauchystress_membrane_principal','strainenergy_membrane'],
                         'simname'               : 'solid_membrane'}

    SOLVER_PARAMS     = {'solve_type'            : 'direct',
                         'tol_res'               : 1.0e-8,
                         'tol_inc'               : 1.0e-5,
                         'maxiter'               : 25,
                         'divergence_continue'   : None}

    TIME_PARAMS       = {'maxtime'               : 1.0,
                         'numstep'               : 1,
                         'timint'                : 'static'}

    FEM_PARAMS        = {'order_disp'            : 1,
                         'quad_degree'           : 5,
                         'incompressible_2field' : False}

    MATERIALS         = {'MAT1' : {'neohooke_compressible' : {'mu' : 10.0, 'nu' : 0.49}},
                         'MAT2' : {'neohooke_compressible' : {'mu' : 10.0, 'nu' : 0.49}},
                         'MAT3' : {'neohooke_compressible' : {'mu' : 10.0, 'nu' : 0.49}},
                         'MAT4' : {'neohooke_compressible' : {'mu' : 10.0, 'nu' : 0.49}},
                         'MAT5' : {'neohooke_compressible' : {'mu' : 10.0, 'nu' : 0.49}}}


    # define your load curves here (syntax: tcX refers to curve X, to be used in BC_DICT key 'curve' : [X,0,0], or 'curve' : X)
    class time_curves():

        def tc1(self, t):
            return 0.5*t

    BC_DICT           = { 'membrane' :  [{'id' : [5], 'params' : {'model' : 'membrane', 'a_0' : 1.0, 'b_0' : 6.0, 'eta' : 0., 'rho0' : 0., 'h0' : {'val' : 0.1}}}, # block1
                                         {'id' : [11], 'params' : {'model' : 'membrane', 'a_0' : 1.0, 'b_0' : 6.0, 'eta' : 0., 'rho0' : 0., 'h0' : {'val' : 0.1}}}, # block2
                                         {'id' : [17], 'params' : {'model' : 'membrane', 'a_0' : 1.0, 'b_0' : 6.0, 'eta' : 0., 'rho0' : 0., 'h0' : {'val' : 0.1}}}, # block3
                                         {'id' : [23], 'params' : {'model' : 'membrane', 'a_0' : 1.0, 'b_0' : 6.0, 'eta' : 0., 'rho0' : 0., 'h0' : {'val' : 0.1}}}, # block4
                                         {'id' : [29], 'params' : {'model' : 'membrane', 'a_0' : 1.0, 'b_0' : 6.0, 'eta' : 0., 'rho0' : 0., 'h0' : {'val' : 0.1}}}], # block5
                          # block1 - tension in membrane plane
                          'dirichlet' : [{'id' : [3,6], 'dir' : 'z', 'val' : 0.}, # out of 2D plane XY
                                         {'id' : [1], 'dir' : 'all', 'val' : 0.},
                                         {'id' : [4], 'dir' : 'y', 'val' : 0.},
                                         {'id' : [4], 'dir' : 'x', 'curve' : 1}, # tension in membrane plane
                          # block2 - tension out of membrane plane
                                         {'id' : [9,12], 'dir' : 'z', 'val' : 0.}, # out of 2D plane XY
                                         {'id' : [8], 'dir' : 'all', 'val' : 0.},
                                         {'id' : [11], 'dir' : 'x', 'val' : 0.},
                                         {'id' : [11], 'dir' : 'y', 'curve' : 1}, # tension out of membrane plane
                          # block3 - simple shear in membrane plane
                                         {'id' : [15,18], 'dir' : 'z', 'val' : 0.}, # out of 2D plane XY
                                         {'id' : [14], 'dir' : 'all', 'val' : 0.},
                                         {'id' : [17], 'dir' : 'y', 'val' : 0.},
                                         {'id' : [17], 'dir' : 'x', 'curve' : 1}, # simple shear in membrane plane
                          # block4 - simple shear out of membrane plane
                                         {'id' : [21,24], 'dir' : 'z', 'val' : 0.}, # out of 2D plane XY
                                         {'id' : [19], 'dir' : 'all', 'val' : 0.},    
                                         {'id' : [22], 'dir' : 'x', 'val' : 0.},
                                         {'id' : [22], 'dir' : 'y', 'curve' : 1}, # simple shear out of membrane plane
                          # block5 - pure shear
                                         {'id' : [27,30], 'dir' : 'z', 'val' : 0.}, # out of 2D plane XY
                                         {'id' : [26], 'dir' : 'all', 'val' : 0.},
                                         {'id' : [29], 'dir' : 'y', 'val' : 0.},
                                         {'id' : [28], 'dir' : 'x', 'val' : 0.},
                                         {'id' : [29], 'dir' : 'x', 'curve' : 1}, # pure shear
                                         {'id' : [28], 'dir' : 'y', 'curve' : 1}] } # pure shear

    # problem setup
    problem = ambit_fe.ambit_main.Ambit(IO_PARAMS, TIME_PARAMS, SOLVER_PARAMS, FEM_PARAMS, MATERIALS, BC_DICT, time_curves=time_curves())

    # solve time-dependent problem
    problem.solve_problem()

    # --- results check
    tol = 1.0e-6

    check_node = [] # coordinates of discontinuous function space for each block
    check_node.append(np.array([0.5, 0.5, 0.5]))
    check_node.append(np.array([2.5, 0.5, 0.5]))
    check_node.append(np.array([4.5, 0.5, 0.5]))
    check_node.append(np.array([6.5, 0.5, 0.5]))
    check_node.append(np.array([8.5, 0.5, 0.5]))

    cauchy_corr = np.zeros(3*len(check_node))

    # correct results - principal Cauchy membrane stresses
    # tension in membrane plane
    cauchy_corr[0] = 116.45850137
    cauchy_corr[1] = 35.83338504
    cauchy_corr[2] = 0.
    # tension out of membrane plane
    cauchy_corr[3] = 0.
    cauchy_corr[4] = 0.
    cauchy_corr[5] = 0.
    # simple shear in membrane plane
    cauchy_corr[6] = 0.
    cauchy_corr[7] = 0.
    cauchy_corr[8] = 0.
    # simple shear out of membrane plane
    cauchy_corr[9] = 0.60743646
    cauchy_corr[10] = 0.26997176
    cauchy_corr[11] = 0.
    # pure shear
    cauchy_corr[12] = 0.60743646
    cauchy_corr[13] = 0.26997176
    cauchy_corr[14] = 0.

    check1 = ambit_fe.resultcheck.results_check_node(problem.mp.io.cauchystress_membrane_principal, check_node, cauchy_corr, problem.mp.Vd_vector, problem.mp.comm, tol=tol, nm='sigma')
    success = ambit_fe.resultcheck.success_check([check1], problem.mp.comm)
    
    if not success:
        raise RuntimeError("Test failed!")



if __name__ == "__main__":

    test_main()
