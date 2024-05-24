#!/usr/bin/env python3

"""
FrSI test case of an axially clamped arterial segment that is prestressed prior to being subject to a constant load
We have some mild dynamics in there, therefore the velocity state is not perfectly zero.
Tests transient Stokes, conservative formulation.
"""

import ambit_fe

import sys
import numpy as np
from pathlib import Path
import pytest


@pytest.mark.frsi
@pytest.mark.fluid_ale
@pytest.mark.rom
def test_main():

    basepath = str(Path(__file__).parent.absolute())

    IO_PARAMS            = {'problem_type'          : 'fluid_ale',
                            'write_results_every'   : 1,
                            'output_path'           : basepath+'/tmp/',
                            'mesh_domain'           : basepath+'/input/artseg-quad_domain.xdmf',
                            'mesh_boundary'         : basepath+'/input/artseg-quad_boundary.xdmf',
                            'results_to_write'      : [['fluiddisplacement','velocity','pressure'],['aledisplacement']], # first fluid, then ale results
                            'simname'               : 'frsi_artseg_prestress'}

    ROM_PARAMS           = {'hdmfilenames'          : [basepath+'/input/artseg_vel_snapshot-*.txt'],
                            'numsnapshots'          : 1,
                            'snapshotincr'          : 1,
                            'numredbasisvec'        : 1,
                            'eigenvalue_cutoff'     : 1.0e-8,
                            'print_eigenproblem'    : True,
                            'surface_rom'           : [1,6]}

    SOLVER_PARAMS        = {'solve_type'            : 'direct',
                            'tol_res'               : [1.0e-8,1.0e-8,1.0e-1],
                            'tol_inc'               : [1.0e-1,1.0e-3,1.0e-1]}

    TIME_PARAMS          = {'maxtime'               : 3.0,
                            'numstep'               : 150,
                            'numstep_stop'          : 1,
                            'timint'                : 'ost',
                            'theta_ost'             : 1.0,
                            'fluid_governing_type'  : 'stokes_transient'}

    FEM_PARAMS_FLUID     = {'order_vel'             : 2,
                            'order_pres'            : 1,
                            'quad_degree'           : 6,
                            'fluid_formulation'     : 'conservative',
                            'prestress_initial'     : True,
                            'prestress_kinetic'     : 'stokes_transient',
                            'prestress_maxtime'     : 1.0,
                            'prestress_numstep'     : 30}

    FEM_PARAMS_ALE       = {'order_disp'            : 2,
                            'quad_degree'           : 6}

    COUPLING_PARAMS      = {'coupling_fluid_ale'    : [{'surface_ids' : [1,6], 'type' : 'strong_dirichlet'}]}

    MATERIALS_FLUID      = { 'MAT1' : {'newtonian' : {'mu' : 4.0e-6},
                                       'inertia'   : {'rho' : 1.025e-6}} }

    MATERIALS_ALE        = { 'MAT1' : {'diffusion' : {'D' : 1.0, 'scale_det' : True}} }


    class expression1: # prestress
        def __init__(self):
            self.t = 0.0
        def evaluate(self, x):
            pmax = 0.01
            val = (-pmax)*self.t/0.5 * (self.t < 0.5) + (-pmax) * (self.t >= 0.5)
            return np.full(x.shape[1], val)

    class expression2: # post-stress load
        def __init__(self):
            self.t = 0.0
        def evaluate(self, x):
            val = -0.01
            return np.full(x.shape[1], val)


    BC_DICT_ALE          = { 'dirichlet' : [{'id' : [2,3], 'dir' : 'z', 'val' : 0.},
                                            {'id' : [4], 'dir' : 'y', 'val' : 0.},
                                            {'id' : [5], 'dir' : 'x', 'val' : 0.}] }

    BC_DICT_FLUID        = { 'membrane' :  [{'id' : [1,6], 'params' : {'model' : 'membrane', 'a_0' : 1.0, 'b_0' : 6.0, 'eta' : 0., 'rho0' : 0., 'h0' : {'val' : 0.1}}}],
                             'neumann_prestress' : [{'id' : [2,3], 'dir' : 'normal_ref', 'expression' : expression1}],
                             'neumann' :   [{'id' : [2,3], 'dir' : 'normal_cur', 'expression' : expression2}],
                             'dirichlet' : [{'id' : [4], 'dir' : 'y', 'val' : 0.},
                                            {'id' : [5], 'dir' : 'x', 'val' : 0.}] }

    # problem setup
    problem = ambit_fe.ambit_main.Ambit(IO_PARAMS, TIME_PARAMS, SOLVER_PARAMS, [FEM_PARAMS_FLUID, FEM_PARAMS_ALE], [MATERIALS_FLUID, MATERIALS_ALE], [BC_DICT_FLUID, BC_DICT_ALE], coupling_params=COUPLING_PARAMS, mor_params=ROM_PARAMS)

    # problem solve
    problem.solve_problem()


    # --- results check
    tol = 1.0e-6

    check_node = []
    check_node.append(np.array([7.071068, 7.071068, 2.500000]))

    v_corr = np.zeros(3*len(check_node))

    # correct results
    v_corr[0] = -4.8193958711105689E-03 # x
    v_corr[1] = -4.8193958711105689E-03 # y
    v_corr[2] = 0.0 # z

    check1 = ambit_fe.resultcheck.results_check_node(problem.mp.pbf.v, check_node, v_corr, problem.mp.pbf.V_v, problem.mp.comm, tol=tol, nm='v', readtol=1e-4)

    success = ambit_fe.resultcheck.success_check([check1], problem.mp.comm)

    if not success:
        raise RuntimeError("Test failed!")



if __name__ == "__main__":

    test_main()
