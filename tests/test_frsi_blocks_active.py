#!/usr/bin/env python3

"""
FrSI test case of three blocks with an active membrane surface, paritioned POD space per block that allows in-plane contraction of the membrane
tests also active stress weight for first block (though using only the first partition field, hence multiply by 1)
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
                            'mesh_domain'           : basepath+'/input/blocks3_domain.xdmf',
                            'mesh_boundary'         : basepath+'/input/blocks3_boundary.xdmf',
                            'fiber_data'            : [basepath+'/input/fib_blocks3_c.txt',basepath+'/input/fib_blocks3_l.txt'],
                            'results_to_write'      : [['velocity','pressure'],['aledisplacement']], # first fluid, then ale results
                            'simname'               : 'frsi_blocks_active'}

    ROM_PARAMS           = {'modes_from_files'      : [basepath+'/input/blocks3_mode_*.txt'],
                            'partitions'            : [basepath+'/input/blocks3_part-1.txt',basepath+'/input/blocks3_part-2.txt',basepath+'/input/blocks3_part-3.txt'],
                            'orthogonalize_rom_basis' : True,
                            'numredbasisvec'        : 2,
                            'numredbasisvec_partition' : [2,2,2],
                            'surface_rom'           : [2,8,14],
                            'write_pod_modes'       : True}

    SOLVER_PARAMS        = {'solve_type'            : 'direct',
                            'tol_res'               : [1.0e-8,1.0e-8,1.0e-5],
                            'tol_inc'               : [1.0e-3,1.0e-3,1.0e-3]}

    TIME_PARAMS          = {'maxtime'               : 0.3,
                            'numstep'               : 50,
                            #'numstep_stop'          : 1,
                            'timint'                : 'ost',
                            'theta_ost'             : 0.67,
                            'eval_nonlin_terms'     : 'midpoint',
                            'fluid_governing_type'  : 'stokes_transient'}

    FEM_PARAMS_FLUID     = {'order_vel'             : 1,
                            'order_pres'            : 1,
                            'quad_degree'           : 5,
                            'stabilization'         : {'scheme'         : 'supg_pspg',
                                                       'vscale'         : 1e3,
                                                       'dscales'        : [1.,1.,1.],
                                                       'symmetric'      : False,
                                                       'reduced_scheme' : False}}

    FEM_PARAMS_ALE       = {'order_disp'            : 1,
                            'quad_degree'           : 5}

    COUPLING_PARAMS      = {'coupling_fluid_ale'    : [{'surface_ids' : [2,8,14], 'type' : 'strong_dirichlet'}]}

    MATERIALS_FLUID      = { 'MAT1' : {'newtonian' : {'mu' : 4.0e-6},
                                       'inertia'   : {'rho' : 1.025e-6}},
                             'MAT2' : {'newtonian' : {'mu' : 4.0e-6},
                                       'inertia'   : {'rho' : 1.025e-6}},
                             'MAT3' : {'newtonian' : {'mu' : 4.0e-6},
                                       'inertia'   : {'rho' : 1.025e-6}}}

    MATERIALS_ALE        = { 'MAT1' : {'diffusion' : {'D' : 1.0}},
                             'MAT2' : {'diffusion' : {'D' : 1.0}},
                             'MAT3' : {'diffusion' : {'D' : 1.0}}}


    class time_curves():

        def tc1(self, t):
            
            K = 5.
            t_contr, t_relax = 0.0, 0.2
            
            alpha_max = BC_DICT_FLUID['membrane'][0]['params']['active_stress']['alpha_max']
            alpha_min = BC_DICT_FLUID['membrane'][0]['params']['active_stress']['alpha_min']
            
            c1 = t_contr + alpha_max/(K*(alpha_max-alpha_min))
            c2 = t_relax - alpha_max/(K*(alpha_max-alpha_min))
            
            # Diss Hirschvogel eq. 2.101
            return (K*(t-c1)+1.)*((K*(t-c1)+1.)>0.) - K*(t-c1)*((K*(t-c1))>0.) - K*(t-c2)*((K*(t-c2))>0.) + (K*(t-c2)-1.)*((K*(t-c2)-1.)>0.)

        def tc2(self, t): # active stress from file - precomputed from the ODE, should yield same results

            actstressinterp = np.loadtxt(basepath+'/input/actstress.txt', skiprows=1, usecols=(1))
            tme = np.loadtxt(basepath+'/input/actstress.txt', skiprows=1, usecols=(0))
            
            return np.interp(t, tme, actstressinterp)


    BC_DICT_ALE          = { 'dirichlet' : [{'id' : [5,11,17], 'dir' : 'all', 'val' : 0.}] } # bottom

    BC_DICT_FLUID        = { 'membrane' :  [{'id' : [2], 'params' : {'model' : 'membrane', 'a_0' : 1.0, 'b_0' : 6.0, 'eta' : 0.01, 'rho0' : 1e-6, 'h0' : {'val' : 1.0}, 'active_stress' : {'type' : 'ode', 'dir' : 'cl', 'sigma0' : 10., 'alpha_max' : 10.0, 'alpha_min' : -30.0, 'activation_curve' : 1, 'omega' : 0.667, 'iota' : 0.333, 'gamma' : 0.0, 'weight' : basepath+'/input/blocks3_part-1.txt'}}},
                                            {'id' : [8], 'params' : {'model' : 'membrane', 'a_0' : 1.0, 'b_0' : 6.0, 'eta' : 0.01, 'rho0' : 1e-6, 'h0' : {'val' : 1.0}, 'active_stress' : {'type' : 'prescribed', 'dir' : 'cl', 'prescribed_curve' : 2, 'omega' : 0.667, 'iota' : 0.333, 'gamma' : 0.0}}},
                                            {'id' : [14], 'params' : {'model' : 'membrane', 'a_0' : 1.0, 'b_0' : 6.0, 'eta' : 0.01, 'rho0' : 1e-6, 'h0' : {'val' : 1.0}, 'active_stress' : {'type' : 'ode', 'dir' : 'iso', 'sigma0' : 100., 'alpha_max' : 10.0, 'alpha_min' : -30.0, 'activation_curve' : 1}}}] }


    # problem setup
    problem = ambit_fe.ambit_main.Ambit(IO_PARAMS, TIME_PARAMS, SOLVER_PARAMS, [FEM_PARAMS_FLUID, FEM_PARAMS_ALE], [MATERIALS_FLUID, MATERIALS_ALE], [BC_DICT_FLUID, BC_DICT_ALE], time_curves=time_curves(), coupling_params=COUPLING_PARAMS, mor_params=ROM_PARAMS)

    # problem solve
    problem.solve_problem()


    # --- results check
    tol = 1.0e-6

    check_node = []
    check_node.append(np.array([1.0, 0.0, 0.0]))
    check_node.append(np.array([1.0, 0.0, 1.25]))
    check_node.append(np.array([1.0, 0.0, 2.5]))

    d_corr = np.zeros(3*len(check_node))

    # correct results
    d_corr[0] = -7.2739472331760840E-02 # x
    d_corr[1] = 0.0 # y
    d_corr[2] = 9.3307639402266801E-03 # z

    d_corr[3] = -7.2739472331760840E-02 # x
    d_corr[4] = 0.0 # y
    d_corr[5] = 9.3307639402266801E-03 # z
    
    d_corr[6] = -8.1979482461195838E-02 # x
    d_corr[7] = 0.0 # y
    d_corr[8] = 8.1979482461195810E-02 # z

    check1 = ambit_fe.resultcheck.results_check_node(problem.mp.pba.d, check_node, d_corr, problem.mp.pba.V_d, problem.mp.comm, tol=tol, nm='d', readtol=1e-4)

    success = ambit_fe.resultcheck.success_check([check1], problem.mp.comm)

    if not success:
        raise RuntimeError("Test failed!")



if __name__ == "__main__":

    test_main()
