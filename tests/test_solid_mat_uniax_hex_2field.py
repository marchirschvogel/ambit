#!/usr/bin/env python3

"""
solid mechanics
perfectly incompressible materials tested on uniaxial stress state, using p2p1 interpolation, hexahedral elements, one load step:
blocks of L_x = L_y = L_z = 1.0
materials:
- Neo-Hookean
- Mooney Rivlin
- Holzapfel-Ogden
- write of standard solid mechanics output
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
                         'mesh_domain'           : basepath+'/input/blocks3hex_domain.xdmf',
                         'mesh_boundary'         : basepath+'/input/blocks3hex_boundary.xdmf',
                         'fiber_data'            : [np.array([1.0,0.0,0.0]),np.array([0.0,1.0,0.0])],
                         'write_results_every'   : 1,
                         'output_path'           : basepath+'/tmp/',
                         'results_to_write'      : ['displacement','cauchystress','cauchystress_principal','vonmises_cauchystress','pk1stress','pk2stress','glstrain','glstrain_principal','eastrain','eastrain_principal','jacobian','fibers','strainenergy','internalpower'],
                         'simname'               : 'solid_mat_uniax_hex_2field'}

    SOLVER_PARAMS     = {'solve_type'            : 'direct',
                         'tol_res'               : 1.0e-8,
                         'tol_inc'               : 1.0e-8,
                         'maxiter'               : 25,
                         'divergence_continue'   : None}

    TIME_PARAMS       = {'maxtime'               : 1.0,
                         'numstep'               : 1,
                         'timint'                : 'static'}

    FEM_PARAMS        = {'order_disp'            : 2, # hex27 elements
                         'order_pres'            : 1, # hex8 elements
                         'quad_degree'           : 6, # should yield 27 Gauss points
                         'incompressible_2field' : True} # True, False

    MATERIALS         = {'MAT1' : {'neohooke_dev'       : {'mu' : 10.}},
                         'MAT2' : {'mooneyrivlin_dev'   : {'c1' : 2.5, 'c2' : 2.5}},
                         'MAT3' : {'holzapfelogden_dev' : {'a_0' : 0.059, 'b_0' : 8.023, 'a_f' : 18.472, 'b_f' : 16.026, 'a_s' : 2.481, 'b_s' : 11.120, 'a_fs' : 0.216, 'b_fs' : 11.436, 'fiber_comp_switch' : 'hard'}}}



    # analytical incompressible P_11 solutions for stretch in 1-direction (= x-direction):

    # constraint is lam_1 * lam_2 * lam_3 = 1
    # strain in 1-direction: lam := lam_1 ---> lam_2 = lam_3 = lam_q ---> lam_q = 1 / sqrt(lam)

    # I1 = lam_1^2 + lam_2^2 + lam_3^2 = lam^2 + 2/lam
    # I2 = lam_1^2 * lam_2^2 + lam_1^2 * lam_3^2 + lam_2^2 * lam_3^2 = 2 lam + 1/lam^2

    # NeoHooke
    def P_nh(lam):
        mu = MATERIALS['MAT1']['neohooke_dev']['mu']
        return mu*(lam-1./lam**2.)

    # Mooney Rivlin
    def P_mr(lam):
        c1, c2 = MATERIALS['MAT2']['mooneyrivlin_dev']['c1'], MATERIALS['MAT2']['mooneyrivlin_dev']['c2']
        return 2.*c1*(lam-1./(lam**2.)) + 2.*c2*(1.-1./(lam**3.))

    # Holzapfel-Ogden with fiber f0 in 1-direction, fiber s0 in 2-direction (I8 term cancels!)
    # no fiber compression, so cross-strains are equal (wouldn't be if s0 would constribute in compression!)
    def P_ho(lam):
        a_0, b_0 = MATERIALS['MAT3']['holzapfelogden_dev']['a_0'], MATERIALS['MAT3']['holzapfelogden_dev']['b_0']
        a_f, b_f = MATERIALS['MAT3']['holzapfelogden_dev']['a_f'], MATERIALS['MAT3']['holzapfelogden_dev']['b_f']
        a_s, b_s = MATERIALS['MAT3']['holzapfelogden_dev']['a_s'], MATERIALS['MAT3']['holzapfelogden_dev']['b_s']
        return a_0*(lam-1./lam**2.)*np.exp(b_0*(lam**2. + 2./lam - 3.)) + \
               a_f * 2.*lam*(lam**2.-1)*np.exp(b_f*(lam**2.-1.)**2.)


    # define your load curves here (syntax: tcX refers to curve X, to be used in BC_DICT key 'curve' : [X,0,0], or 'curve' : X)
    class time_curves():

        def tc1(self, t):
            umax = 1.0
            return umax*t/TIME_PARAMS['maxtime']

        def tc2(self, t):
            umax = 1.0
            return umax*t/TIME_PARAMS['maxtime']

        def tc3(self, t):
            umax = 0.1
            return umax*t/TIME_PARAMS['maxtime']

        # PK1 stress that yields to a x-displacement of 1.0 for NH material
        def tc4(self, t):
            tmax = 17.5
            return tmax*t/TIME_PARAMS['maxtime']

        # PK1 stress that yields to a x-displacement of 1.0 for MR material
        def tc5(self, t):
            tmax = 13.125
            return tmax*t/TIME_PARAMS['maxtime']

        # PK1 stress that yields to a x-displacement of 0.1 for HO material
        def tc6(self, t):
            tmax = 17.32206451195601
            return tmax*t/TIME_PARAMS['maxtime']



    BC_DICT           = { 'dirichlet' : [{'id' : [1], 'dir' : 'x', 'val' : 0.},
                                         {'id' : [2], 'dir' : 'y', 'val' : 0.},
                                         {'id' : [3], 'dir' : 'z', 'val' : 0.},
                                         #{'id' : [4], 'dir' : 'x', 'curve' : [1]},
                                         {'id' : [7], 'dir' : 'x', 'val' : 0.},
                                         {'id' : [8], 'dir' : 'y', 'val' : 0.},
                                         {'id' : [9], 'dir' : 'z', 'val' : 0.},
                                         #{'id' : [10], 'dir' : 'x', 'curve' : [2]},
                                         {'id' : [13], 'dir' : 'x', 'val' : 0.},
                                         {'id' : [14], 'dir' : 'y', 'val' : 0.},
                                         {'id' : [15], 'dir' : 'z', 'val' : 0.}],
                                         #{'id' : [16], 'dir' : 'x', 'curve' : [3]}]}
                            'neumann' : [{'id' : [4], 'dir' : 'xyz_ref', 'curve' : [4,0,0]},
                                         {'id' : [10], 'dir' : 'xyz_ref', 'curve' : [5,0,0]},
                                         {'id' : [16], 'dir' : 'xyz_ref', 'curve' : [6,0,0]}] }


    # problem setup
    problem = ambit_fe.ambit_main.Ambit(IO_PARAMS, TIME_PARAMS, SOLVER_PARAMS, FEM_PARAMS, MATERIALS, BC_DICT, time_curves=time_curves())

    # solve time-dependent problem
    problem.solve_problem()


    # --- results check
    tol = 1.0e-6

    check_node = []
    check_node.append(np.array([1.0, 1.0, 1.0]))
    check_node.append(np.array([1.0, 3.0, 1.0]))
    check_node.append(np.array([1.0, 5.0, 1.0]))

    u_corr = np.zeros(3*len(check_node))

    ## correct results
    u_corr[0] = 1.0 # x
    u_corr[1] = -2.9289321881345320E-01 # y
    u_corr[2] = -2.9289321881345320E-01 # z

    u_corr[3] = 1.0 # x
    u_corr[4] = -2.9289321881345320E-01 # y
    u_corr[5] = -2.9289321881345320E-01 # z

    u_corr[6] = 0.1 # x
    u_corr[7] = -4.6537410754407753E-02 # y
    u_corr[8] = -4.6537410754407753E-02 # z

    check1 = ambit_fe.resultcheck.results_check_node(problem.mp.u, check_node, u_corr, problem.mp.V_u, problem.mp.comm, tol=tol, nm='u')
    success = ambit_fe.resultcheck.success_check([check1], problem.mp.comm)

    if not success:
        raise RuntimeError("Test failed!")



if __name__ == "__main__":

    test_main()
