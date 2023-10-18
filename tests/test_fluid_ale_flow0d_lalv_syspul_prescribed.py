#!/usr/bin/env python3

"""
LA-LV patient-specific prescribed ALE motion problem

tests:
- ALE-fluid coupled to 0D circulation model
- prescribed ALE motion via volume Dirichlet condition
- stress-symmetric version of reduced stabilization scheme in ALE form
- duplicate pressure nodes (discontinuity) at mitral valve plane
- dt scaling of residual
"""

import ambit_fe

import sys
import numpy as np
from pathlib import Path
import pytest


@pytest.mark.fluid_ale_flow0d
def test_main():

    basepath = str(Path(__file__).parent.absolute())

    # reads in restart step from the command line
    try: restart_step = int(sys.argv[1])
    except: restart_step = 0

    IO_PARAMS            = {'problem_type'          : 'fluid_ale_flow0d',
                            'USE_MIXED_DOLFINX_BRANCH' : True,
                            'duplicate_mesh_domains': [1,2],
                            'write_results_every'   : 1,
                            'write_restart_every'   : 1,
                            'indicate_results_by'   : 'step',
                            'restart_step'          : restart_step,
                            'output_path'           : basepath+'/tmp/',
                            'mesh_domain'           : basepath+'/input/lalv_domain.xdmf',
                            'mesh_boundary'         : basepath+'/input/lalv_boundary.xdmf',
                            'results_to_write'      : [['velocity','pressure'],['aledisplacement']], # first fluid, then ale results
                            'simname'               : 'fluid_ale_flow0d_lalv_syspul_prescribed'}

    SOLVER_PARAMS        = {'solve_type'            : 'direct', # direct, iterative
                            'tol_res'               : 1.0e-7,
                            'tol_inc'               : 1.0e-7,
                            'subsolver_params'      : {'tol_res' : 1.0e-6, 'tol_inc' : 1.0e-6},
                            'k_ptc_initial'         : 0.1,
                            'catch_max_res_value'   : 1e12}

    TIME_PARAMS          = {'maxtime'               : 1.0,
                            'numstep'               : 1000,
                            'numstep_stop'          : 2,
                            'timint'                : 'ost',
                            'theta_ost'             : 1.0,
                            'residual_scale'        : [0.001,0.001,0.001]}

    TIME_PARAMS_FLOW0D   = {'timint'                : 'ost',
                            'theta_ost'             : 0.5,
                            'initial_backwardeuler' : True,
                            'initial_conditions'    : init()}

    MODEL_PARAMS_FLOW0D  = {'modeltype'             : 'syspul',
                            'coronary_model'        : None,
                            'parameters'            : param(),
                            'chamber_models'        : {'lv' : {'type' : '3D_fluid', 'num_inflows':0, 'num_outflows':1},
                                                       'rv' : {'type' : '0D_elast', 'activation_curve' : 1},
                                                       'la' : {'type' : '3D_fluid', 'num_inflows':4, 'num_outflows':0},
                                                       'ra' : {'type' : '0D_elast', 'activation_curve' : 2}},
                            'valvelaws'             : {'av' : ['pwlin_time'], # time-controlled aortic valve
                                                       'mv' : ['pwlin_pres'], # overridden by prescibed q_vin_l - mitral valve is an immersed surface in 3D!
                                                       'pv' : ['pwlin_pres'],
                                                       'tv' : ['pwlin_pres']},
                            'prescribed_variables'  : {'q_vin_l' : {'flux_monitor' : 0}}}

    FEM_PARAMS_FLUID     = {'order_vel'             : 1,
                            'order_pres'            : 1,
                            'quad_degree'           : 5,
                            'stabilization'         : {'scheme' : 'supg_pspg2', 'vscale' : 1e3, 'dscales' : [1.,1.,1.], 'symmetric' : True} }

    FEM_PARAMS_ALE       = {'order_disp'            : 1,
                            'quad_degree'           : 5}

    COUPLING_PARAMS_ALE_FLUID = {'coupling_ale_fluid' : [{'surface_ids' : [1], 'type' : 'strong_dirichlet'}], # strong_dirichlet, weak_dirichlet
                                 'fluid_on_deformed'  : 'consistent'}

    COUPLING_PARAMS_FLUID_FLOW0D = {'surface_ids'   : [[5],[6],[7],[8], [4]],
                            'coupling_quantity'     : ['pressure']*5,
                            'variable_quantity'     : ['flux']*5,
                            'cq_factor'             : [1., -1.,-1.,-1.,-1.], # in-flows negative, out-flows positive! - lv first, then la
                            'coupling_type'         : 'monolithic_lagrange',
                            'print_subiter'         : True}

    MATERIALS_FLUID      = { 'MAT1' : {'newtonian' : {'mu' : 4.0e-6},
                                       'inertia' : {'rho' : 1.025e-6}},
                             'MAT2' : {'newtonian' : {'mu' : 4.0e-6},
                                       'inertia' : {'rho' : 1.025e-6}}}

    MATERIALS_ALE        = { 'MAT1' : {'linelast' : {'Emod' : 10.0, 'kappa' : 100.}},
                             'MAT2' : {'linelast' : {'Emod' : 10.0, 'kappa' : 100.}}}


    # define your load curves here (syntax: tcX refers to curve X, to be used in BC_DICT key 'curve' : [X,0,0], or 'curve' : X)
    # some examples... up to 20 possible (tc1 until tc20 - feel free to implement more in timeintegration.py --> timecurves function if needed...)
    class time_curves():

        def tc1(self, t): # RV

            return 0.5*(1.-np.cos(2.*np.pi*(t-0.37)/(0.8-0.37))) * (t >= 0.37) * (t <= 0.37 + 0.8-0.37)


        def tc2(self, t): # RA

            return 0.5*(1.-np.cos(2.*np.pi*(t-0.)/(0.2))) * (t >= 0.1) * (t <= 0.1 + 0.3)



    BC_DICT_ALE          = { 'dirichlet_vol' : [{'id' : [1,2], 'file' : basepath+'/input/aledisp_lalv_prescr-*.txt'}] }

    BC_DICT_FLUID        = { 'robin_valve' : [{'id' : [3], 'type' : 'temporal', 'beta_max' : 1e3, 'beta_min' : 0, 'to' : 0.0, 'tc' : 0.37}], # MV
                             'dp_monitor' : [{'id' : [3], 'upstream_domain' : 2, 'downstream_domain' : 1}], # MV
                             'flux_monitor' : [{'id' : [3], 'on_subdomain' : True, 'domain' : 2}],  # MV
                             'stabilized_neumann' : [{'id' : [5,6,7,8, 4], 'par1' : 0.205e-6, 'par2' : 1.}] } # par1 should be ~ 0.2*rho


    # problem setup
    problem = ambit_fe.ambit_main.Ambit(IO_PARAMS, [TIME_PARAMS, TIME_PARAMS_FLOW0D], SOLVER_PARAMS, [FEM_PARAMS_FLUID, FEM_PARAMS_ALE], [MATERIALS_FLUID, MATERIALS_ALE, MODEL_PARAMS_FLOW0D], [BC_DICT_FLUID, BC_DICT_ALE], time_curves=time_curves(), coupling_params=[COUPLING_PARAMS_ALE_FLUID,COUPLING_PARAMS_FLUID_FLOW0D])


    # problem solve
    problem.solve_problem()


    # --- results check
    tol = 1.0e-6

    s_corr = np.zeros(problem.mp.pb0.cardvasc0D.numdof)

    # correct 0D results
    s_corr[0] = 3.8882867807798226E+04
    s_corr[1] = 5.5563429211113005E+04
    s_corr[2] = -5.5564381126668784E+04
    s_corr[3] = -9.5191555577660569E-01
    s_corr[4] = 1.0924979624559237E+01
    s_corr[5] = -9.5191555577660569E-01
    s_corr[6] = 1.0924985336052572E+01
    s_corr[7] = 2.1047096618496435E+04
    s_corr[8] = 2.2871777634299062E+00
    s_corr[9] = 9.5236185798506907E+04
    s_corr[10] = 1.4906158530701109E+03
    s_corr[11] = 1.5093042657425306E-03
    s_corr[12] = -2.2265151472645608E-01
    s_corr[13] = 1.8688412672419695E-05
    s_corr[14] = 2.2265338356772331E+00
    s_corr[15] = 3.5951841419818549E+04
    s_corr[16] = 1.6872562143799550E+00
    s_corr[17] = 3.2933879944333407E+03
    s_corr[18] = 3.7056609678707014E+03
    s_corr[19] = 2.5876697333575962E+03
    s_corr[20] = 7.0938427076531352E+03


    check1 = ambit_fe.resultcheck.results_check_vec_sq(problem.mp.pb0.s, s_corr, problem.mp.comm, tol=tol)
    success = ambit_fe.resultcheck.success_check([check1], problem.mp.comm)

    if not success:
        raise RuntimeError("Test failed!")



def init():

            # initial conditions
    return {'q_vin_l_0' : 0.0,
            'Q_at_l_0' : 0.0,
            'q_vout_l_0' : 0.0,
            'Q_v_l_0' : 0.0,
            'p_ar_sys_0' : 1.0926945419777734E+01,
            'p_ard_sys_0' : 1.0926945419777734E+01,
            'q_arp_sys_0' : 0.0,
            'q_ar_sys_0' : 0.0,
            'p_ven_sys_0' : 2.2875736545217800E+00,
            'q_ven1_sys_0' : 0.0,
            'q_vin_r_0' : 0.0,
            'p_at_r_0' : 0.0,
            'q_vout_r_0' : 0.0,
            'p_v_r_0' : 0.0,
            'p_ar_pul_0' : 2.2301399591379436E+00,
            'q_ar_pul_0' : 0.0,
            'p_ven_pul_0' : 1.6864951426543255E+00,
            'q_ven1_pul_0' : 0.0,
            'q_ven2_pul_0' : 0.0,
            'q_ven3_pul_0' : 0.0,
            'q_ven4_pul_0' : 0.0,
            # now the LM values
            'p_at_l_i1' : 0.,
            'p_at_l_i2' : 0.,
            'p_at_l_i3' : 0.,
            'p_at_l_i4' : 0.,
            'p_v_l_o1' : 0.}


def param():

            #resistances (R), compliances (C), intertances (L), arterial characteristic impedances (Z)
    return {'R_ar_sys' : 120.0e-6,
            'C_ar_sys' : 1.377019e4,
            'Z_ar_sys' : 6.0e-6,
            'R_ven_sys' : 24.0e-6,
            'C_ven_sys' : 413100.0,
            'R_ar_pul' : 15.0e-6,
            'C_ar_pul' : 20000.0,
            'R_ven_pul' : 15.0e-6,
            'C_ven_pul' : 50000.0,
            'L_ar_sys' : 0.667e-6,
            'L_ven_sys' : 0.,
            'L_ar_pul' : 0.,
            'L_ven_pul' : 0.,
            # ventricular elastances (for 0D chambers)
            'E_v_max_l' : 7.0e-5,
            'E_v_min_l' : 12.0e-6,
            'E_v_max_r' : 3.0e-5,
            'E_v_min_r' : 10.0e-6,
            # atrial elastances (for 0D chambers)
            'E_at_min_l' : 9.0e-6,
            'E_at_max_l' : 29.0e-6,
            'E_at_min_r' : 8.0e-6,
            'E_at_max_r' : 18.0e-6,
            # valve resistances
            'R_vin_l_min' : 1.0e-6,
            'R_vin_l_max' : 1.0e1,
            'R_vout_l_min' : 1.0e-6,
            'R_vout_l_max' : 1.0e1,
            'R_vin_r_min' : 1.0e-6,
            'R_vin_r_max' : 1.0e1,
            'R_vout_r_min' : 1.0e-6,
            'R_vout_r_max' : 1.0e1,
            # timings
            't_ed' : 0.37,
            't_es' : 0.8,
            'T_cycl' : 0.8,
            # unstressed compartment volumes (for post-processing)
            'V_at_l_u' : 0.0,
            'V_at_r_u' : 0.0,
            'V_v_l_u' : 0.0,
            'V_v_r_u' : 0.0,
            'V_ar_sys_u' : 0.0,
            'V_ar_pul_u' : 0.0,
            'V_ven_sys_u' : 0.0,
            'V_ven_pul_u' : 0.0}



if __name__ == "__main__":

    test_main()
