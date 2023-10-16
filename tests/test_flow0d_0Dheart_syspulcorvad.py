#!/usr/bin/env python3

"""
ATTENTION: 0D VAD model not yet fully functional!
"""

import ambit_fe

import sys
import numpy as np
from pathlib import Path
import pytest


@pytest.mark.flow0d
@pytest.mark.skip(reason="Not yet ready for testing.")
def test_main():

    basepath = str(Path(__file__).parent.absolute())

    # reads in restart step from the command line
    try: restart_step = int(sys.argv[1])
    except: restart_step = 0

    IO_PARAMS         = {'problem_type'          : 'flow0d',
                         'write_results_every'   : -999,
                         'output_path'           : basepath+'/tmp',
                         'simname'               : 'test',
                         'write_restart_every'   : 50,
                         'restart_step'          : restart_step}

    SOLVER_PARAMS     = {'tol_res'               : 1.0e-8,
                         'tol_inc'               : 1.0e-8}

    TIME_PARAMS       = {'maxtime'               : 10*1.0,
                         'numstep'               : 10*100,
                         'timint'                : 'ost',
                         'theta_ost'             : 0.5,
                         'initial_conditions'    : init(),
                         'eps_periodic'          : 0.03,
                         'periodic_checktype'    : ['pQvar']}

    MODEL_PARAMS      = {'modeltype'             : 'syspul',
                         'coronary_model'        : 'ZCRp_CRd_lr',
                         'vad_model'             : 'lvad',
                         'parameters'            : param(),
                         'chamber_models'        : {'lv' : {'type' : '0D_elast', 'activation_curve' : 2},
                                                    'rv' : {'type' : '0D_elast', 'activation_curve' : 2},
                                                    'la' : {'type' : '0D_elast', 'activation_curve' : 1},
                                                    'ra' : {'type' : '0D_elast', 'activation_curve' : 1}},
                         'valvelaws'             : {'av' : ['smooth_pres_momentum',0],
                                                    'mv' : ['pwlin_pres'],
                                                    'pv' : ['pwlin_pres'],
                                                    'tv' : ['pwlin_pres']}}


    # define your time curves here (syntax: tcX refers to curve X)
    class time_curves():

        def tc1(self, t): # atrial activation

            act_dur = 2.*param()['t_ed']
            t0 = 0.

            if t >= t0 and t <= t0 + act_dur:
                return 0.5*(1.-np.cos(2.*np.pi*(t-t0)/act_dur))
            else:
                return 0.0

        def tc2(self, t): # ventricular activation

            act_dur = 1.8*(param()['t_es'] - param()['t_ed'])
            t0 = param()['t_ed']

            if t >= t0 and t <= t0 + act_dur:
                return 0.5*(1.-np.cos(2.*np.pi*(t-t0)/act_dur))
            else:
                return 0.0


    # problem setup
    problem = ambit_fe.ambit_main.Ambit(IO_PARAMS, TIME_PARAMS, SOLVER_PARAMS, constitutive_params=MODEL_PARAMS, time_curves=time_curves())

    # solve time-dependent problem
    problem.solve_problem()


    # --- results check
    tol = 1.0e-6

    s_corr = np.zeros(problem.mp.cardvasc0D.numdof)

    # correct results
    s_corr[0] = 2.7156831671111686E+04
    s_corr[1] = 6.5014832315152615E-01
    s_corr[2] = 2.3863884814303105E-01
    s_corr[3] = 6.2299149148041377E-01
    s_corr[4] = 7.3204325568836603E+00
    s_corr[5] = 2.3863885498607551E-01
    s_corr[6] = 7.3204311250505567E+00
    s_corr[7] = 4.4855244723865435E+04
    s_corr[8] = 1.9666233818994407E+00
    s_corr[9] = -2.3737219188883661E+04
    s_corr[10] = 3.2279121521802968E+04
    s_corr[11] = 4.9648390872022957E-01
    s_corr[12] = 1.6997499381247427E-01
    s_corr[13] = 4.6420478719842723E-01
    s_corr[14] = 1.8990559860434069E+00
    s_corr[15] = -8.1705347914691476E+04
    s_corr[16] = 1.4965782216339247E+00
    s_corr[17] = -1.0232540549572852E+04

    check1 = ambit_fe.resultcheck.results_check_vec_sq(problem.mp.s, s_corr, problem.mp.comm, tol=tol)
    success = ambit_fe.resultcheck.success_check([check1], problem.mp.comm)

    if not success:
        raise RuntimeError("Test failed!")



def init():

    return {'q_vin_l_0' : 0.0,
            'p_at_l_0' : 0.599950804034,
            'q_vout_l_0' : 0.0,
            'p_v_l_0' : 0.599950804034,
            'p_ar_sys_0' : 9.68378038166,
            'q_ar_sys_0' : 0.0,
            'p_ven_sys_0' : 2.13315841434,
            'q_ven_sys_0' : 0.0,
            'q_vin_r_0' : 0.0,
            'p_at_r_0' : 0.0933256806275,
            'q_vout_r_0' : 0.0,
            'p_v_r_0' : 0.0933256806275,
            'p_ar_pul_0' : 3.22792679389,
            'q_ar_pul_0' : 0.0,
            'p_ven_pul_0' : 1.59986881076,
            'q_ven_pul_0' : 0.0,
            # coronary circulation submodel - left and right
            'q_cor_sys_l_0' : 0,
            'q_cord_sys_l_0' : 0,
            'q_corp_sys_l_0' : 0.,
            'p_cord_sys_l_0' : 0.,
            'q_cor_sys_l_0' : 0,
            'q_cord_sys_l_0' : 0,
            'q_corp_sys_l_0' : 0.,
            'p_cord_sys_l_0' : 0.,
            'q_cor_sys_r_0' : 0,
            'q_cord_sys_r_0' : 0,
            'q_corp_sys_r_0' : 0.,
            'p_cord_sys_r_0' : 0.,
            'q_cor_sys_r_0' : 0,
            'q_cord_sys_r_0' : 0,
            'q_corp_sys_r_0' : 0.,
            'p_cord_sys_r_0' : 0.}


def param():

    # parameters in kg-mm-s unit system

    R_ar_sys = 120.0e-6
    tau_ar_sys = 1.0311433159
    tau_ar_pul = 0.3

    # Diss Hirschvogel tab. 2.7
    C_ar_sys = tau_ar_sys/R_ar_sys
    Z_ar_sys = R_ar_sys/20.
    R_ven_sys = R_ar_sys/5.
    C_ven_sys = 30.*C_ar_sys
    R_ar_pul = R_ar_sys/8.
    C_ar_pul = tau_ar_pul/R_ar_pul
    R_ven_pul = R_ar_pul
    C_ven_pul = 2.5*C_ar_pul

    L_ar_sys = 0.667e-6
    L_ven_sys = 0.
    L_ar_pul = 0.
    L_ven_pul = 0.

    # timings
    t_ed = 0.2
    t_es = 0.53
    T_cycl = 1.0

    # atrial elastances
    E_at_max_l = 2.9e-5
    E_at_min_l = 9.0e-6
    E_at_max_r = 1.8e-5
    E_at_min_r = 8.0e-6
    # ventricular elastances
    E_v_max_l = 30.0e-5
    E_v_min_l = 12.0e-6
    E_v_max_r = 20.0e-5
    E_v_min_r = 10.0e-6


    return {'R_ar_sys' : R_ar_sys,
            'C_ar_sys' : C_ar_sys,
            'L_ar_sys' : L_ar_sys,
            'Z_ar_sys' : Z_ar_sys,
            'R_ar_pul' : R_ar_pul,
            'C_ar_pul' : C_ar_pul,
            'L_ar_pul' : L_ar_pul,
            'R_ven_sys' : R_ven_sys,
            'C_ven_sys' : C_ven_sys,
            'L_ven_sys' : L_ven_sys,
            'R_ven_pul' : R_ven_pul,
            'C_ven_pul' : C_ven_pul,
            'L_ven_pul' : L_ven_pul,
            # atrial elastances
            'E_at_max_l' : E_at_max_l,
            'E_at_min_l' : E_at_min_l,
            'E_at_max_r' : E_at_max_r,
            'E_at_min_r' : E_at_min_r,
            # ventricular elastances
            'E_v_max_l' : E_v_max_l,
            'E_v_min_l' : E_v_min_l,
            'E_v_max_r' : E_v_max_r,
            'E_v_min_r' : E_v_min_r,
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
            't_ed' : t_ed,
            't_es' : t_es,
            'T_cycl' : T_cycl,
            # unstressed compartment volumes (for post-processing)
            'V_at_l_u' : 0.0,
            'V_at_r_u' : 0.0,
            'V_v_l_u' : 0.0,
            'V_v_r_u' : 0.0,
            'V_ar_sys_u' : 0.0,
            'V_ar_pul_u' : 0.0,
            'V_ven_sys_u' : 0.0,
            'V_ven_pul_u' : 0.0,
            # coronary circulation submodel parameters - values from Arthurs et al. 2016, Tab. 3
            # left
            'Z_corp_sys_l' : 3.2e-3,
            'C_corp_sys_l' : 4.5e0,
            'R_corp_sys_l' : 6.55e-3,
            'C_cord_sys_l' : 2.7e1,
            'R_cord_sys_l' : 1.45e-1,
            # right
            'Z_corp_sys_r' : 3.2e-3,
            'C_corp_sys_r' : 4.5e0,
            'R_corp_sys_r' : 6.55e-3,
            'C_cord_sys_r' : 2.7e1,
            'R_cord_sys_r' : 1.45e-1}




if __name__ == "__main__":

    test_main()
