#!/usr/bin/env python3

"""
"""

import ambit_fe

import sys
import numpy as np
from pathlib import Path
import pytest


@pytest.mark.flow0d
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
                         'numstep_stop'          : 100,
                         'timint'                : 'ost',
                         'theta_ost'             : 0.5,
                         'initial_conditions'    : init(),
                         'eps_periodic'          : 0.03,
                         'periodic_checktype'    : ['pQvar']}

    MODEL_PARAMS      = {'modeltype'             : 'syspul',
                         'coronary_model'        : 'ZCRp_CRd',
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
    s_corr[0] = 2.1891215121189365E+04
    s_corr[1] = 6.5539843855431468E-01
    s_corr[2] = 1.6805327962454747E-01
    s_corr[3] = 6.3350722343312516E-01
    s_corr[4] = 8.0368039078659965E+00
    s_corr[5] = -1.4415864662615999E+02
    s_corr[6] = 8.0376688597457395E+00
    s_corr[7] = 4.9889324331161697E+04
    s_corr[8] = 2.0830544812886083E+00
    s_corr[9] = -1.4582294134657655E+04
    s_corr[10] = 3.4971689677010771E+04
    s_corr[11] = 3.9319680680790853E-01
    s_corr[12] = 1.8896764674462069E-01
    s_corr[13] = 3.5822511713089783E-01
    s_corr[14] = 1.6031497654365026E+00
    s_corr[15] = -9.0263340297088216E+04
    s_corr[16] = 1.3290418867628540E+00
    s_corr[17] = -2.1751637234497350E+04
    s_corr[18] = 1.4432669990573908E+02
    s_corr[19] = -1.3235004521686183E+03
    s_corr[20] = 6.5601060482121811E+00
    s_corr[21] = 4.3174033945046787E+01

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
            # coronary circulation submodel
            'q_cor_sys_0' : 0,
            'q_cord_sys_0' : 0,
            'q_corp_sys_0' : 0.,
            'p_cord_sys_0' : 0.}


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
            # coronary circulation submodel parameters - values from Arthurs et al. 2016, Tab. 3
            'Z_corp_sys' : 3.2e-3,
            'C_corp_sys' : 4.5e0,
            'R_corp_sys' : 6.55e-3,
            'C_cord_sys' : 2.7e1,
            'R_cord_sys' : 1.45e-1}


if __name__ == "__main__":

    test_main()
