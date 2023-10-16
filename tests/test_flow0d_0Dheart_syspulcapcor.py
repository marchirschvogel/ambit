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

    IO_PARAMS         = {'problem_type'          : 'flow0d',
                         'write_results_every'   : -999,
                         'output_path'           : basepath+'/tmp',
                         'simname'               : 'test'}

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

    MODEL_PARAMS      = {'modeltype'             : 'syspulcapcor',
                         'coronary_model'        : 'ZCRp_CRd',
                         'parameters'            : param(),
                         'chamber_models'        : {'lv' : {'type' : '0D_elast', 'activation_curve' : 2},
                                                    'rv' : {'type' : '0D_elast', 'activation_curve' : 2},
                                                    'la' : {'type' : '0D_elast', 'activation_curve' : 1},
                                                    'ra' : {'type' : '0D_elast', 'activation_curve' : 1}}}


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
    s_corr[0] = 2.9720219873885115E+04
    s_corr[1] = 7.0505615946727873E-01
    s_corr[2] = 4.3319003546716228E-01
    s_corr[3] = 6.7533593959339200E-01
    s_corr[4] = 7.7815769888820085E+00
    s_corr[5] = 4.1985489612549129E+04
    s_corr[6] = 7.7975415697868531E+00
    s_corr[7] = 4.3475704330285233E+04
    s_corr[8] = 7.5128467188033508E+00
    s_corr[9] = -1.1550492652351211E+04
    s_corr[10] = -1.0778246834175183E+04
    s_corr[11] = -8.4477103162224957E+03
    s_corr[12] = -5.7742397212347441E+03
    s_corr[13] = 2.1059354264141033E+00
    s_corr[14] = 2.0275402266303041E+04
    s_corr[15] = 2.0694342876485248E+00
    s_corr[16] = 1.7374493993183587E+04
    s_corr[17] = 2.0674090884041485E+00
    s_corr[18] = 1.3626408788968787E+04
    s_corr[19] = 2.0726079626353830E+00
    s_corr[20] = 9.3082801929229390E+03
    s_corr[21] = 1.7750408614280289E+00
    s_corr[22] = -4.8649155298043741E+04
    s_corr[23] = 3.6789867002006940E+04
    s_corr[24] = 5.6261398871729074E-01
    s_corr[25] = 9.1289396906923420E-02
    s_corr[26] = 5.2582412171528459E-01
    s_corr[27] = 1.9005584023400093E+00
    s_corr[28] = 1.2920878029780110E+04
    s_corr[29] = 1.8036518171166580E+00
    s_corr[30] = -8.3889295048798944E+04
    s_corr[31] = 1.6250718865201932E+00
    s_corr[32] = -5.3268186447011039E+03
    s_corr[33] = 6.7433006838558038E+02
    s_corr[34] = -1.5307703341615488E+03
    s_corr[35] = 1.1619577578510618E+00
    s_corr[36] = -5.5966144109129673E+04


    check1 = ambit_fe.resultcheck.results_check_vec_sq(problem.mp.s, s_corr, problem.mp.comm, tol=tol)
    success = ambit_fe.resultcheck.success_check([check1], problem.mp.comm)

    if not success:
        raise RuntimeError("Test failed!")



def init():

    return {'q_vin_l_0' : 2.9122879355134799E+04,
            'p_at_l_0' : 6.8885657594702698E-01,
            'q_vout_l_0' : 4.4126414250284074E-01,
            'p_v_l_0' : 6.5973369659189085E-01,
            'p_ar_sys_0' : 7.6852336907652035E+00,
            'q_ar_sys_0' : 4.4646253096693101E+04,

            'p_arperi_sys_0' : 7.3924415864114579E+00,
            'q_arspl_sys_0' : -1.1942736655945399E+04,
            'q_arespl_sys_0' : -1.1129301639510835E+04,
            'q_armsc_sys_0' : -8.7229603630348920E+03,
            'q_arcer_sys_0' : -5.9623606948858287E+03,
            'p_venspl_sys_0' : 2.1337514581004355E+00,
            'q_venspl_sys_0' : 2.0406124240978173E+04,
            'p_venespl_sys_0' : 2.0900015313258282E+00,
            'q_venespl_sys_0' : 1.7072593297814688E+04,
            'p_venmsc_sys_0' : 2.0879628079853534E+00,
            'q_venmsc_sys_0' : 1.3387364723046561E+04,
            'p_vencer_sys_0' : 2.0933161349988683E+00,
            'q_vencer_sys_0' : 9.1526721881635949E+03,
            'q_corp_sys_0' : -1.9864253416510032E+03,
            'p_cord_sys_0' : 2.0910022623881237E+00,
            'q_cord_sys_0' : 3.0343572493359602E+03,

            'p_ven_sys_0' : 1.8007235104876642E+00,
            'q_ven_sys_0' : -4.5989218100751634E+04,
            'q_vin_r_0' : 3.4747706215569546E+04,
            'p_at_r_0' : 5.3722584358891634E-01,
            'q_vout_r_0' : 9.2788006831497391E-02,
            'p_v_r_0' : 5.0247813737334734E-01,
            'p_ar_pul_0' : 1.8622263176170106E+00,
            'q_ar_pul_0' : 1.2706171472263239E+04,
            'p_cap_pul_0' : 1.7669300315750378E+00,
            'q_cap_pul_0' : -8.4296230468394206E+04,
            'p_ven_pul_0' : 1.5914021166255159E+00,
            'q_ven_pul_0' : -6.4914977363299859E+03}


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


    ## systemic arterial
    # now we have to separate the resistance into a proximal and a peripheral part
    frac_Rprox_Rtotal = 0.06 # Ursino et al. factor: 0.06 - OK
    R_arperi_sys = (1.-frac_Rprox_Rtotal)*R_ar_sys
    R_ar_sys *= frac_Rprox_Rtotal # now R_ar_sys(prox)

    frac_Cprox_Ctotal = 0.95#0.07 # Ursino et al. factor: 0.07 - XXXX too small???!!!!! - keep in mind that most compliance lies in the aorta / proximal!
    C_arperi_sys = (1.-frac_Cprox_Ctotal)*C_ar_sys
    C_ar_sys *= frac_Cprox_Ctotal # now C_ar_sys(prox)

    # R in parallel:
    # R_arperi_sys = (1/R_arspl_sys + 1/R_arespl_sys + 1/R_armsc_sys + 1/R_arcer_sys + 1/R_arcor_sys)^(-1)
    R_arspl_sys = 3.35 * R_arperi_sys # Ursino et al. factor: 3.35 - OK
    R_arespl_sys = 3.56 * R_arperi_sys # Ursino et al. factor: 3.56 - OK
    R_armsc_sys = 4.54 * R_arperi_sys # Ursino et al. factor: 4.54 - OK
    R_arcer_sys = 6.65 * R_arperi_sys # Ursino et al. factor: 6.65 - OK
    R_arcor_sys = 19.95 * R_arperi_sys # Ursino et al. factor: 19.95 - OK

    # C in parallel (fractions have to sum to 1):
    # C_arperi_sys = C_arspl_sys + C_arespl_sys + C_armsc_sys + C_arcer_sys + C_arcor_sys
    C_arspl_sys = 0.55 * C_arperi_sys # Ursino et al. factor: 0.55 - OK
    C_arespl_sys = 0.18 * C_arperi_sys # Ursino et al. factor: 0.18 - OK
    C_armsc_sys = 0.14 * C_arperi_sys # Ursino et al. factor: 0.14 - OK
    C_arcer_sys = 0.11 * C_arperi_sys # Ursino et al. factor: 0.11 - OK
    C_arcor_sys = 0.03 * C_arperi_sys # Ursino et al. factor: 0.03 - OK

    ## systemic venous
    frac_Rprox_Rtotal = 0.8 # no Ursino et al. factor since they do not have that extra compartment!
    R_venperi_sys = (1.-frac_Rprox_Rtotal) * R_ven_sys
    R_ven_sys *= frac_Rprox_Rtotal # now R_ven_sys(prox)

    frac_Cprox_Ctotal = 0.2 # no Ursino et al. factor since they do not have that extra compartment!
    C_venperi_sys = (1.-frac_Cprox_Ctotal)*C_ven_sys
    C_ven_sys *= frac_Cprox_Ctotal # now C_ven_sys(prox)

    # R in parallel:
    # R_venperi_sys = (1/R_venspl_sys + 1/R_venespl_sys + 1/R_venmsc_sys + 1/R_vencer_sys + 1/R_vencor_sys)^(-1)
    R_venspl_sys = 3.4 * R_venperi_sys # Ursino et al. factor: 3.4 - OK
    R_venespl_sys = 3.53 * R_venperi_sys # Ursino et al. factor: 3.53 - OK
    R_venmsc_sys = 4.47 * R_venperi_sys # Ursino et al. factor: 4.47 - OK
    R_vencer_sys = 6.66 * R_venperi_sys # Ursino et al. factor: 6.66 - OK
    R_vencor_sys = 19.93 * R_venperi_sys # Ursino et al. factor: 19.93 - OK

    # C in parallel (fractions have to sum to 1):
    # C_venperi_sys = C_venspl_sys + C_venespl_sys + C_venmsc_sys + C_vencer_sys + C_vencor_sys
    C_venspl_sys = 0.55 * C_venperi_sys # Ursino et al. factor: 0.55 - OK
    C_venespl_sys = 0.18 * C_venperi_sys # Ursino et al. factor: 0.18 - OK
    C_venmsc_sys = 0.14 * C_venperi_sys # Ursino et al. factor: 0.14 - OK
    C_vencer_sys = 0.1 * C_venperi_sys # Ursino et al. factor: 0.1 - OK
    C_vencor_sys = 0.03 * C_venperi_sys # Ursino et al. factor: 0.03 - OK

    ## pulmonary arterial
    frac_Rprox_Rtotal = 0.5#0.72  # Ursino et al. factor: 0.72 - hm... doubt that - stick with 0.5
    R_cap_pul = (1.-frac_Rprox_Rtotal)*R_ar_pul
    R_ar_pul *= frac_Rprox_Rtotal # now R_ar_pul(prox)

    ## pulmonary venous
    frac_Cprox_Ctotal = 0.5#0.12  # Ursino et al. factor: 0.12 - XXX?: gives shitty p_puls... - stick with 0.5
    C_cap_pul = (1.-frac_Cprox_Ctotal)*C_ar_pul
    C_ar_pul *= frac_Cprox_Ctotal # now C_ar_pul(prox)

    ### unstressed compartment volumes, diffult to estimate - use literature values!
    # these volumes only become relevant for the gas transport models as they determine the capacity of each
    # compartment to store constituents - however, they are also used for postprocessing of the flow models...
    V_at_l_u = 5000.0 # applies only in case of 0D or prescribed atria
    V_at_r_u = 4000.0 # applies only in case of 0D or prescribed atria
    V_v_l_u = 10000.0 # applies only in case of 0D or prescribed ventricles
    V_v_r_u = 8000.0 # applies only in case of 0D or prescribed ventricles
    V_ar_sys_u = 0.0 # Ursino et al. Am J Physiol Heart Circ Physiol (2000), mm^3
    V_ar_pul_u = 0.0 # Ursino et al. Am J Physiol Heart Circ Physiol (2000), mm^3
    V_ven_pul_u = 120.0e3 # Ursino et al. Am J Physiol Heart Circ Physiol (2000), mm^3
    # peripheral systemic arterial
    V_arspl_sys_u = 274.4e3 # Ursino et al. Am J Physiol Heart Circ Physiol (2000), mm^3
    V_arespl_sys_u = 134.64e3 # Ursino et al. Am J Physiol Heart Circ Physiol (2000), mm^3
    V_armsc_sys_u = 105.8e3 # Ursino et al. Am J Physiol Heart Circ Physiol (2000), mm^3
    V_arcer_sys_u = 72.13e3 # Ursino et al. Am J Physiol Heart Circ Physiol (2000), mm^3
    V_arcor_sys_u = 24.0e3 # Ursino et al. Am J Physiol Heart Circ Physiol (2000), mm^3
    # peripheral systemic venous
    V_venspl_sys_u = 1121.0e3 # Ursino et al. Am J Physiol Heart Circ Physiol (2000), mm^3
    V_venespl_sys_u = 550.0e3 # Ursino et al. Am J Physiol Heart Circ Physiol (2000), mm^3
    V_venmsc_sys_u = 432.14e3 # Ursino et al. Am J Physiol Heart Circ Physiol (2000), mm^3
    V_vencer_sys_u = 294.64e3 # Ursino et al. Am J Physiol Heart Circ Physiol (2000), mm^3
    V_vencor_sys_u = 98.21e3 # Ursino et al. Am J Physiol Heart Circ Physiol (2000), mm^3
    V_ven_sys_u = 100.0e3 # estimated (Ursino et al. do not have that extra venous compartment...)
    # pulmonary capillary
    V_cap_pul_u = 123.0e3 # Ursino et al. Am J Physiol Heart Circ Physiol (2000), mm^3


    return {'R_ar_sys' : R_ar_sys,
            'C_ar_sys' : C_ar_sys,
            'L_ar_sys' : L_ar_sys,
            'Z_ar_sys' : Z_ar_sys,
            'R_arspl_sys' : R_arspl_sys,
            'C_arspl_sys' : C_arspl_sys,
            'R_arespl_sys' : R_arespl_sys,
            'C_arespl_sys' : C_arespl_sys,
            'R_armsc_sys' : R_armsc_sys,
            'C_armsc_sys' : C_armsc_sys,
            'R_arcer_sys' : R_arcer_sys,
            'C_arcer_sys' : C_arcer_sys,
            'Z_corp_sys' : 0,
            'R_corp_sys' : R_arcor_sys,
            'C_corp_sys' : C_arcor_sys,
            'R_venspl_sys' : R_venspl_sys,
            'C_venspl_sys' : C_venspl_sys,
            'R_venespl_sys' : R_venespl_sys,
            'C_venespl_sys' : C_venespl_sys,
            'R_venmsc_sys' : R_venmsc_sys,
            'C_venmsc_sys' : C_venmsc_sys,
            'R_vencer_sys' : R_vencer_sys,
            'C_vencer_sys' : C_vencer_sys,
            'R_cord_sys' : R_vencor_sys,
            'C_cord_sys' : C_vencor_sys,
            'R_ar_pul' : R_ar_pul,
            'C_ar_pul' : C_ar_pul,
            'L_ar_pul' : L_ar_pul,
            'R_cap_pul' : R_cap_pul,
            'C_cap_pul' : C_cap_pul,
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
            'V_at_l_u' : V_at_l_u,
            'V_at_r_u' : V_at_r_u,
            'V_v_l_u' : V_v_l_u,
            'V_v_r_u' : V_v_r_u,
            'V_ar_sys_u' : V_ar_sys_u,
            'V_arspl_sys_u' : V_arspl_sys_u,
            'V_arespl_sys_u' : V_arespl_sys_u,
            'V_armsc_sys_u' : V_armsc_sys_u,
            'V_arcer_sys_u' : V_arcer_sys_u,
            'V_arcor_sys_u' : V_arcor_sys_u,
            'V_venspl_sys_u' : V_venspl_sys_u,
            'V_venespl_sys_u' : V_venespl_sys_u,
            'V_venmsc_sys_u' : V_venmsc_sys_u,
            'V_vencer_sys_u' : V_vencer_sys_u,
            'V_vencor_sys_u' : V_vencor_sys_u,
            'V_ven_sys_u' : V_ven_sys_u,
            'V_ar_pul_u' : V_ar_pul_u,
            'V_cap_pul_u' : V_cap_pul_u,
            'V_ven_pul_u' : V_ven_pul_u}




if __name__ == "__main__":

    test_main()
