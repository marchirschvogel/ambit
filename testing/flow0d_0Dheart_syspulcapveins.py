#!/usr/bin/env python3

import ambit

import sys, traceback
import numpy as np
from pathlib import Path

import results_check


def main():
    
    basepath = str(Path(__file__).parent.absolute())

    IO_PARAMS         = {'problem_type'          : 'flow0d', # solid, fluid, flow0d, solid_flow0d, fluid_flow0d
                         'write_results_every'   : -999,
                         'output_path'           : ''+basepath+'/tmp/',
                         'simname'               : 'test'}

    SOLVER_PARAMS     = {'tol_res'               : 1.0e-8,
                         'tol_inc'               : 1.0e-8}

    TIME_PARAMS       = {'maxtime'               : 0.9,
                         'numstep'               : 100,
                         'numstep_stop'          : 100,
                         'timint'                : 'ost', # ost
                         'theta_ost'             : 1.0,
                         'initial_conditions'    : init()}
    
    MODEL_PARAMS      = {'modeltype'             : 'syspulcapveins',
                         'parameters'            : param(),
                         'chamber_models'        : {'lv' : {'type' : '0D_elast', 'activation_curve' : 2}, 'rv' : {'type' : '0D_elast', 'activation_curve' : 2}, 'la' : {'type' : '0D_elast', 'activation_curve' : 1}, 'ra' : {'type' : '0D_elast', 'activation_curve' : 1}}}
    

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
            
            act_dur = param()['t_es'] - param()['t_ed']
            t0 = 2.*param()['t_ed']
            
            if t >= t0 and t <= t0 + act_dur:
                return 0.5*(1.-np.cos(2.*np.pi*(t-t0)/act_dur))
            else:
                return 0.0


    # problem setup
    problem = ambit.Ambit(IO_PARAMS, TIME_PARAMS, SOLVER_PARAMS, constitutive_params=MODEL_PARAMS, time_curves=time_curves())
    
    # solve time-dependent problem
    problem.solve_problem()


    # --- results check
    tol = 1.0e-6

    s_corr = np.zeros(problem.mp.cardvasc0D.numdof)

    # correct results
    s_corr[0]    = -3.7244986085293509E-02
    s_corr[1]    = 6.6932639780917180E-01
    s_corr[2]    = -5.2573583030550763E-01
    s_corr[3]    = 1.0417762586621069E+00
    s_corr[4]    = 6.2991345617171826E+00
    s_corr[5]    = 2.3324218329598953E+03
    s_corr[6]    = 3.1168360834225379E+04
    s_corr[7]    = 6.1085239357105783E+00
    s_corr[8]    = 1.0219963957371492E+04
    s_corr[9]    = 9.7409540209457809E+03
    s_corr[10]    = 7.6426672398391693E+03
    s_corr[11]    = 5.2083153002715408E+03
    s_corr[12]    = 2.2466039554990407E+00
    s_corr[13]    = 2.2349167509196632E+04
    s_corr[14]    = 2.1968685114274211E+00
    s_corr[15]    = 1.8590826822383689E+04
    s_corr[16]    = 2.1946223301820558E+00
    s_corr[17]    = 1.4576658670446670E+04
    s_corr[18]    = 2.2016624626708823E+00
    s_corr[19]    = 1.0003657436246565E+04
    s_corr[20]    = 1.0503457656975546E+00
    s_corr[21]    = 5.5863706802023999E+03
    s_corr[22]    = 1.8818655417489523E+00
    s_corr[23]    = 0.5*7.1142413583451998E+04
    s_corr[24]    = 0.5*7.1142413583451998E+04
    s_corr[25]    = 4.1786976379070533E+04
    s_corr[26]    = 5.1593120094667277E-01
    s_corr[27]    = -8.2215743561552895E-02
    s_corr[28]    = 4.7414422456760225E-01
    s_corr[29]    = 1.2963016601831310E+00
    s_corr[30]    = 6.1365436157208123E+03
    s_corr[31]    = 1.2502775830652249E+00
    s_corr[32]    = 1.1628173324207832E+04
    s_corr[33]    = 1.1630662831336660E+00
    s_corr[34]    = 0.25*3.2915992354966293E+04
    s_corr[35]    = 0.25*3.2915992354966293E+04
    s_corr[36]    = 0.25*3.2915992354966293E+04
    s_corr[37]    = 0.25*3.2915992354966293E+04

    check1 = results_check.results_check_vec(problem.mp.s, s_corr, problem.mp.comm, tol=tol)
    success = results_check.success_check([check1], problem.mp.comm)

    return success



def init():
    
    factor_kPa_mmHg = 7.500615
    
    return {'q_vin_l_0' : 0.0,
            'p_at_l_0' : 4.5/factor_kPa_mmHg,
            'q_vout_l_0' : 0.0,
            'p_v_l_0' : 4.5/factor_kPa_mmHg,
            'p_ar_sys_0' : 90.29309546/factor_kPa_mmHg,
            'q_ar_sys_0' : 0.0,

            'p_arperi_sys_0' : 90.29309546/factor_kPa_mmHg,
            'q_arspl_sys_0' : 0.0,
            'q_arespl_sys_0' : 0.0,
            'q_armsc_sys_0' : 0.0,
            'q_arcer_sys_0' : 0.0,
            'q_arcor_sys_0' : 0.0,
            'p_venspl_sys_0' : 17.0/factor_kPa_mmHg,
            'q_venspl_sys_0' : 0.0,
            'p_venespl_sys_0' : 17.0/factor_kPa_mmHg,
            'q_venespl_sys_0' : 0.0,
            'p_venmsc_sys_0' : 17.0/factor_kPa_mmHg,
            'q_venmsc_sys_0' : 0.0,
            'p_vencer_sys_0' : 17.0/factor_kPa_mmHg,
            'q_vencer_sys_0' : 0.0,
            'p_vencor_sys_0' : 17.0/factor_kPa_mmHg,
            'q_vencor_sys_0' : 0.0,

            'p_ven_sys_0' : 17.0/factor_kPa_mmHg,
            'q_ven_sys_0' : 0.0,
            'q_ven1_sys_0' : 0.0,
            'q_ven2_sys_0' : 0.0,
            'q_vin_r_0' : 0.0,
            'p_at_r_0' : 4.5/(5.*factor_kPa_mmHg),
            'q_vout_r_0' : 0.0,
            'p_v_r_0' : 4.5/(5.*factor_kPa_mmHg),
            'p_ar_pul_0' : 90.29309546/(5.*factor_kPa_mmHg),
            'q_ar_pul_0' : 0.0,
            'p_cap_pul_0' : 90.29309546/(5.*factor_kPa_mmHg),
            'q_cap_pul_0' : 0.0,
            'p_ven_pul_0' : 12.0/factor_kPa_mmHg,
            'q_ven_pul_0' : 0.0,
            'q_ven1_pul_0' : 0.0,
            'q_ven2_pul_0' : 0.0,
            'q_ven3_pul_0' : 0.0,
            'q_ven4_pul_0' : 0.0}


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
    Z_ar_pul = 0.
    R_ven_pul = R_ar_pul
    C_ven_pul = 2.5*C_ar_pul
    
    L_ar_sys = 0.667e-6
    L_ven_sys = 0.
    L_ar_pul = 0.
    L_ven_pul = 0.
    
    # timings
    t_ed = 0.18
    t_es = 0.58
    
    

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
            'R_arcor_sys' : R_arcor_sys,
            'C_arcor_sys' : C_arcor_sys,
            'R_venspl_sys' : R_venspl_sys,
            'C_venspl_sys' : C_venspl_sys,
            'R_venespl_sys' : R_venespl_sys,
            'C_venespl_sys' : C_venespl_sys,
            'R_venmsc_sys' : R_venmsc_sys,
            'C_venmsc_sys' : C_venmsc_sys,
            'R_vencer_sys' : R_vencer_sys,
            'C_vencer_sys' : C_vencer_sys,
            'R_vencor_sys' : R_vencor_sys,
            'C_vencor_sys' : C_vencor_sys,
            'R_ar_pul' : R_ar_pul,
            'C_ar_pul' : C_ar_pul,
            'L_ar_pul' : L_ar_pul,
            'Z_ar_pul' : Z_ar_pul,
            'R_cap_pul' : R_cap_pul,
            'C_cap_pul' : C_cap_pul,
            'R_ven_sys' : R_ven_sys,
            'C_ven_sys' : C_ven_sys, 
            'L_ven_sys' : L_ven_sys,
            'R_ven_pul' : R_ven_pul,
            'C_ven_pul' : C_ven_pul,
            'L_ven_pul' : L_ven_pul,
            # atrial elastances
            'E_at_max_l' : 2.9e-5,
            'E_at_min_l' : 9.0e-6,
            'E_at_max_r' : 1.8e-5,
            'E_at_min_r' : 8.0e-6,
            # ventricular elastances
            'E_v_max_l' : 7.0e-5,
            'E_v_min_l' : 12.0e-6,
            'E_v_max_r' : 3.0e-5,
            'E_v_min_r' : 10.0e-6,
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
            'T_cycl' : 0.9,
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
    
    success = False
    
    try:
        success = main()
    except:
        print(traceback.format_exc())
    
    if success:
        sys.exit(0)
    else:
        sys.exit(1)
