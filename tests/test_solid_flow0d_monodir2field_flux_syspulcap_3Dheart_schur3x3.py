#!/usr/bin/env python3

"""
3D biventricular generic heart, testing of:
- incompressible Neo-Hookean material (p2p1 interpolation), viscous Green material
- 3D-0D monolithic solution of 3D heart w/ syspulcap circulation (flux coupling)
- Robin BCs in xyz and normal direction (spring and dashpot)
- OST time-integration for solid
- 3x3 block iterative method for incompressible solid coupled to 0D model
"""

import ambit_fe

import sys
import numpy as np
from pathlib import Path
import pytest


@pytest.mark.solid_flow0d
def test_main():

    basepath = str(Path(__file__).parent.absolute())


    IO_PARAMS            = {'problem_type'          : 'solid_flow0d',
                            'mesh_domain'           : basepath+'/input/heart3Dcoarse_domain.xdmf',
                            'mesh_boundary'         : basepath+'/input/heart3Dcoarse_boundary.xdmf',
                            'write_results_every'   : -999,
                            'output_path'           : basepath+'/tmp/',
                            'results_to_write'      : ['displacement','pressure'],
                            'simname'               : 'test',
                            'ode_parallel'          : True}

    SOLVER_PARAMS        = {'solve_type'            : 'iterative',
                            'iterative_solver'      : 'gmres',
                            'block_precond'         : 'schur3x3',
                            'precond_fields'        : [{'prec':'amg'}, {'prec':'amg'}, {'prec':'direct'}],
                            'tol_res'               : [1.0e-8,1.0e-8,1.0e-6], # u,p,0d
                            'tol_inc'               : [1.0e-8,1.0e-8,1.0e-6], # u,p,0d
                            'tol_lin_rel'           : 1.0e-9,
                            'lin_norm_type'         : 'preconditioned',
                            'print_liniter_every'   : 50,
                            'divergence_continue'   : None,
                            'ptc'                   : False,
                            'k_ptc_initial'         : 0.1}

    TIME_PARAMS_SOLID    = {'maxtime'               : 1.0,
                            'numstep'               : 100,
                            'numstep_stop'          : 1,
                            'timint'                : 'ost',
                            'theta_ost'             : 0.5}

    TIME_PARAMS_FLOW0D   = {'timint'                : 'ost',
                            'theta_ost'             : 0.5,
                            'initial_conditions'    : init(),
                            'eps_periodic'          : 1.0e-3,
                            'periodic_checktype'    : None}

    MODEL_PARAMS_FLOW0D  = {'modeltype'             : 'syspulcap',
                            'parameters'            : param(),
                            'chamber_models'        : {'lv' : {'type' : '3D_solid'},
                                                       'rv' : {'type' : '3D_solid'},
                                                       'la' : {'type' : '0D_elast', 'activation_curve' : 1},
                                                       'ra' : {'type' : '0D_elast', 'activation_curve' : 1}}}

    FEM_PARAMS           = {'order_disp'            : 2,
                            'order_pres'            : 1,
                            'quad_degree'           : 5,
                            'incompressible_2field' : True}

    COUPLING_PARAMS      = {'surface_ids'           : [[1],[2]],
                            'surface_p_ids'         : [[1],[2]],
                            'coupling_quantity'     : ['flux','flux'],
                            'coupling_type'         : 'monolithic_direct'}

    MATERIALS            = {'MAT1' : {'neohooke_dev'      : {'mu' : 10.},
                                      'inertia'           : {'rho0' : 1.0e-6},
                                      'visco_green'       : {'eta' : 0.0001}}}



    # define your load curves here (syntax: tcX refers to curve X, to be used in BC_DICT key 'curve' : [X,0,0], or 'curve' : X)
    # some examples... up to 9 possible (tc1 until tc9 - feel free to implement more in timeintegration.py --> timecurves function if needed...)
    class time_curves():

        def tc1(self, t): # atrial activation

            act_dur = 2.*param()['t_ed']
            t0 = 0.

            if t >= t0 and t <= t0 + act_dur:
                return 0.5*(1.-np.cos(2.*np.pi*(t-t0)/act_dur))
            else:
                return 0.0


    BC_DICT              = { 'robin' : [{'type' : 'spring',  'id' : [3], 'dir' : 'normal_ref', 'stiff' : 0.075},
                                        {'type' : 'dashpot', 'id' : [3], 'dir' : 'normal_ref', 'visc'  : 0.005},
                                        {'type' : 'spring',  'id' : [4], 'dir' : 'normal_ref', 'stiff' : 2.5},
                                        {'type' : 'dashpot', 'id' : [4], 'dir' : 'normal_ref', 'visc'  : 0.0005},
                                        {'type' : 'spring',  'id' : [4], 'dir' : 'xyz_ref', 'stiff' : 0.25},
                                        {'type' : 'dashpot', 'id' : [4], 'dir' : 'xyz_ref', 'visc'  : 0.0005}] }

    # problem setup
    problem = ambit_fe.ambit_main.Ambit(IO_PARAMS, [TIME_PARAMS_SOLID, TIME_PARAMS_FLOW0D], SOLVER_PARAMS, FEM_PARAMS, [MATERIALS, MODEL_PARAMS_FLOW0D], BC_DICT, time_curves=time_curves(), coupling_params=COUPLING_PARAMS)

    # problem solve
    problem.solve_problem()

    # --- results check
    tol = 1.0e-6

    s_corr = np.zeros(problem.mp.pb0.cardvasc0D.numdof)

    # correct 0D results
    s_corr[0] = 1.5140154884790708E+06
    s_corr[1] = 1.2842189823315804E+00
    s_corr[2] = -2.2969712709549372E+00
    s_corr[3] = -2.2979650614749048E-01
    s_corr[4] = 1.2035047941266040E+01
    s_corr[5] = -2.2969712709549372E+00
    s_corr[6] = 1.2035061723093666E+01
    s_corr[7] = 7.9266376767811471E+03
    s_corr[8] = 1.0920576465738236E+01
    s_corr[9] = 4.8757248185760414E+04
    s_corr[10] = 4.5874998121789955E+04
    s_corr[11] = 3.5972406593841457E+04
    s_corr[12] = 2.4558826877863074E+04
    s_corr[13] = 8.1860969826577357E+03
    s_corr[14] = 2.2677989771637215E+00
    s_corr[15] = 8.3769512865623369E+02
    s_corr[16] = 2.2702566758279015E+00
    s_corr[17] = 9.5189348228574841E+02
    s_corr[18] = 2.2702868360134905E+00
    s_corr[19] = 7.5312469003723868E+02
    s_corr[20] = 2.2701207039761671E+00
    s_corr[21] = 5.0027875726087944E+02
    s_corr[22] = 2.2705227157051970E+00
    s_corr[23] = 1.7138027932289609E+02
    s_corr[24] = 2.2541277926640513E+00
    s_corr[25] = 2.0733859810437830E+05
    s_corr[26] = 9.9018455409868766E+04
    s_corr[27] = 2.7306272250688318E-01
    s_corr[28] = -4.3686457082345953E-01
    s_corr[29] = 1.7404426709701443E-01
    s_corr[30] = 2.4017163229044409E+00
    s_corr[31] = 1.1803825766722653E+04
    s_corr[32] = 2.3131876296540206E+00
    s_corr[33] = 2.0066547489885686E+05
    s_corr[34] = 1.6159462113751601E+00
    s_corr[35] = 3.9891468722433310E+04

    check1 = ambit_fe.resultcheck.results_check_vec(problem.mp.pb0.s, s_corr, problem.mp.comm, tol=tol)
    success = ambit_fe.resultcheck.success_check([check1], problem.mp.comm)

    if not success:
        raise RuntimeError("Test failed!")



# syspulcap circulation model initial condition and parameter dicts...

def init():

    factor_kPa_mmHg = 7.500615

    return {'q_vin_l_0' : 0.0,
            'p_at_l_0' : 10.0/factor_kPa_mmHg,
            'q_vout_l_0' : 0.0,
            'p_v_l_0' : 10.0/factor_kPa_mmHg,
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
            'q_vin_r_0' : 0.0,
            'p_at_r_0' : 10.0/(5.*factor_kPa_mmHg),
            'q_vout_r_0' : 0.0,
            'p_v_r_0' : 10.0/(5.*factor_kPa_mmHg),
            'p_ar_pul_0' : 90.29309546/(5.*factor_kPa_mmHg),
            'q_ar_pul_0' : 0.0,
            'p_cap_pul_0' : 90.29309546/(5.*factor_kPa_mmHg),
            'q_cap_pul_0' : 0.0,
            'p_ven_pul_0' : 12.0/factor_kPa_mmHg,
            'q_ven_pul_0' : 0.0}



def param():

    R_ar_sys = 120.0e-6
    tau_ar_sys = 1.65242332
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

    # atrial elastances
    E_at_A_l, E_at_min_l = 20.0e-6, 9.0e-6
    E_at_A_r, E_at_min_r = 10.0e-6, 8.0e-6

    # timings
    t_ed = 0.2
    t_es = 0.53
    T_cycl = 1.0

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

    ## muscular resistance (and hence total systemic arterial resistance!) falls in sportsmode (Hogan 2009)
    #if sportsmode:
        #R_armsc_sys *= 0.3

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
            'E_at_max_l' : E_at_min_l+E_at_A_l,
            'E_at_min_l' : E_at_min_l,
            'E_at_max_r' : E_at_min_r+E_at_A_r,
            'E_at_min_r' : E_at_min_r,
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
