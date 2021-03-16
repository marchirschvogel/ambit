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

    SOLVER_PARAMS     = {'tol_res'               : 1.0e-6,
                         'tol_inc'               : 1.0e-6}

    TIME_PARAMS       = {'maxtime'               : 0.9,
                         'numstep'               : 100,
                         'numstep_stop'          : 100,
                         'timint'                : 'ost', # ost
                         'theta_ost'             : 1.0,
                         'initial_conditions'    : init()}
    
    MODEL_PARAMS      = {'modeltype'             : 'syspul',
                         'parameters'            : param()}
    

    # define your time curves here (syntax: tcX refers to curve X)
    # None to be defined


    # problem setup
    problem = ambit.Ambit(IO_PARAMS, TIME_PARAMS, SOLVER_PARAMS, constitutive_params=MODEL_PARAMS)
    
    # solve time-dependent problem
    problem.solve_problem()


    # --- results check
    tol = 1.0e-6

    s_corr = np.zeros(problem.mp.cardvasc0D.numdof)

    # correct results
    s_corr[0] = -3.3332274374530700e-02
    s_corr[1] = 7.1273201174680567e-01
    s_corr[2] = -4.2212343057960694e-01
    s_corr[3] = 1.0460547554921127e+00
    s_corr[4] = 5.2672890612881815e+00
    s_corr[5] = 2.7199350742243074e+04
    s_corr[6] = 2.0201935597893912e+00
    s_corr[7] = 6.7199687489435382e+04
    s_corr[8] = 3.5962423510257089e+04
    s_corr[9] = 4.0740106004294180e-01
    s_corr[10] = -1.1518064932704578e-01
    s_corr[11] = 3.7143863653268472e-01
    s_corr[12] = 1.5232451298031424e+00
    s_corr[13] = 1.5708616653606650e+04
    s_corr[14] = 1.2876158799990427e+00
    s_corr[15] = 3.8325591216815810e+04
    
    check1 = results_check.results_check_vec(problem.mp.s, s_corr, problem.mp.comm, tol=tol)
    success = results_check.success_check([check1], problem.mp.comm)
    
    return success


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
            'q_ven_pul_0' : 0.0}


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
    
    
    return {'R_ar_sys' : R_ar_sys,
            'C_ar_sys' : C_ar_sys,
            'L_ar_sys' : L_ar_sys,
            'Z_ar_sys' : Z_ar_sys,
            'R_ar_pul' : R_ar_pul,
            'C_ar_pul' : C_ar_pul,
            'L_ar_pul' : L_ar_pul,
            'Z_ar_pul' : Z_ar_pul,
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
            'V_at_l_u' : 0.0,
            'V_at_r_u' : 0.0,
            'V_v_l_u' : 0.0,
            'V_v_r_u' : 0.0,
            'V_ar_sys_u' : 0.0,
            'V_ar_pul_u' : 0.0,
            'V_ven_sys_u' : 0.0,
            'V_ven_pul_u' : 0.0}




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
