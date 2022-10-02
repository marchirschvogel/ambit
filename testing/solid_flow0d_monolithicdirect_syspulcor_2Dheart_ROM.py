#!/usr/bin/env python3

### 2D biventricular generic heart, testing of:
# - Holzapfel-Ogden material
# - active stress
# - 3D-0D monolithic solution of 2D heart w/ syspul circulation (volume coupling) and coronary model ZCRp_CRd_lr
# - Robin BCs in normal direction (spring and dashpot)
# - Gen-Alpha time-integration for solid

import ambit

import sys, traceback
import numpy as np
from pathlib import Path

import resultcheck


def main():
    
    basepath = str(Path(__file__).parent.absolute())

    # reads in restart step from the command line
    try: restart_step = int(sys.argv[1])
    except: restart_step = 0

    IO_PARAMS            = {'problem_type'          : 'solid_flow0d', # solid, fluid, flow0d, solid_flow0d, fluid_flow0d
                            'use_model_order_red'   : True,
                            'mesh_domain'           : basepath+'/input/heart2D_domain.xdmf',
                            'mesh_boundary'         : basepath+'/input/heart2D_boundary.xdmf',
                            'fiber_data'            : {'nodal' : [basepath+'/input/fib_fiber_coords_nodal_2D.txt',basepath+'/input/fib_sheet_coords_nodal_2D.txt']},
                            'write_results_every'   : -999,
                            'output_path'           : basepath+'/tmp/',
                            'results_to_write'      : ['displacement'],
                            'simname'               : 'ROM_2Dheart', # ROM_2Dheart
                            'write_restart_every'   : -999,
                            'restart_step'          : restart_step}

    ROM_PARAMS           = {'hdmfilenames'          : [basepath+'/input/checkpoint_romsnaps_2Dheart_u_*_1proc.dat'],
                            'numhdms'               : 1,
                            'numsnapshots'          : 5,
                            'snapshotincr'          : 1,
                            'numredbasisvec'        : 5,
                            'eigenvalue_cutoff'     : 1.0e-8,
                            'print_eigenproblem'    : True}

    SOLVER_PARAMS_SOLID  = {'solve_type'            : 'direct', # direct, iterative
                            'tol_res'               : 1.0e-8,
                            'tol_inc'               : 1.0e-8}
    
    SOLVER_PARAMS_FLOW0D = {'tol_res'               : 1.0e-6,
                            'tol_inc'               : 1.0e-6}

    TIME_PARAMS_SOLID    = {'maxtime'               : 1.0,
                            'numstep'               : 500,
                            'numstep_stop'          : 5,
                            'timint'                : 'genalpha',
                            'rho_inf_genalpha'      : 0.8}
    
    TIME_PARAMS_FLOW0D   = {'timint'                : 'ost', # ost
                            'theta_ost'             : 0.5,
                            'initial_conditions'    : init()}

    MODEL_PARAMS_FLOW0D  = {'modeltype'             : 'syspul',
                            'coronary_model'        : 'ZCRp_CRd_lr',
                            'parameters'            : param(),
                            'chamber_models'        : {'lv' : {'type' : '3D_solid'}, 'rv' : {'type' : '3D_solid'}, 'la' : {'type' : '0D_elast', 'activation_curve' : 2}, 'ra' : {'type' : '0D_elast', 'activation_curve' : 2}}}

    FEM_PARAMS           = {'order_disp'            : 1,
                            'order_pres'            : 1,
                            'quad_degree'           : 2,
                            'incompressible_2field' : False}
    
    COUPLING_PARAMS      = {'surface_ids'           : [[1],[2]],
                            'cq_factor'             : [80.,80.],
                            'coupling_quantity'     : ['volume']*2,
                            'coupling_type'         : 'monolithic_direct'}

    MATERIALS            = {'MAT1' : {'holzapfelogden_dev'    : {'a_0' : 0.059, 'b_0' : 8.023, 'a_f' : 18.472, 'b_f' : 16.026, 'a_s' : 2.481, 'b_s' : 11.120, 'a_fs' : 0.216, 'b_fs' : 11.436},
                                      'sussmanbathe_vol'      : {'kappa' : 1.0e3},
                                      'active_fiber'          : {'sigma0' : 50.0, 'alpha_max' : 15.0, 'alpha_min' : -20.0, 'activation_curve' : 1},
                                      'inertia'               : {'rho0' : 1.0e-6}}}



    # define your load curves here (syntax: tcX refers to curve X, to be used in BC_DICT key 'curve' : [X,0,0], or 'curve' : X)
    class time_curves():
        
        def tc1(self, t):
            
            K = 5.
            t_contr, t_relax = 0.0, 0.53
            
            alpha_max = MATERIALS['MAT1']['active_fiber']['alpha_max']
            alpha_min = MATERIALS['MAT1']['active_fiber']['alpha_min']
            
            c1 = t_contr + alpha_max/(K*(alpha_max-alpha_min))
            c2 = t_relax - alpha_max/(K*(alpha_max-alpha_min))
            
            # Diss Hirschvogel eq. 2.101
            return (K*(t-c1)+1.)*((K*(t-c1)+1.)>0.) - K*(t-c1)*((K*(t-c1))>0.) - K*(t-c2)*((K*(t-c2))>0.) + (K*(t-c2)-1.)*((K*(t-c2)-1.)>0.)

        def tc2(self, t): # atrial activation
            
            act_dur = 2.*param()['t_ed']
            t0 = 0.
            
            if t >= t0 and t <= t0 + act_dur:
                return 0.5*(1.-np.cos(2.*np.pi*(t-t0)/act_dur))
            else:
                return 0.0


    BC_DICT              = { 'dirichlet' : [{'dir' : '2dimZ', 'val' : 0.}],
                            'robin' : [{'type' : 'spring', 'id' : [3], 'dir' : 'normal', 'stiff' : 0.075},
                                       {'type' : 'dashpot', 'id' : [3], 'dir' : 'normal', 'visc' : 0.005}] }

    # problem setup
    problem = ambit.Ambit(IO_PARAMS, [TIME_PARAMS_SOLID, TIME_PARAMS_FLOW0D], [SOLVER_PARAMS_SOLID, SOLVER_PARAMS_FLOW0D], FEM_PARAMS, [MATERIALS, MODEL_PARAMS_FLOW0D], BC_DICT, time_curves=time_curves(), coupling_params=COUPLING_PARAMS, mor_params=ROM_PARAMS)
    
    # solve time-dependent problem
    problem.solve_problem()


    # --- results check
    tol = 1.0e-6
        
    s_corr = np.zeros(problem.mp.pbf.cardvasc0D.numdof)
    
    # correct 0D results
    s_corr[0] = 1.0216731075276814E+05
    s_corr[1] = 3.8512297794648398E-01
    s_corr[2] = -1.0582718169772605E+00
    s_corr[3] = 2.8295566718368870E-01
    s_corr[4] = 1.0865673836956294E+01
    s_corr[5] = -7.3632801641344558E+04
    s_corr[6] = 1.0874047381917078E+01
    s_corr[7] = 7.1800294676209058E+04
    s_corr[8] = 2.2872592766740136E+00
    s_corr[9] = 8.4948270377698762E+04
    s_corr[10] = 6.2810990726745433E+04
    s_corr[11] = 2.4850078760924649E-01
    s_corr[12] = -2.0264606775367924E-01
    s_corr[13] = 1.8568979691508847E-01
    s_corr[14] = 2.2121504744518803E+00
    s_corr[15] = 3.5717307399521123E+04
    s_corr[16] = 1.6763908634590652E+00
    s_corr[17] = 8.6084525700838829E+04
    s_corr[18] = 6.9673864322706243E+02
    s_corr[19] = 2.9191035744740575E+03
    s_corr[20] = 4.4292718560236699E-01
    s_corr[21] = -3.6278491617589065E-01
    s_corr[22] = 6.9673864322706231E+02
    s_corr[23] = 2.9191035744740580E+03
    s_corr[24] = 4.4292718560236699E-01
    s_corr[25] = -3.6278491617589070E-01
    s_corr[26] = -8.5023369056630516E+04

    check1 = resultcheck.results_check_vec(problem.mp.pbf.s, s_corr, problem.mp.comm, tol=tol)
    success = resultcheck.success_check([check1], problem.mp.comm)
    
    return success



def init():
    
    return {'q_vin_l_0' : 1.1549454594333263E+04,
            'p_at_l_0' : 3.8580961077622145E-01,
            'q_vout_l_0' : -1.0552685263595845E+00,
            'p_v_l_0' : 3.7426015618188813E-01,
            'p_ar_sys_0' : 1.0926945419777734E+01,
            'q_ar_sys_0' : 7.2237210814547114E+04,
            'p_ven_sys_0' : 2.2875736545217800E+00,
            'q_ven_sys_0' : 8.5022643486798144E+04,
            'q_vin_r_0' : 4.1097788677528049E+04,
            'p_at_r_0' : 2.4703021083862464E-01,
            'q_vout_r_0' : -2.0242075369768467E-01,
            'p_v_r_0' : 2.0593242216109664E-01,
            'p_ar_pul_0' : 2.2301399591379436E+00,
            'q_ar_pul_0' : 3.6242987765574515E+04,
            'p_ven_pul_0' : 1.6864951426543255E+00,
            'q_ven_pul_0' : 8.6712368791873596E+04,
            # coronary circulation submodel
            'q_cor_sys_l_0' : 0,
            'q_cord_sys_l_0' : 0,
            'q_corp_sys_l_0' : 0.,
            'p_cord_sys_l_0' : 0.,
            'q_cor_sys_r_0' : 0,
            'q_cord_sys_r_0' : 0,
            'q_corp_sys_r_0' : 0.,
            'p_cord_sys_r_0' : 0.}


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
            'T_cycl' : 1.0,
            # coronary circulation submodel parameters - values from Arthurs et al. 2016, Tab. 3
            'Z_corp_sys_l' : 3.2e-3,
            'C_corp_sys_l' : 4.5e0,
            'R_corp_sys_l' : 6.55e-3,
            'C_cord_sys_l' : 2.7e1,
            'R_cord_sys_l' : 1.45e-1,
            'Z_corp_sys_r' : 3.2e-3,
            'C_corp_sys_r' : 4.5e0,
            'R_corp_sys_r' : 6.55e-3,
            'C_cord_sys_r' : 2.7e1,
            'R_cord_sys_r' : 1.45e-1}



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
