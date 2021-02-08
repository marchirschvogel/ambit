#!/usr/bin/env python3

### 2D biventricular generic heart, testing of:
# - Holzapfel-Ogden material
# - active stress
# - 3D-0D monolithic solution of 2D heart w/ syspul circulation (volume coupling)
# - Robin BCs in normal direction (spring and dashpot)
# - Gen-Alpha time-integration for solid

import ambit

import sys, traceback
import numpy as np
from pathlib import Path

import results_check


def main():
    
    basepath = str(Path(__file__).parent.absolute())

    IO_PARAMS            = {'problem_type'          : 'solid_flow0d_multiscale_gandr',
                            'mesh_domain'           : ''+basepath+'/input/heart2Dcoarse_domain.xdmf',
                            'mesh_boundary'         : ''+basepath+'/input/heart2Dcoarse_boundary.xdmf',
                            'fiber_data'            : {'nodal' : [''+basepath+'/input/fib_fiber_coords_nodal_2Dcoarse.txt',''+basepath+'/input/fib_sheet_coords_nodal_2Dcoarse.txt']},
                            'write_results_every'   : 1,
                            'output_path'           : ''+basepath+'/tmp/',
                            'results_to_write'      : ['displacement'],
                            'simname'               : 'multiscale_gandr'}

    SOLVER_PARAMS_SOLID  = {'solve_type'            : 'direct', # direct, iterative
                            'tol_res'               : 1.0e-8,
                            'tol_inc'               : 1.0e-8}
    
    SOLVER_PARAMS_FLOW0D = {'tol_res'               : 1.0e-6,
                            'tol_inc'               : 1.0e-6}

    TIME_PARAMS_SOLID_SMALL = {'maxtime'            : 2.0,
                            'numstep'               : 100,
                            'timint'                : 'static', # genalpha, ost, static
                            'theta_ost'             : 1.0,
                            'rho_inf_genalpha'      : 0.8}

    TIME_PARAMS_SOLID_LARGE = {'maxtime'            : 1.0,
                            'numstep'               : 100,
                            'timint'                : 'static'}

    TIME_PARAMS_FLOW0D   = {'timint'                : 'ost', # ost
                            'theta_ost'             : 0.5,
                            'initial_conditions'    : init(),
                            'eps_periodic'          : 999,
                            'periodic_checktype'    : 'pQvar'}

    MODEL_PARAMS_FLOW0D  = {'modeltype'             : 'syspul',
                            'parameters'            : param(),
                            'chamber_models'        : {'lv' : '3D_fem', 'rv' : '3D_fem', 'la' : '0D_elast', 'ra' : '0D_elast'},
                            'perturb_type'          : 'as',
                            'perturb_after_cylce'   : 1}

    FEM_PARAMS           = {'order_disp'            : 1,
                            'order_pres'            : 1,
                            'quad_degree'           : 1,
                            'incompressible_2field' : False}
    
    COUPLING_PARAMS      = {'surface_ids'           : [1,2],
                            'cq_factor'             : [80.,80.],
                            'coupling_quantity'     : 'volume',
                            'coupling_type'         : 'monolithic_direct'}
    
    MULTISCALE_GR_PARAMS = {'gandr_trigger_phase'   : 'end_systole', # end_diastole, end_systole
                            'numcycles'             : 2}

    #MATERIALS            = {'MAT1' : {'holzapfelogden_dev'    : {'a_0' : 0.059, 'b_0' : 8.023, 'a_f' : 18.472, 'b_f' : 16.026, 'a_s' : 2.481, 'b_s' : 11.120, 'a_fs' : 0.216, 'b_fs' : 11.436},
                                      #'sussmanbathe_vol'      : {'kappa' : 1.0e3},
                                      #'active_fiber'          : {'sigma0' : 50.0, 'alpha_max' : 15.0, 'alpha_min' : -20.0, 't_contr' : 0.0, 't_relax' : 0.53},
                                      #'inertia'               : {'rho0' : 1.0e-6},
                                      #'growth'           : {'growth_dir' : 'isotropic', # isotropic, fiber, crossfiber, radial
                                                            #'growth_trig' : 'fibstretch', # fibstretch, volstress, prescribed
                                                            #'thetamax' : 3.0,
                                                            #'thetamin' : 1.0,
                                                            #'stretch_crit' : 1.15,
                                                            #'tau_gr' : 1.0,
                                                            #'gamma_gr' : 1.72,
                                                            #'tau_gr_rev' : 10000.0,
                                                            #'gamma_gr_rev' : 1.0,
                                                            #'remodeling_mat' : {'neohooke_dev' : {'mu' : 3.},
                                                                                #'ogden_vol'    : {'kappa' : 3./(1.-2.*0.49)}}}}}


    MATERIALS            = {'MAT1' : {'neohooke_dev'    : {'mu' : 10.},
                                      'sussmanbathe_vol'      : {'kappa' : 10./(1.-2.*0.49)},
                                      'active_fiber'          : {'sigma0' : 50.0, 'alpha_max' : 15.0, 'alpha_min' : -20.0, 't_contr' : 0.2, 't_relax' : 0.53},
                                      'inertia'               : {'rho0' : 1.0e-6},
                                      'growth'           : {'growth_dir' : 'isotropic', # isotropic, fiber, crossfiber, radial
                                                            'growth_trig' : 'volstress', # fibstretch, volstress, prescribed
                                                            'growth_thres' : 1.15,
                                                            'thetamax' : 3.0,
                                                            'thetamin' : 1.0,
                                                            'tau_gr' : 100.0,
                                                            'gamma_gr' : 1.72,
                                                            'tau_gr_rev' : 10000.0,
                                                            'gamma_gr_rev' : 1.0,
                                                            'remodeling_mat' : {'neohooke_dev' : {'mu' : 3.},
                                                                                'ogden_vol'    : {'kappa' : 3./(1.-2.*0.49)}}}}}


    #MATERIALS            = {'MAT1' : {'neohooke_dev'    : {'mu' : 10.},
                                      #'sussmanbathe_vol'      : {'kappa' : 10./(1.-2.*0.49)},
                                      #'active_fiber'          : {'sigma0' : 50.0, 'alpha_max' : 15.0, 'alpha_min' : -20.0, 't_contr' : 0.2, 't_relax' : 0.53},
                                      #'inertia'               : {'rho0' : 1.0e-6}}}


    # define your load curves here (syntax: tcX refers to curve X, to be used in BC_DICT key 'curve' : [X,0,0], or 'curve' : X)
    # None to be defined


    BC_DICT              = { 'dirichlet' : [{'dir' : '2dim', 'val' : 0.}],
                            'robin' : [{'type' : 'spring', 'id' : 3, 'dir' : 'normal', 'stiff' : 0.075},
                                       {'type' : 'dashpot', 'id' : 3, 'dir' : 'normal', 'visc' : 0.005}] }

    # problem setup
    problem = ambit.Ambit(IO_PARAMS, [TIME_PARAMS_SOLID_SMALL, TIME_PARAMS_SOLID_LARGE, TIME_PARAMS_FLOW0D], [SOLVER_PARAMS_SOLID, SOLVER_PARAMS_FLOW0D], FEM_PARAMS, [MATERIALS, MODEL_PARAMS_FLOW0D], BC_DICT, coupling_params=COUPLING_PARAMS, multiscale_params=MULTISCALE_GR_PARAMS)
    
    # solve time-dependent problem
    problem.solve_problem()


    ## --- results check
    #tol = 1.0e-6
        
    #s_corr = np.zeros(problem.mp.pbf.cardvasc0D.numdof)

    ## correct 0D results
    #s_corr[0]    = -2.0114350666760171E-02
    #s_corr[1]    = 3.9868184096752618E-01
    #s_corr[2]    = -1.0274818383992834E+00
    #s_corr[3]    = 5.9982534828086920E-01
    #s_corr[4]    = 1.0874643732273704E+01
    #s_corr[5]    = 7.1802465982943570E+04
    #s_corr[6]    = 2.2872622295989982E+00
    #s_corr[7]    = 8.4716159930435591E+04
    #s_corr[8]    = -1.2619948195877369E-02
    #s_corr[9]    = 2.5407439126854764E-01
    #s_corr[10]    = -1.8318777040206888E-01
    #s_corr[11]    = 3.8027387249972561E-01
    #s_corr[12]    = 2.2121515765204145E+00
    #s_corr[13]    = 3.5710868582746974E+04
    #s_corr[14]    = 1.6764885477792106E+00
    #s_corr[15]    = 8.5187113787445720E+04

    #check1 = results_check.results_check_vec(problem.mp.pbf.s, s_corr, problem.mp.comm, tol=tol)
    #success = results_check.success_check([check1], problem.mp.comm)
    
    #return success



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
            'q_ven_pul_0' : 8.6712368791873596E+04}


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
