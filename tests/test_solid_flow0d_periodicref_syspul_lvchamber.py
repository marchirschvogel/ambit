#!/usr/bin/env python3

"""
Tests solid-flow0d problem perdiodicity on reference geometry: Problem is re-initialized with new 0D initial conditions on
the reference solid geometry and run until perdiodicity is reached - like this, one can obtain "periodic state initial conditions",
meaning initial conditions which produce a periodic cycle state starting from the reference geometry
"""

import ambit_fe

import sys
import numpy as np
from pathlib import Path
import pytest


@pytest.mark.solid_flow0d
def test_main():

    basepath = str(Path(__file__).parent.absolute())

    # reads in restart step from the command line
    try: restart_step = int(sys.argv[1])
    except: restart_step = 0
    
    try: restart_step_outer = int(sys.argv[2])
    except: restart_step_outer = 0

    IO_PARAMS            = {'problem_type'          : 'solid_flow0d_periodicref',
                            'write_results_every'   : 10,
                            'write_restart_every'   : 50,
                            'restart_step'          : restart_step,
                            'output_path'           : basepath+'/tmp/',
                            'mesh_domain'           : basepath+'/input/chamberhex_domain.xdmf',
                            'mesh_boundary'         : basepath+'/input/chamberhex_boundary.xdmf',
                            'fiber_data'            : [np.array([1.0,0.0,0.0]),np.array([0.0,1.0,0.0])],
                            'results_to_write'      : ['displacement','pressure','fibers','counters', 'tau_a'],
                            'simname'               : 'solid_flow0d_periodicref'}

    SOLVER_PARAMS        = {'solve_type'            : 'direct',
                            'solve_type_prestr'     : 'direct',
                            'tol_res'               : [1e-8,1e-8,1e-8],
                            'tol_inc'               : [1e-0,1e-0,1e-0],
                            'subsolver_params'      : {'tol_res' : 1.0e-6, 'tol_inc' : 1.0e-6}}

    number_of_cycles = 1

    TIME_PARAMS          = {'maxtime'               : 1.0,
                            'numstep'               : 100,
                            'timint'                : 'genalpha',
                            'rho_inf_genalpha'      : 0.8}

    TIME_PARAMS_FLOW0D   = {'timint'                : 'ost',
                            'theta_ost'             : 0.67,
                            'initial_backwardeuler' : True,
                            'initial_conditions'    : init(),
                            'periodic_checktype'    : ['allvar'],
                            'eps_periodic'          : 0.25} # large, only for testing purposes

    MODEL_PARAMS_FLOW0D  = {'modeltype'             : 'syspul',
                            'parameters'            : param(),
                            'chamber_models'        : {'lv' : {'type' : '3D_solid'},
                                                       'rv' : {'type' : '0D_elast', 'activation_curve' : 2},
                                                       'la' : {'type' : '0D_elast', 'activation_curve' : 3},
                                                       'ra' : {'type' : '0D_elast', 'activation_curve' : 3}}}

    FEM_PARAMS           = {'order_disp'            : 1, 
                            'order_pres'            : 1,
                            'quad_degree'           : 5,
                            'incompressible_2field' : False,
                            'prestress_initial'     : True,
                            'prestress_numstep'     : 1}
    
    COUPLING_PARAMS      = {'surface_ids'           : [[1]],
                            'coupling_quantity'     : ['pressure']*4,
                            'variable_quantity'     : ['flux']*4,
                            'coupling_type'         : 'monolithic_lagrange',
                            'print_subiter'         : True,
                            'write_checkpoints_periodicref' : True,
                            'restart_periodicref'   : restart_step_outer} # TODO: Currently broken...

    MATERIALS            = { 'MAT1' : {'guccione_dev'     : {'c_0' : 1.662, 'b_f' : 14.31, 'b_t' : 4.49, 'b_fs' : 10.},
                                       'sussmanbathe_vol' : {'kappa' : 1.0e1}, # very compressible
                                       'visco_green'      : {'eta' : 0.1},
                                       'active_fiber'     : {'sigma0' : 150., 'alpha_max' : 10.0, 'alpha_min' : -30.0, 'activation_curve' : 1, 'frankstarling' : True, 'amp_min' : 1., 'amp_max' : 1.5, 'lam_threslo' : 1.01, 'lam_maxlo' : 1.15, 'lam_threshi' : 999., 'lam_maxhi' : 9999.},
                                       'inertia'          : {'rho0' : 1.0e-6}} }


    class time_curves:
        
        def tc1(self, t):
            
            K = 5.
            t_contr, t_relax = 0.2, 0.53
            
            alpha_max = MATERIALS['MAT1']['active_fiber']['alpha_max']
            alpha_min = MATERIALS['MAT1']['active_fiber']['alpha_min']
            
            c1 = t_contr + alpha_max/(K*(alpha_max-alpha_min))
            c2 = t_relax - alpha_max/(K*(alpha_max-alpha_min))
            
            # Diss Hirschvogel eq. 2.101
            return (K*(t-c1)+1.)*((K*(t-c1)+1.)>0.) - K*(t-c1)*((K*(t-c1))>0.) - K*(t-c2)*((K*(t-c2))>0.) + (K*(t-c2)-1.)*((K*(t-c2)-1.)>0.)
        
        def tc2(self, t): # RV
            
            t_ed, t_es = 0.2, 0.53
            
            tvu, tvr = 0.25, 0.1
            
            act_rv_up = 0.5*(1.0)*(1.0-np.cos(np.pi*(t-t_ed)/tvu))
            act_rv_down = 0.5*(1.0)*(1.0-np.cos(np.pi*(t-(t_es+tvr))/tvr))
            
            # return 0.5*(1.-cos(2.*np.pi*(t-t_ed)/(1.8*(t_es-t_ed)))) * (t >= t_ed) * (t <= t_ed + 1.8*(t_es-t_ed))
            return 0 * (t < t_ed) + act_rv_up * (t >= t_ed)*(t < (t_ed+tvu)) + (1.0) * (t >= (t_ed+tvu))*(t < (t_es)) + act_rv_down * (t >= t_es)*(t < (t_es+tvr)) + 0 * (t > (t_es+tvr))

        def tc3(self, t): # RA
            
            t_ed, t_es = 0.2, 0.53
            
            return 0.5*(1.-np.cos(2.*np.pi*(t)/(2.*t_ed))) * (t >= 0.) * (t <= 2.*t_ed)

    class expression1: # prestress load
        def __init__(self):
            self.t = 0.0
            self.val = init()['p_v_l_0']

        def evaluate(self, x):
            pres = (-0.5*(self.val)*(1.-np.cos(np.pi*self.t/1.0)))
            return np.full(x.shape[1], pres)


    BC_DICT  = { 'dirichlet' : [{'id' : [1], 'dir' : 'y', 'val' : 0.},
                                {'id' : [1], 'dir' : 'z', 'val' : 0.}],
                 'robin' : [{'type' : 'spring', 'id' : [2,3,4,5,6,7], 'dir' : 'xyz_ref', 'stiff' : 0.5},
                            {'type' : 'dashpot', 'id' : [2,3,4,5,6,7], 'dir' : 'xyz_ref', 'visc' : 0.05}],
                 'neumann_prestress' : [{'id' : [1], 'dir' : 'normal_ref', 'expression' : expression1}] } # endo


    # problem setup
    problem = ambit_fe.ambit_main.Ambit(IO_PARAMS, [TIME_PARAMS, TIME_PARAMS_FLOW0D], SOLVER_PARAMS, FEM_PARAMS, [MATERIALS, MODEL_PARAMS_FLOW0D], BC_DICT, time_curves=time_curves(), coupling_params=COUPLING_PARAMS)
    
    # problem solve
    problem.solve_problem()


    # --- results check
    tol = 1.0e-6

    s_corr = np.zeros(problem.mp.pb0.cardvasc0D.numdof)

    # correct results
    s_corr[0] = 6.9421708117080034E+03
    s_corr[1] = 4.4439462150660586E-01
    s_corr[2] = -6.9429701758124638E+03
    s_corr[3] = -7.9936410446000439E-01
    s_corr[4] = 8.4310934952949417E+00
    s_corr[5] = -7.9936410446000439E-01
    s_corr[6] = 8.4310982914795680E+00
    s_corr[7] = 5.1723054532288545E+04
    s_corr[8] = 2.2450224403302048E+00
    s_corr[9] = 7.0956502600423570E+04
    s_corr[10] = 3.6548443149263097E+04
    s_corr[11] = 5.4206637792003998E-01
    s_corr[12] = -1.2050357157719344E-01
    s_corr[13] = 5.0551793477077689E-01
    s_corr[14] = 1.7105536505427112E+00
    s_corr[15] = 2.6480887501748493E+04
    s_corr[16] = 1.3133403380164841E+00
    s_corr[17] = 5.7929714433991918E+04

    check1 = ambit_fe.resultcheck.results_check_vec_sq(problem.mp.pb0.s, s_corr, problem.mp.comm, tol=tol)
    success = ambit_fe.resultcheck.success_check([check1], problem.mp.comm)

    if not success:
        raise RuntimeError("Test failed!")

    

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
            'Q_v_l_0' : 0.0}


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
            'E_v_max_r' : 9.0e-5,
            'E_v_min_r' : 8.0e-6,
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

    test_main()

