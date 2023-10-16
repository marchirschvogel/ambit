#!/usr/bin/env python3

### 2D biventricular generic heart, testing of:
# - Holzapfel-Ogden material
# - active stress
# - 3D-0D monolithic solution of 2D heart w/ syspul circulation (volume coupling)
# - Robin BCs in normal direction (spring and dashpot)
# - Gen-Alpha time-integration for solid

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

    IO_PARAMS            = {'problem_type'          : 'solid_flow0d',
                            'mesh_domain'           : basepath+'/input/heart2D_domain.xdmf',
                            'mesh_boundary'         : basepath+'/input/heart2D_boundary.xdmf',
                            'fiber_data'            : [basepath+'/input/fib_fiber_nodal_2D.txt',basepath+'/input/fib_sheet_nodal_2D.txt'],
                            'write_results_every'   : 1,
                            'output_path'           : basepath+'/tmp/',
                            'results_to_write'      : ['displacement','fibers'],
                            'simname'               : 'solid_flow0d_2Dheart',
                            'write_restart_every'   : 1,
                            'restart_step'          : restart_step,
                            'ode_parallel'          : True}

    SOLVER_PARAMS         = {'solve_type'            : 'direct',
                            'tol_res'               : [1.0e-8,1.0e-6],
                            'tol_inc'               : [1.0e-8,1.0e-6]}

    TIME_PARAMS_SOLID    = {'maxtime'               : 1.0,
                            'numstep'               : 500,
                            'numstep_stop'          : 5,
                            'timint'                : 'genalpha',
                            'theta_ost'             : 1.0,
                            'rho_inf_genalpha'      : 0.8}

    TIME_PARAMS_FLOW0D   = {'timint'                : 'ost',
                            'theta_ost'             : 0.5,
                            'initial_conditions'    : init()}

    MODEL_PARAMS_FLOW0D  = {'modeltype'             : 'syspul',
                            'parameters'            : param(),
                            'chamber_models'        : {'lv' : {'type' : '3D_solid'}, 'rv' : {'type' : '3D_solid'}, 'la' : {'type' : '0D_elast', 'activation_curve' : 2}, 'ra' : {'type' : '0D_elast', 'activation_curve' : 2}}}

    FEM_PARAMS           = {'order_disp'            : 1,
                            'order_pres'            : 1,
                            'quad_degree'           : 4,
                            'incompressible_2field' : False,
                            'prestress_initial'     : True,
                            'prestress_numstep'     : 1,
                            'prestress_ptc'         : True}

    COUPLING_PARAMS      = {'surface_ids'           : [[1],[2]],
                            'cq_factor'             : [80.,80.],
                            'coupling_quantity'     : ['volume']*4,
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

        def tc3(self, t): # prestress LV
            return -init()['p_v_l_0']*t

        def tc4(self, t): # prestress RV
            return -init()['p_v_r_0']*t


    BC_DICT              = { 'dirichlet' : [{'dir' : '2dimZ', 'val' : 0.}],
                             'robin' : [{'type' : 'spring', 'id' : [3], 'dir' : 'normal_ref', 'stiff' : 0.075},
                                       {'type' : 'dashpot', 'id' : [3], 'dir' : 'normal_ref', 'visc' : 0.005}],
                             'neumann_prestress' : [{'id' : [1], 'dir' : 'normal_ref', 'curve' : 3},
                                                    {'id' : [2], 'dir' : 'normal_ref', 'curve' : 4}] }

    # problem setup
    problem = ambit_fe.ambit_main.Ambit(IO_PARAMS, [TIME_PARAMS_SOLID, TIME_PARAMS_FLOW0D], SOLVER_PARAMS, FEM_PARAMS, [MATERIALS, MODEL_PARAMS_FLOW0D], BC_DICT, time_curves=time_curves(), coupling_params=COUPLING_PARAMS)

    # solve time-dependent problem
    problem.solve_problem()


    # --- results check
    tol = 1.0e-6

    s_corr = np.zeros(problem.mp.pb0.cardvasc0D.numdof)

    # correct 0D results
    s_corr[0] = -1.9983532816979475E-02
    s_corr[1] = 3.9869697103353280E-01
    s_corr[2] = -1.0276098789418020E+00
    s_corr[3] = 5.9853229971264454E-01
    s_corr[4] = 1.0874631089130659E+01
    s_corr[5] = -7.2239293692952386E+04
    s_corr[6] = 1.0874643586401094E+01
    s_corr[7] = 7.1802421951344339E+04
    s_corr[8] = 2.2872621917206417E+00
    s_corr[9] = 8.4721438737444128E+04
    s_corr[10] = -1.0558776135039956E-02
    s_corr[11] = 2.5394766202198654E-01
    s_corr[12] = -1.8526161544132652E-01
    s_corr[13] = 3.5953542322686693E-01
    s_corr[14] = 2.2121515776401313E+00
    s_corr[15] = 3.5710860692899922E+04
    s_corr[16] = 1.6764886672466330E+00
    s_corr[17] = 8.5186113080873460E+04

    check1 = ambit_fe.resultcheck.results_check_vec(problem.mp.pb0.s, s_corr, problem.mp.comm, tol=tol)
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

    test_main()
