#!/usr/bin/env python3

### 2D biventricular generic heart, testing of:
# - Holzapfel-Ogden material
# - active stress read from files
# - 3D-0D monolithic solution of 2D heart w/ syspul circulation (volume coupling) and coronary model ZCRp_CRd_lr
# - Robin BCs in normal direction (spring and dashpot)
# - Gen-Alpha time-integration for solid

import ambit_fe

import sys
import numpy as np
from pathlib import Path
import pytest


@pytest.mark.solid_flow0d
@pytest.mark.rom
def test_main():
    basepath = str(Path(__file__).parent.absolute())

    # reads in restart step from the command line
    try:
        restart_step = int(sys.argv[1])
    except:
        restart_step = 0

    IO_PARAMS = {
        "problem_type": "solid_flow0d",
        "mesh_domain": basepath + "/input/heart2D_domain.xdmf",
        "mesh_boundary": basepath + "/input/heart2D_boundary.xdmf",
        "fiber_data": [
            basepath + "/input/fib_fiber_nodal_2D.xdmf",
            basepath + "/input/fib_sheet_nodal_2D.xdmf",
        ],
        "write_results_every": -999,
        "output_path": basepath + "/tmp/",
        "results_to_write": ["displacement"],
        "simname": "ROM_2Dheart",
        "write_restart_every": -999,
        "restart_step": restart_step,
        "ode_parallel": True,
    }

    CONTROL_PARAMS = {"maxtime": 1.0, "numstep": 500, "numstep_stop": 5}

    ROM_PARAMS = {
        "hdmfilenames": [basepath + "/input/checkpoint_romsnaps_2Dheart_u_*.xdmf"],
        "numsnapshots": 5,
        "snapshotincr": 1,
        "numredbasisvec": 5,
        "eigenvalue_cutoff": 1.0e-8,
        "print_eigenproblem": True,
        "write_pod_modes": True,
    }

    SOLVER_PARAMS = {
        "solve_type": "direct",
        "tol_res": [1.0e-8, 1.0e-6],
        "tol_inc": [1.0e-8, 1.0e-6],
    }

    TIME_PARAMS_SOLID = {"timint": "genalpha", "rho_inf_genalpha": 0.8}

    TIME_PARAMS_FLOW0D = {
        "timint": "ost",
        "theta_ost": 0.5,
        "initial_conditions": init(),
    }

    MODEL_PARAMS_FLOW0D = {
        "modeltype": "syspul",
        "coronary_model": "ZCRp_CRd_lr",
        "parameters": param(),
        "chamber_models": {
            "lv": {"type": "3D_solid"},
            "rv": {"type": "3D_solid"},
            "la": {"type": "0D_elast", "activation_curve": 1},
            "ra": {"type": "0D_elast", "activation_curve": 1},
        },
    }

    FEM_PARAMS = {
        "order_disp": 1,
        "order_pres": 1,
        "quad_degree": 2,
        "incompressibility": "no",
    }

    COUPLING_PARAMS = {
        "surface_ids": [[1], [2]],
        "cq_factor": [80.0, 80.0],
        "coupling_quantity": ["volume"] * 2,
        "coupling_type": "monolithic_direct",
    }

    MATERIALS = {
        "MAT1": {
            "holzapfelogden_dev": {
                "a_0": 0.059,
                "b_0": 8.023,
                "a_f": 18.472,
                "b_f": 16.026,
                "a_s": 2.481,
                "b_s": 11.120,
                "a_fs": 0.216,
                "b_fs": 11.436,
            },
            "sussmanbathe_vol": {"kappa": 1.0e3},
            "active_fiber": {
                "type": "prescribed_from_file",
                "prescribed_file": basepath + "/input/checkpoint_ROM_2Dheart_solid_tau_a_*.xdmf",
            },
            "inertia": {"rho0": 1.0e-6},
        }
    }

    # define your load curves here (syntax: tcX refers to curve X, to be used in BC_DICT key 'curve' : [X,0,0], or 'curve' : X)
    class time_curves:
        def tc1(self, t):  # atrial activation
            act_dur = 2.0 * param()["t_ed"]
            t0 = 0.0

            if t >= t0 and t <= t0 + act_dur:
                return 0.5 * (1.0 - np.cos(2.0 * np.pi * (t - t0) / act_dur))
            else:
                return 0.0

    BC_DICT = {
        "dirichlet": [{"dir": "2dimZ", "val": 0.0}],
        "robin": [
            {"type": "spring", "id": [3], "dir": "normal_ref", "stiff": 0.075},
            {"type": "dashpot", "id": [3], "dir": "normal_ref", "visc": 0.005},
        ],
    }

    # problem setup
    problem = ambit_fe.ambit_main.Ambit(
        IO_PARAMS,
        CONTROL_PARAMS,
        [TIME_PARAMS_SOLID, TIME_PARAMS_FLOW0D],
        SOLVER_PARAMS,
        FEM_PARAMS,
        [MATERIALS, MODEL_PARAMS_FLOW0D],
        BC_DICT,
        time_curves=time_curves(),
        coupling_params=COUPLING_PARAMS,
        mor_params=ROM_PARAMS,
    )

    # solve time-dependent problem
    problem.solve_problem()

    # --- results check
    tol = 1.0e-6

    s_corr = np.zeros(problem.mp.pb0.cardvasc0D.numdof)

    # correct 0D results
    s_corr[0] = 1.0072498846634732e05
    s_corr[1] = 3.8514559531428966e-01
    s_corr[2] = -1.0581253546863147e00
    s_corr[3] = 2.8442060684794246e-01
    s_corr[4] = 1.0865674153711085e01
    s_corr[5] = -7.3632748798559376e04
    s_corr[6] = 1.0874047381615156e01
    s_corr[7] = 7.1800294544126518e04
    s_corr[8] = 2.2872593107992092e00
    s_corr[9] = 8.4945733706216852e04
    s_corr[10] = 6.2553713813202026e04
    s_corr[11] = 2.4856170185000773e-01
    s_corr[12] = -2.0261424856731150e-01
    s_corr[13] = 1.8600798803680565e-01
    s_corr[14] = 2.2121504737099205e00
    s_corr[15] = 3.5717308623102937e04
    s_corr[16] = 1.6763908443633773e00
    s_corr[17] = 8.6083016603272583e04
    s_corr[18] = 6.9671229506558996e02
    s_corr[19] = 2.9188950634144771e03
    s_corr[20] = 4.4437756391412614e-01
    s_corr[21] = -3.5320240534142283e-01
    s_corr[22] = 6.9671229506558996e02
    s_corr[23] = 2.9188950634144771e03
    s_corr[24] = 4.4437756391412614e-01
    s_corr[25] = -3.5320240534142283e-01
    s_corr[26] = -8.5023349891608843e04

    check1 = ambit_fe.resultcheck.results_check_vec(problem.mp.pb0.s, s_corr, problem.mp.comm, tol=tol)
    success = ambit_fe.resultcheck.success_check([check1], problem.mp.comm)

    if not success:
        raise RuntimeError("Test failed!")


def init():
    return {
        "q_vin_l_0": 1.1549454594333263e04,
        "p_at_l_0": 3.8580961077622145e-01,
        "q_vout_l_0": -1.0552685263595845e00,
        "p_v_l_0": 3.7426015618188813e-01,
        "p_ar_sys_0": 1.0926945419777734e01,
        "q_ar_sys_0": 7.2237210814547114e04,
        "p_ven_sys_0": 2.2875736545217800e00,
        "q_ven_sys_0": 8.5022643486798144e04,
        "q_vin_r_0": 4.1097788677528049e04,
        "p_at_r_0": 2.4703021083862464e-01,
        "q_vout_r_0": -2.0242075369768467e-01,
        "p_v_r_0": 2.0593242216109664e-01,
        "p_ar_pul_0": 2.2301399591379436e00,
        "q_ar_pul_0": 3.6242987765574515e04,
        "p_ven_pul_0": 1.6864951426543255e00,
        "q_ven_pul_0": 8.6712368791873596e04,
        # coronary circulation submodel
        "q_cor_sys_l_0": 0,
        "q_cord_sys_l_0": 0,
        "q_corp_sys_l_0": 0.0,
        "p_cord_sys_l_0": 0.0,
        "q_cor_sys_r_0": 0,
        "q_cord_sys_r_0": 0,
        "q_corp_sys_r_0": 0.0,
        "p_cord_sys_r_0": 0.0,
    }


def param():
    R_ar_sys = 120.0e-6
    tau_ar_sys = 1.65242332
    tau_ar_pul = 0.3

    # Diss Hirschvogel tab. 2.7
    C_ar_sys = tau_ar_sys / R_ar_sys
    Z_ar_sys = R_ar_sys / 20.0
    R_ven_sys = R_ar_sys / 5.0
    C_ven_sys = 30.0 * C_ar_sys
    R_ar_pul = R_ar_sys / 8.0
    C_ar_pul = tau_ar_pul / R_ar_pul
    Z_ar_pul = 0.0
    R_ven_pul = R_ar_pul
    C_ven_pul = 2.5 * C_ar_pul

    L_ar_sys = 0.667e-6
    L_ven_sys = 0.0
    L_ar_pul = 0.0
    L_ven_pul = 0.0

    # atrial elastances
    E_at_A_l, E_at_min_l = 20.0e-6, 9.0e-6
    E_at_A_r, E_at_min_r = 10.0e-6, 8.0e-6

    # timings
    t_ed = 0.2
    t_es = 0.53

    return {
        "R_ar_sys": R_ar_sys,
        "C_ar_sys": C_ar_sys,
        "L_ar_sys": L_ar_sys,
        "Z_ar_sys": Z_ar_sys,
        "R_ar_pul": R_ar_pul,
        "C_ar_pul": C_ar_pul,
        "L_ar_pul": L_ar_pul,
        "Z_ar_pul": Z_ar_pul,
        "R_ven_sys": R_ven_sys,
        "C_ven_sys": C_ven_sys,
        "L_ven_sys": L_ven_sys,
        "R_ven_pul": R_ven_pul,
        "C_ven_pul": C_ven_pul,
        "L_ven_pul": L_ven_pul,
        # atrial elastances
        "E_at_max_l": E_at_min_l + E_at_A_l,
        "E_at_min_l": E_at_min_l,
        "E_at_max_r": E_at_min_r + E_at_A_r,
        "E_at_min_r": E_at_min_r,
        # ventricular elastances
        "E_v_max_l": 7.0e-5,
        "E_v_min_l": 12.0e-6,
        "E_v_max_r": 3.0e-5,
        "E_v_min_r": 10.0e-6,
        # valve resistances
        "R_vin_l_min": 1.0e-6,
        "R_vin_l_max": 1.0e1,
        "R_vout_l_min": 1.0e-6,
        "R_vout_l_max": 1.0e1,
        "R_vin_r_min": 1.0e-6,
        "R_vin_r_max": 1.0e1,
        "R_vout_r_min": 1.0e-6,
        "R_vout_r_max": 1.0e1,
        # timings
        "t_ed": t_ed,
        "t_es": t_es,
        "T_cycl": 1.0,
        # coronary circulation submodel parameters - values from Arthurs et al. 2016, Tab. 3
        "Z_corp_sys_l": 3.2e-3,
        "C_corp_sys_l": 4.5e0,
        "R_corp_sys_l": 6.55e-3,
        "C_cord_sys_l": 2.7e1,
        "R_cord_sys_l": 1.45e-1,
        "Z_corp_sys_r": 3.2e-3,
        "C_corp_sys_r": 4.5e0,
        "R_corp_sys_r": 6.55e-3,
        "C_cord_sys_r": 2.7e1,
        "R_cord_sys_r": 1.45e-1,
    }


if __name__ == "__main__":
    test_main()
