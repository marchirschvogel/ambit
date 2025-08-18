#!/usr/bin/env python3

### 3D biventricular generic heart, testing of:
# - Multiscale staggered G&R problem
# - Guccione material
# - active stress
# - 3D-0D monolithic solution of 2D heart w/ syspulcap circulation (volume coupling)

import ambit_fe

import numpy as np
from pathlib import Path
import pytest


@pytest.mark.solid_flow0d_growthremodel
@pytest.mark.skip(reason="Not yet ready for testing.")
def test_main():
    basepath = str(Path(__file__).parent.absolute())

    IO_PARAMS = {
        "problem_type": "solid_flow0d_multiscale_gandr",
        "mesh_domain": basepath + "/input/heart3Dcoarse_domain.xdmf",
        "mesh_boundary": basepath + "/input/heart3Dcoarse_boundary.xdmf",
        "fiber_data": [
            basepath + "/input/fib_fiber_nodal_3Dcoarse.xdmf",
            basepath + "/input/fib_sheet_nodal_3Dcoarse.xdmf",
        ],
        "write_results_every": 1,
        "output_path": basepath + "/tmp",
        "results_to_write": [
            "displacement",
            "theta",
            "phi_remod",
            "trmandelstress_e",
        ],
        "simname": "multiscale_concentric_as",
    }

    SOLVER_PARAMS = {
        "solve_type": "direct",
        "tol_res": [1.0e-8, 1.0e-6],
        "tol_inc": [1.0e-8, 1.0e-6],
        "divergence_continue": "PTC",
        "k_ptc_initial": 10.0,
        "print_local_iter": False,
        "tol_res_local": 1.0e-10,
        "tol_inc_local": 1.0e-10,
    }

    TIME_PARAMS_SOLID_SMALL = {
        "maxtime": 1.0 * 100,
        "numstep": 50 * 100,
        "timint": "genalpha",
        "theta_ost": 1.0,
        "rho_inf_genalpha": 0.8,
    }

    TIME_PARAMS_SOLID_LARGE = {
        "maxtime": 2592000.0,  # 1 month: 30*24*60*60 s
        "numstep": 1000,
        "timint": "static",
    }

    TIME_PARAMS_FLOW0D = {
        "timint": "ost",  # ost
        "theta_ost": 0.5,
        "eps_periodic": 999,
        "periodic_checktype": ["pQvar"],
        "initial_file": basepath + "/input/initial_syspulcap_multiscale.txt",
    }

    MODEL_PARAMS_FLOW0D = {
        "modeltype": "syspulcap",
        "parameters": param(),
        "chamber_models": {
            "lv": {"type": "3D_solid"},
            "rv": {"type": "3D_solid"},
            "la": {"type": "0D_elast", "activation_curve": 2},
            "ra": {"type": "0D_elast", "activation_curve": 2},
        },
        "perturb_type": ["as", 50.0],
        "perturb_after_cylce": 1,
    }

    FEM_PARAMS = {
        "order_disp": 1,
        "order_pres": 1,
        "quad_degree": 1,
        "incompressibility": "no",
        "prestress_initial": True,
        "lin_remodeling_full": False,
    }

    COUPLING_PARAMS = {
        "surface_ids": [[1], [2]],
        "coupling_quantity": ["volume", "volume"],
        "coupling_type": "monolithic_direct",
    }

    MULTISCALE_GR_PARAMS = {
        "gandr_trigger_phase": "end_systole",  # end_diastole, end_systole
        "numcycles": 10,
        "tol_small": 0.08,  # cycle error tolerance: overrides eps_periodic from TIME_PARAMS_FLOW0D
        "tol_large": 5.0e-3,  # growth rate tolerance [mm^3/s]
        "tol_outer": 3.0e-3,
        "write_checkpoints": True,
        "restart_cycle": 0,
        "restart_from_small": False,
    }

    MATERIALS = {
        "MAT1": {
            "guccione_dev": {
                "c_0": 1.662,
                "b_f": 14.31,
                "b_t": 4.49,
                "b_fs": 10.0,
            },
            "sussmanbathe_vol": {"kappa": 1.0e3},
            "active_fiber": {
                "sigma0": 150.0,
                "alpha_max": 10.0,
                "alpha_min": -30.0,
                "activation_curve": 1,
                "frankstarling": True,
                "amp_min": 1.0,
                "amp_max": 1.5,
                "lam_threslo": 1.01,
                "lam_maxlo": 1.15,
                "lam_threshi": 999.0,
                "lam_maxhi": 9999.0,
            },
            "inertia": {"rho0": 1.0e-6},
            "growth": {
                "growth_dir": "crossfiber",
                "growth_trig": "volstress",
                "trigger_reduction": 0.99,
                "growth_thres": 60.0,
                "thres_tol": 1.0e-3,
                "thetamax": 2.0,
                "thetamin": 1.0,
                "tau_gr": 64.0e4,
                "gamma_gr": 2.0,
                "tau_gr_rev": 128.0e4,
                "gamma_gr_rev": 2.0,
                "remodeling_mat": {
                    "guccione_dev": {
                        "c_0": 10 * 1.662,
                        "b_f": 14.31,
                        "b_t": 4.49,
                        "b_fs": 10.0,
                    },
                    "sussmanbathe_vol": {"kappa": 1.0e3},
                    "active_fiber": {
                        "sigma0": 150.0,
                        "alpha_max": 10.0,
                        "alpha_min": -30.0,
                        "activation_curve": 1,
                        "frankstarling": True,
                        "amp_min": 1.0,
                        "amp_max": 1.5,
                        "lam_threslo": 1.01,
                        "lam_maxlo": 1.15,
                        "lam_threshi": 999.0,
                        "lam_maxhi": 9999.0,
                    },
                },
            },
        }
    }

    # define your load curves here (syntax: tcX refers to curve X, to be used in BC_DICT key 'curve' : [X,0,0], or 'curve' : X)
    class time_curves:
        def tc1(self, t):
            K = 5.0
            t_contr, t_relax = 0.2, 0.53

            alpha_max = MATERIALS["MAT1"]["active_fiber"]["alpha_max"]
            alpha_min = MATERIALS["MAT1"]["active_fiber"]["alpha_min"]

            c1 = t_contr + alpha_max / (K * (alpha_max - alpha_min))
            c2 = t_relax - alpha_max / (K * (alpha_max - alpha_min))

            # Diss Hirschvogel eq. 2.101
            return (
                (K * (t - c1) + 1.0) * ((K * (t - c1) + 1.0) > 0.0)
                - K * (t - c1) * ((K * (t - c1)) > 0.0)
                - K * (t - c2) * ((K * (t - c2)) > 0.0)
                + (K * (t - c2) - 1.0) * ((K * (t - c2) - 1.0) > 0.0)
            )

        def tc2(self, t):  # atrial activation
            act_dur = 2.0 * param()["t_ed"]
            t0 = 0.0

            if t >= t0 and t <= t0 + act_dur:
                return 0.5 * (1.0 - np.cos(2.0 * np.pi * (t - t0) / act_dur))
            else:
                return 0.0

    BC_DICT = {
        "robin": [
            {"type": "spring", "id": [3], "dir": "normal_ref", "stiff": 0.075},
            {"type": "dashpot", "id": [3], "dir": "normal_ref", "visc": 0.005},
            {
                "type": "spring",
                "id": [4],
                "dir": "normal_ref",
                "stiff": 10.0,
            },  # 2.5, 1.25
            {
                "type": "dashpot",
                "id": [4],
                "dir": "normal_ref",
                "visc": 0.0005,
            },
            {"type": "spring", "id": [4], "dir": "xyz_ref", "stiff": 0.25},
            {"type": "dashpot", "id": [4], "dir": "xyz_ref", "visc": 0.0005},
        ]
    }

    # problem setup
    problem = ambit_fe.ambit_main.Ambit(
        IO_PARAMS,
        [TIME_PARAMS_SOLID_SMALL, TIME_PARAMS_SOLID_LARGE, TIME_PARAMS_FLOW0D],
        SOLVER_PARAMS,
        FEM_PARAMS,
        [MATERIALS, MODEL_PARAMS_FLOW0D],
        BC_DICT,
        time_curves=time_curves(),
        coupling_params=COUPLING_PARAMS,
        multiscale_params=MULTISCALE_GR_PARAMS,
    )

    # solve time-dependent problem
    problem.solve_problem()

    # --- results check
    tol = 1.0e-6

    s_corr = np.zeros(problem.mp.pb0.cardvasc0D.numdof)

    # correct 0D results
    s_corr[0] = 1.5095864743040130e06
    s_corr[1] = 1.2844204058649662e00
    s_corr[2] = -2.2965082274795714e00
    s_corr[3] = -2.2516606843904677e-01
    s_corr[4] = 1.2035047944220826e01
    s_corr[5] = 7.9266376779754364e03
    s_corr[6] = 1.0920576465746846e01
    s_corr[7] = 4.8757248185773104e04
    s_corr[8] = 4.5874998121783552e04
    s_corr[9] = 3.5972406593836102e04
    s_corr[10] = 2.4558826877860327e04
    s_corr[11] = 8.1860969826562832e03
    s_corr[12] = 2.2677989771675335e00
    s_corr[13] = 8.3769499009227275e02
    s_corr[14] = 2.2702566758390823e00
    s_corr[15] = 9.5189334925959599e02
    s_corr[16] = 2.2702868360248418e00
    s_corr[17] = 7.5312458499319700e02
    s_corr[18] = 2.2701207039868367e00
    s_corr[19] = 5.0027868673701960e02
    s_corr[20] = 2.2705227157170751e00
    s_corr[21] = 1.7138025576859900e02
    s_corr[22] = 2.2541277949292278e00
    s_corr[23] = 2.0733856020336604e05
    s_corr[24] = 9.9000307744360238e04
    s_corr[25] = 2.7306345247149444e-01
    s_corr[26] = -4.3686268354670260e-01
    s_corr[27] = 1.7406314472713416e-01
    s_corr[28] = 2.4017163277669913e00
    s_corr[29] = 1.1803816043509409e04
    s_corr[30] = 2.3131877074406706e00
    s_corr[31] = 2.0066530960234333e05
    s_corr[32] = 1.6159475288856615e00
    s_corr[33] = 3.9878128320907672e04

    check1 = ambit_fe.resultcheck.results_check_vec(problem.mp.pb0.s, s_corr, problem.mp.comm, tol=tol)
    success = ambit_fe.resultcheck.success_check([check1], problem.mp.comm)

    if not success:
        raise RuntimeError("Test failed!")


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

    # atrial elastances (only for 0D atria)
    E_at_A_l, E_at_min_l = 20.0e-6, 9.0e-6
    E_at_A_r, E_at_min_r = 10.0e-6, 8.0e-6

    # timings
    t_ed = 0.2
    t_es = 0.53
    T_cycl = 1.0

    ## systemic arterial
    # now we have to separate the resistance into a proximal and a peripheral part
    frac_Rprox_Rtotal = 0.06  # Ursino et al. factor: 0.06 - OK
    R_arperi_sys = (1.0 - frac_Rprox_Rtotal) * R_ar_sys
    R_ar_sys *= frac_Rprox_Rtotal  # now R_ar_sys(prox)

    frac_Cprox_Ctotal = 0.95  # 0.07 # Ursino et al. factor: 0.07 - XXXX too small???!!!!! - keep in mind that most compliance lies in the aorta / proximal!
    C_arperi_sys = (1.0 - frac_Cprox_Ctotal) * C_ar_sys
    C_ar_sys *= frac_Cprox_Ctotal  # now C_ar_sys(prox)

    # R in parallel:
    # R_arperi_sys = (1/R_arspl_sys + 1/R_arespl_sys + 1/R_armsc_sys + 1/R_arcer_sys + 1/R_arcor_sys)^(-1)
    R_arspl_sys = 3.35 * R_arperi_sys  # Ursino et al. factor: 3.35 - OK
    R_arespl_sys = 3.56 * R_arperi_sys  # Ursino et al. factor: 3.56 - OK
    R_armsc_sys = 4.54 * R_arperi_sys  # Ursino et al. factor: 4.54 - OK
    R_arcer_sys = 6.65 * R_arperi_sys  # Ursino et al. factor: 6.65 - OK
    R_arcor_sys = 19.95 * R_arperi_sys  # Ursino et al. factor: 19.95 - OK

    # C in parallel (fractions have to sum to 1):
    # C_arperi_sys = C_arspl_sys + C_arespl_sys + C_armsc_sys + C_arcer_sys + C_arcor_sys
    C_arspl_sys = 0.55 * C_arperi_sys  # Ursino et al. factor: 0.55 - OK
    C_arespl_sys = 0.18 * C_arperi_sys  # Ursino et al. factor: 0.18 - OK
    C_armsc_sys = 0.14 * C_arperi_sys  # Ursino et al. factor: 0.14 - OK
    C_arcer_sys = 0.11 * C_arperi_sys  # Ursino et al. factor: 0.11 - OK
    C_arcor_sys = 0.03 * C_arperi_sys  # Ursino et al. factor: 0.03 - OK

    ## systemic venous
    frac_Rprox_Rtotal = 0.8  # no Ursino et al. factor since they do not have that extra compartment!
    R_venperi_sys = (1.0 - frac_Rprox_Rtotal) * R_ven_sys
    R_ven_sys *= frac_Rprox_Rtotal  # now R_ven_sys(prox)

    frac_Cprox_Ctotal = 0.2  # no Ursino et al. factor since they do not have that extra compartment!
    C_venperi_sys = (1.0 - frac_Cprox_Ctotal) * C_ven_sys
    C_ven_sys *= frac_Cprox_Ctotal  # now C_ven_sys(prox)

    # R in parallel:
    # R_venperi_sys = (1/R_venspl_sys + 1/R_venespl_sys + 1/R_venmsc_sys + 1/R_vencer_sys + 1/R_vencor_sys)^(-1)
    R_venspl_sys = 3.4 * R_venperi_sys  # Ursino et al. factor: 3.4 - OK
    R_venespl_sys = 3.53 * R_venperi_sys  # Ursino et al. factor: 3.53 - OK
    R_venmsc_sys = 4.47 * R_venperi_sys  # Ursino et al. factor: 4.47 - OK
    R_vencer_sys = 6.66 * R_venperi_sys  # Ursino et al. factor: 6.66 - OK
    R_vencor_sys = 19.93 * R_venperi_sys  # Ursino et al. factor: 19.93 - OK

    # C in parallel (fractions have to sum to 1):
    # C_venperi_sys = C_venspl_sys + C_venespl_sys + C_venmsc_sys + C_vencer_sys + C_vencor_sys
    C_venspl_sys = 0.55 * C_venperi_sys  # Ursino et al. factor: 0.55 - OK
    C_venespl_sys = 0.18 * C_venperi_sys  # Ursino et al. factor: 0.18 - OK
    C_venmsc_sys = 0.14 * C_venperi_sys  # Ursino et al. factor: 0.14 - OK
    C_vencer_sys = 0.1 * C_venperi_sys  # Ursino et al. factor: 0.1 - OK
    C_vencor_sys = 0.03 * C_venperi_sys  # Ursino et al. factor: 0.03 - OK

    ## pulmonary arterial
    frac_Rprox_Rtotal = 0.5  # 0.72  # Ursino et al. factor: 0.72 - hm... doubt that - stick with 0.5
    R_cap_pul = (1.0 - frac_Rprox_Rtotal) * R_ar_pul
    R_ar_pul *= frac_Rprox_Rtotal  # now R_ar_pul(prox)

    ## pulmonary venous
    frac_Cprox_Ctotal = 0.5  # 0.12  # Ursino et al. factor: 0.12 - XXX?: gives shitty p_puls... - stick with 0.5
    C_cap_pul = (1.0 - frac_Cprox_Ctotal) * C_ar_pul
    C_ar_pul *= frac_Cprox_Ctotal  # now C_ar_pul(prox)

    ### unstressed compartment volumes, diffult to estimate - use literature values!
    # these volumes only become relevant for the gas transport models as they determine the capacity of each
    # compartment to store constituents - however, they are also used for postprocessing of the flow models...
    V_at_l_u = 5000.0  # applies only in case of 0D or prescribed atria
    V_at_r_u = 4000.0  # applies only in case of 0D or prescribed atria
    V_v_l_u = 10000.0  # applies only in case of 0D or prescribed ventricles
    V_v_r_u = 8000.0  # applies only in case of 0D or prescribed ventricles
    V_ar_sys_u = 0.0  # Ursino et al. Am J Physiol Heart Circ Physiol (2000), mm^3
    V_ar_pul_u = 0.0  # Ursino et al. Am J Physiol Heart Circ Physiol (2000), mm^3
    V_ven_pul_u = 120.0e3  # Ursino et al. Am J Physiol Heart Circ Physiol (2000), mm^3
    # peripheral systemic arterial
    V_arspl_sys_u = 274.4e3  # Ursino et al. Am J Physiol Heart Circ Physiol (2000), mm^3
    V_arespl_sys_u = 134.64e3  # Ursino et al. Am J Physiol Heart Circ Physiol (2000), mm^3
    V_armsc_sys_u = 105.8e3  # Ursino et al. Am J Physiol Heart Circ Physiol (2000), mm^3
    V_arcer_sys_u = 72.13e3  # Ursino et al. Am J Physiol Heart Circ Physiol (2000), mm^3
    V_arcor_sys_u = 24.0e3  # Ursino et al. Am J Physiol Heart Circ Physiol (2000), mm^3
    # peripheral systemic venous
    V_venspl_sys_u = 1121.0e3  # Ursino et al. Am J Physiol Heart Circ Physiol (2000), mm^3
    V_venespl_sys_u = 550.0e3  # Ursino et al. Am J Physiol Heart Circ Physiol (2000), mm^3
    V_venmsc_sys_u = 432.14e3  # Ursino et al. Am J Physiol Heart Circ Physiol (2000), mm^3
    V_vencer_sys_u = 294.64e3  # Ursino et al. Am J Physiol Heart Circ Physiol (2000), mm^3
    V_vencor_sys_u = 98.21e3  # Ursino et al. Am J Physiol Heart Circ Physiol (2000), mm^3
    V_ven_sys_u = 100.0e3  # estimated (Ursino et al. do not have that extra venous compartment...)
    # pulmonary capillary
    V_cap_pul_u = 123.0e3  # Ursino et al. Am J Physiol Heart Circ Physiol (2000), mm^3

    return {
        "R_ar_sys": R_ar_sys,
        "C_ar_sys": C_ar_sys,
        "L_ar_sys": L_ar_sys,
        "Z_ar_sys": Z_ar_sys,
        "R_arspl_sys": R_arspl_sys,
        "C_arspl_sys": C_arspl_sys,
        "R_arespl_sys": R_arespl_sys,
        "C_arespl_sys": C_arespl_sys,
        "R_armsc_sys": R_armsc_sys,
        "C_armsc_sys": C_armsc_sys,
        "R_arcer_sys": R_arcer_sys,
        "C_arcer_sys": C_arcer_sys,
        "R_arcor_sys": R_arcor_sys,
        "C_arcor_sys": C_arcor_sys,
        "R_venspl_sys": R_venspl_sys,
        "C_venspl_sys": C_venspl_sys,
        "R_venespl_sys": R_venespl_sys,
        "C_venespl_sys": C_venespl_sys,
        "R_venmsc_sys": R_venmsc_sys,
        "C_venmsc_sys": C_venmsc_sys,
        "R_vencer_sys": R_vencer_sys,
        "C_vencer_sys": C_vencer_sys,
        "R_vencor_sys": R_vencor_sys,
        "C_vencor_sys": C_vencor_sys,
        "R_ar_pul": R_ar_pul,
        "C_ar_pul": C_ar_pul,
        "L_ar_pul": L_ar_pul,
        "Z_ar_pul": Z_ar_pul,
        "R_cap_pul": R_cap_pul,
        "C_cap_pul": C_cap_pul,
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
        "E_v_max_r": 8.0e-5,
        "E_v_min_r": 1.0e-5,
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
        "T_cycl": T_cycl,
        # unstressed compartment volumes (for post-processing)
        "V_at_l_u": V_at_l_u,
        "V_at_r_u": V_at_r_u,
        "V_v_l_u": V_v_l_u,
        "V_v_r_u": V_v_r_u,
        "V_ar_sys_u": V_ar_sys_u,
        "V_arspl_sys_u": V_arspl_sys_u,
        "V_arespl_sys_u": V_arespl_sys_u,
        "V_armsc_sys_u": V_armsc_sys_u,
        "V_arcer_sys_u": V_arcer_sys_u,
        "V_arcor_sys_u": V_arcor_sys_u,
        "V_venspl_sys_u": V_venspl_sys_u,
        "V_venespl_sys_u": V_venespl_sys_u,
        "V_venmsc_sys_u": V_venmsc_sys_u,
        "V_vencer_sys_u": V_vencer_sys_u,
        "V_vencor_sys_u": V_vencor_sys_u,
        "V_ven_sys_u": V_ven_sys_u,
        "V_ar_pul_u": V_ar_pul_u,
        "V_cap_pul_u": V_cap_pul_u,
        "V_ven_pul_u": V_ven_pul_u,
    }


if __name__ == "__main__":
    test_main()
