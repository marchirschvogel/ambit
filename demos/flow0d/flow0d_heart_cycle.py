#!/usr/bin/env python3

"""
A closed-loop lumped-parameter (0D) systemic and pulmonary circulation model, where the heart chambers are represented as time-varying elastance models.
"""

import ambit_fe
import numpy as np
from pathlib import Path


def main():
    basepath = str(Path(__file__).parent.absolute())

    """
    Parameters for input/output
    """
    IO_PARAMS = {  # problem type 'flow0d' indicates a pure ODE problem of pressure-flow relationships
        "problem_type": "flow0d",
        # at which step frequency to write results (set to 0 in order to not write any output)
        "write_results_every": 1,
        # where to write the output to
        "output_path": basepath + "/tmp",
        # the 'midfix' for all simulation result file names: will be results_<simname>_<field>.txt
        "simname": "flow0d_heart_cycle",
    }

    number_of_cycles = 10
    """
    Parameters for the global time control
    """
    CONTROL_PARAMS = {
        "maxtime": number_of_cycles * 1.0,
        "numstep": number_of_cycles * 100,
    }

    """
    Parameters for the nonlinear solution scheme (Newton solver)
    """
    SOLVER_PARAMS = {  # residual and increment tolerances
        "tol_res": 1.0e-8,
        "tol_inc": 1.0e-8,
    }

    """
    Parameters for the 0D model time integration scheme
    """
    TIME_PARAMS = {  # the 0D model time integration scheme: we use a One-Step-theta method with theta = 0.5, which corresponds to the trapezoidal rule
        "timint": "ost",
        "theta_ost": 0.5,
        # do initial time step using backward scheme (theta=1), to avoid fluctuations for quantities whose d/dt is zero
        "initial_backwardeuler": True,
        # the initial conditions of the 0D ODE model (defined below)
        "initial_conditions": init(),
        # the periodic state criterion tolerance
        "eps_periodic": 0.05,
        # which variables to check for periodicity (default, 'allvar')
        "periodic_checktype": ["allvar"],
    }

    MODEL_PARAMS = {  # the type of 0D model: 'syspul' refers to the closed-loop systemic+pulmonary circulation model
        "modeltype": "syspul",
        # the parameters of the 0D model (defined below)
        "parameters": param(),
        # models for cardiac chambers: all four are 0D time-varying elastance models, activated with different time curves (curve no. 2 for ventricles, curve no. 1 for atria, cf. below)
        "chamber_models": {
            "lv": {"type": "0D_elast", "activation_curve": 2},
            "rv": {"type": "0D_elast", "activation_curve": 2},
            "la": {"type": "0D_elast", "activation_curve": 1},
            "ra": {"type": "0D_elast", "activation_curve": 1},
        },
        # models for valves: all are piecewise-linear pressure-dependent ('pwlin_pres', default)
        "valvelaws": {
            "av": ["pwlin_pres"],  # aortic valve
            "mv": ["pwlin_pres"],  # mitral valve
            "pv": ["pwlin_pres"],  # pulmonary valve
            "tv": ["pwlin_pres"],
        },
    }  # tricuspid valve

    # define your time curves here (syntax: tcX refers to curve X)
    class time_curves:
        # the activation curves for the contraction of the 0D atria
        def tc1(self, t):
            tmod = t % param()["T_cycl"]

            act_dur = 2.0 * param()["t_ed"]
            t0 = 0.0

            if tmod >= t0 and tmod <= t0 + act_dur:
                return 0.5 * (1.0 - np.cos(2.0 * np.pi * (tmod - t0) / act_dur))
            else:
                return 0.0

        # the activation curves for the contraction of the 0D ventricles
        def tc2(self, t):
            tmod = t % param()["T_cycl"]

            act_dur = 1.8 * (param()["t_es"] - param()["t_ed"])
            t0 = param()["t_ed"]

            if tmod >= t0 and tmod <= t0 + act_dur:
                return 0.5 * (1.0 - np.cos(2.0 * np.pi * (tmod - t0) / act_dur))
            else:
                return 0.0

    # problem setup
    problem = ambit_fe.ambit_main.Ambit(
        IO_PARAMS,
        CONTROL_PARAMS,
        TIME_PARAMS,
        SOLVER_PARAMS,
        constitutive_params=MODEL_PARAMS,
        time_curves=time_curves(),
    )

    # solve time-dependent problem
    problem.solve_problem()


def init():
    # values in kg-mm-s unit system

    return {
        "q_vin_l_0": 0.0,  # initial left ventricular in-flow
        "p_at_l_0": 0.599950804034,  # initial left atrial pressure
        "q_vout_l_0": 0.0,  # initial left ventricular out-flow
        "p_v_l_0": 0.599950804034,  # initial left ventricular pressure
        "p_ar_sys_0": 9.68378038166,  # initial systemic arterial pressure
        "q_ar_sys_0": 0.0,  # initial systemic arterial flux
        "p_ven_sys_0": 2.13315841434,  # initial systemic venous pressure
        "q_ven_sys_0": 0.0,  # initial systemic venous flux
        "q_vin_r_0": 0.0,  # initial right ventricular in-flow
        "p_at_r_0": 0.0933256806275,  # initial right atrial pressure
        "q_vout_r_0": 0.0,  # initial right ventricular out-flow
        "p_v_r_0": 0.0933256806275,  # initial right ventricular pressure
        "p_ar_pul_0": 3.22792679389,  # initial pulmonary arterial pressure
        "q_ar_pul_0": 0.0,  # initial pulmonary arterial flux
        "p_ven_pul_0": 1.59986881076,  # initial pulmonary venous pressure
        "q_ven_pul_0": 0.0,
    }  # initial pulmonary venous flux


def param():
    # parameters in kg-mm-s unit system

    R_ar_sys = 120.0e-6  # systemic arterial resistance
    tau_ar_sys = 1.0311433159  # systemic arterial Windkessel time constant
    tau_ar_pul = 0.3  # pulmonary arterial resistance

    # Diss Hirschvogel tab. 2.7
    C_ar_sys = tau_ar_sys / R_ar_sys  # systemic arterial compliance
    Z_ar_sys = R_ar_sys / 20.0  # systemic arterial characteristic impedance
    R_ven_sys = R_ar_sys / 5.0  # systemic venous resistance
    C_ven_sys = 30.0 * C_ar_sys  # systemic venous compliance
    R_ar_pul = R_ar_sys / 8.0  # pulmonary arterial resistance
    C_ar_pul = tau_ar_pul / R_ar_pul  # pulmonary arterial compliance
    R_ven_pul = R_ar_pul  # pulmonary venous resistance
    C_ven_pul = 2.5 * C_ar_pul  # pulmonary venous resistance

    L_ar_sys = 0.667e-6  # systemic arterial inertance
    L_ven_sys = 0.0  # systemic venous inertance
    L_ar_pul = 0.0  # pulmonary arterial inertance
    L_ven_pul = 0.0  # pulmonary venous inertance

    # timings
    t_ed = 0.2  # end-diastolic time
    t_es = 0.53  # end-systolic time
    T_cycl = 1.0  # cardiac cycle time

    # atrial elastances
    E_at_max_l = 2.9e-5  # maximum left atrial elastance
    E_at_min_l = 9.0e-6  # minimum left atrial elastance
    E_at_max_r = 1.8e-5  # maximum right atrial elastance
    E_at_min_r = 8.0e-6  # minimum right atrial elastance
    # ventricular elastances
    E_v_max_l = 30.0e-5  # maximum left ventricular elastance
    E_v_min_l = 12.0e-6  # minimum left ventricular elastance
    E_v_max_r = 20.0e-5  # maximum right ventricular elastance
    E_v_min_r = 10.0e-6  # minimum right ventricular elastance

    return {
        "R_ar_sys": R_ar_sys,
        "C_ar_sys": C_ar_sys,
        "L_ar_sys": L_ar_sys,
        "Z_ar_sys": Z_ar_sys,
        "R_ar_pul": R_ar_pul,
        "C_ar_pul": C_ar_pul,
        "L_ar_pul": L_ar_pul,
        "R_ven_sys": R_ven_sys,
        "C_ven_sys": C_ven_sys,
        "L_ven_sys": L_ven_sys,
        "R_ven_pul": R_ven_pul,
        "C_ven_pul": C_ven_pul,
        "L_ven_pul": L_ven_pul,
        # atrial elastances
        "E_at_max_l": E_at_max_l,
        "E_at_min_l": E_at_min_l,
        "E_at_max_r": E_at_max_r,
        "E_at_min_r": E_at_min_r,
        # ventricular elastances
        "E_v_max_l": E_v_max_l,
        "E_v_min_l": E_v_min_l,
        "E_v_max_r": E_v_max_r,
        "E_v_min_r": E_v_min_r,
        # valve resistances
        "R_vin_l_min": 1.0e-6,  # mitral valve open resistance
        "R_vin_l_max": 1.0e1,  # mitral valve closed resistance
        "R_vout_l_min": 1.0e-6,  # aortic valve open resistance
        "R_vout_l_max": 1.0e1,  # aortic valve closed resistance
        "R_vin_r_min": 1.0e-6,  # tricuspid valve open resistance
        "R_vin_r_max": 1.0e1,  # tricuspid valve closed resistance
        "R_vout_r_min": 1.0e-6,  # pulmonary valve open resistance
        "R_vout_r_max": 1.0e1,  # pulmonary valve closed resistance
        # timings
        "t_ed": t_ed,
        "t_es": t_es,
        "T_cycl": T_cycl,
        # unstressed compartment volumes (only for post-processing, since 0D model is formulated in fluxes = dVolume/dt)
        "V_at_l_u": 5e3,  # unstressed left atrial volume
        "V_at_r_u": 4e3,  # unstressed right atrial volume
        "V_v_l_u": 10e3,  # unstressed left ventricular volume
        "V_v_r_u": 8e3,  # unstressed right ventricular volume
        "V_ar_sys_u": 611e3,  # unstressed systemic arterial volume
        "V_ar_pul_u": 123e3,  # unstressed pulmonary arterial volume
        "V_ven_sys_u": 2596e3,  # unstressed systemic venous volume
        "V_ven_pul_u": 120e3,
    }  # unstressed pulmonary venous volume


if __name__ == "__main__":
    main()
