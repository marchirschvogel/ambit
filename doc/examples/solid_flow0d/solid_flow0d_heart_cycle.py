#!/usr/bin/env python3

"""
A solid mechanics biventricular heart model coupled to a closed-loop lumped-parameter (0D) systemic and pulmonary circulation model.
"""

import ambit_fe
import numpy as np
from pathlib import Path


def main():

    basepath = str(Path(__file__).parent.absolute())


    """
    Parameters for input/output
    """
    IO_PARAMS            = {# problem type 'solid_flow0d' indicates a coupling of the individual problems 'solid' and 'flow0d'
                            'problem_type'          : 'solid_flow0d',
                            # the meshes for the domain and boundary topology are specified separately
                            'mesh_domain'           : basepath+'/input/heart3D_domain.xdmf',
                            'mesh_boundary'         : basepath+'/input/heart3D_boundary.xdmf',
                            # since we use fiber orientations in both the passive as well as active model, we need to specify those fields
                            'fiber_data'            : [basepath+'/input/fib_fiber_indices_nodal.txt',
                                                       basepath+'/input/fib_sheet_indices_nodal.txt'],
                            'order_fib_input'       : 1,
                            # at which step frequency to write results (set to 0 in order to not write any output)
                            'write_results_every'   : 1,
                            # where to write the output to
                            'output_path'           : basepath+'/tmp/',
                            # which results to write: here, all 3D fields need to be specified, while the 0D model results are output nevertheless
                            'results_to_write'      : ['displacement','fibers'],
                            # the 'midfix' for all simulation result file names: will be results_<simname>_<field>.xdmf/.h5/.txt
                            'simname'               : 'solid_flow0d_heart_cycle'}
                      
    """
    Parameters for the linear and nonlinear solution schemes
    """
    SOLVER_PARAMS        = {# this specifies which linear solution strategy to use; since this problem has less than 10'000 degrees of freedom, we comfortably can use a direct solver
                            'solve_type'            : 'direct',
                            # residual and increment tolerances: first value for the solid mechanics problem (momentum balance), second value for the 3D-0D constraint problem
                            'tol_res'               : [1e-8, 1e-6],
                            'tol_inc'               : [1e-8, 1e-6],
                            # subsolver tolerances for the 0D model solver: tolerances for residual and increment
                            'subsolver_params'      : {'tol_res' : 1e-6,
                                                       'tol_inc' : 1e-6}}

    number_of_cycles = 1
    """
    Parameters for the solid mechanics time integration scheme, plus the global time parameters
    """
    TIME_PARAMS_SOLID    = {# the maximum simulation time - here, we want our heart cycle to last 1 s
                            'maxtime'               : number_of_cycles*1.0,
                            # the number of time steps which the simulation time is divided into - so here, a time step lasts 1/500 s = 0.002 s = 2 ms
                            'numstep'               : number_of_cycles*500,
                            # the solid mechanics time integration scheme: we use the Generalized-alpha method with a spectral radius of 0.8
                            'timint'                : 'genalpha',
                            'rho_inf_genalpha'      : 0.8}

    """
    Parameters for the 0D model time integration scheme
    """
    TIME_PARAMS_FLOW0D   = {# the 0D model time integration scheme: we use a One-Step-theta method with theta = 0.5, which corresponds to the trapezoidal rule
                            'timint'                : 'ost',
                            'theta_ost'             : 0.5,
                            # do initial time step using backward scheme (theta=1), to avoid fluctuations for quantities whose d/dt is zero
                            'initial_backwardeuler' : True,
                            # the initial conditions of the 0D ODE model (defined below)
                            'initial_conditions'    : init(),
                            # the periodic state criterion tolerance
                            'eps_periodic'          : 0.05,
                            # which variables to check for periodicity (default, 'allvar')
                            'periodic_checktype'    : ['allvar']}

    """
    Parameters for the 0D model
    """
    MODEL_PARAMS_FLOW0D  = {# the type of 0D model: 'syspul' refers to the closed-loop systemic+pulmonary circulation model
                            'modeltype'             : 'syspul',
                            # the parameters of the 0D model (defined below)
                            'parameters'            : param(),
                            # The 0D model is setup in a modular fashion which allows each of the four cardiac chambers to be either modeled as purely 0D (time-varying elastance models),
                            # as a 3D solid mechanics chamber, or as a 3D fluid domain.
                            # Since we have a biventricular heart model, our left and right ventricle (lv and rv, respectively) are of type '3D_solid', whereas the left and right atrium
                            # (la and ra, respectively) are treated as 0D chambers. Their elastance are controlled by time curve no. 2 (cf. below), which mimics the atrial systole.
                            'chamber_models'        : {'lv' : {'type' : '3D_solid'},
                                                       'rv' : {'type' : '3D_solid'},
                                                       'la' : {'type' : '0D_elast', 'activation_curve' : 2},
                                                       'ra' : {'type' : '0D_elast', 'activation_curve' : 2}}}
    """
    Finite element parameters and parameters relating to the prestress
    """
    FEM_PARAMS           = {# the order of the finite element ansatz functions for the displacement
                            'order_disp'            : 1,
                            # the quadrature degree (should be > 1 but can be only 2 here for linear tetrahedral finite elements)
                            'quad_degree'           : 2,
                            # whether we want to model the heart as purely incompressible (would involve a 2-field functional with additional pressure degrees of freedom)
                            'incompressible_2field' : False,
                            # the prestress settings: initial prestressing with the MULF method using 5 load steps and PTC (pseudo-transient continuation) for more stable load stepping
                            'prestress_initial'     : True,
                            'prestress_numstep'     : 5,
                            'prestress_ptc'         : True}

    """
    3D-0D coupling parameters
    """
    COUPLING_PARAMS      = {# the surfaces IDs which couple to the 0D world: 1 is the left ventricle, 2 is the right (order here would be lv, rv, la, ra)
                            'surface_ids'           : [[1],[2]],
                            # the coupling type: 'monolithic_lagrange' here is a more general scheme that enforces equality of 3D and 0D fluxes/volumes via (Lagrange) multipliers, outsourcing the 0D
                            # solve to a sub-solver in each nonlinear iteration, whereas 'monolithic_direct' would embed the 0D system of equations directly into the monolithic system matrix
                            'coupling_type'         : 'monolithic_lagrange',
                            # for 'monolithic_lagrange', we need the pressures to be the exchanged quantities between 3D and 0D world (being the Lagrange multipliers)
                            'coupling_quantity'     : ['pressure','pressure'],
                            # the coupling variables which are enforced between 3D and 0D: can be volume or flux
                            'variable_quantity'     : ['flux','flux']}

    """
    Constitutive parameters for the 3D solid mechanics model of the heart
    """
    MATERIALS            = {# Here, we only have one region defined (hence only MAT1) - while, in principal, each element could be associated to a different material.
                                      # the classical Holzapfel-Ogden material for passive myocardium (Holzapfel and Ogden 2009), in its deviatoric version (only for the isotropic part)
                            'MAT1' : {'holzapfelogden_dev' : {'a_0' : 0.059,
                                                              'b_0' : 8.023,
                                                              'a_f' : 18.472,
                                                              'b_f' : 16.026,
                                                              'a_s' : 2.481,
                                                              'b_s' : 11.120,
                                                              'a_fs' : 0.216,
                                                              'b_fs' : 11.436},
                                      # a volumetric material penalizing volume changes, yielding a nearly incompressible behavior
                                      'sussmanbathe_vol'  : {'kappa' : 1e3},
                                      # a strain rate-dependent material
                                      'visco_green'       : {'eta' : 0.1},
                                      # the active stress law that is exerted along the muscle fiber direction and controlled by time curve no. 1 (cf. below)
                                      # this model is reponsible for the contraction of the heart
                                      'active_fiber'      : {'sigma0' : 100.0, 'alpha_max' : 15.0, 'alpha_min' : -30.0, 'activation_curve' : 1},
                                      # the inertia: density
                                      'inertia'           : {'rho0' : 1e-6}}}


    """
    Time curves, e.g. any prescribed time-controlled/-varying loads or functions
    """
    class time_curves():

        # the activation curve for the contraction of the 3D heart ventricles
        def tc1(self, t):
            
            tmod = t % param()['T_cycl']

            K = 5.
            t_contr, t_relax = 0.2, 0.53

            alpha_max = MATERIALS['MAT1']['active_fiber']['alpha_max']
            alpha_min = MATERIALS['MAT1']['active_fiber']['alpha_min']

            c1 = t_contr + alpha_max/(K*(alpha_max-alpha_min))
            c2 = t_relax - alpha_max/(K*(alpha_max-alpha_min))

            # Diss Hirschvogel eq. 2.101
            return (K*(tmod-c1)+1.)*((K*(tmod-c1)+1.)>0.) - K*(tmod-c1)*((K*(tmod-c1))>0.) - K*(tmod-c2)*((K*(tmod-c2))>0.) + (K*(tmod-c2)-1.)*((K*(tmod-c2)-1.)>0.)

        # the activation curves for the contraction of the 0D atria
        def tc2(self, t):
            
            tmod = t % param()['T_cycl']

            act_dur = 2.*param()['t_ed']
            t0 = 0.

            if tmod >= t0 and tmod <= t0 + act_dur:
                return 0.5*(1.-np.cos(2.*np.pi*(tmod-t0)/act_dur))
            else:
                return 0.0

        # The curves that contoll the prestress in each ventricular chamber: Note that we need a negative sign, since the pressure
        # acts against the surface outward normal.
        # We ramp up these loads linearly during the 5 prestress steps. Since the prestress phase is quasi-static, the results only
        # depend on the end value, not on the type of ramp-up. Hence, any ramp-up that converges can be used.
        def tc3(self, t): # LV
            return -init()['p_v_l_0']*t

        def tc4(self, t): # RV
            return -init()['p_v_r_0']*t


    """
    Boundary conditions
    """
    BC_DICT              = { # the prestress boundary conditions, using time curves no. 3 and 4
                             'neumann_prestress' : [{'id' : [1], 'dir' : 'normal_ref', 'curve' : 3},
                                                    {'id' : [2], 'dir' : 'normal_ref', 'curve' : 4}],
                             # Robin conditions (springs and dashpots) are applied to the heart's epicardium (surface ID 3) as well as to the base (surface ID 4),
                             # either acting in reference normal direction ('normal_ref') or in reference cartesian directions ('xyz_ref').
                             # The parameters are chosen according to Hirschvogel 2018 (PhD Thesis), which yielded acceptable results regarding base and epicardial
                             # movement.
                             'robin' : [{'type' : 'spring',  'id' : [3], 'dir' : 'normal_ref', 'stiff' : 0.075},
                                        {'type' : 'dashpot', 'id' : [3], 'dir' : 'normal_ref', 'visc'  : 0.005},
                                        {'type' : 'spring',  'id' : [4], 'dir' : 'normal_ref', 'stiff' : 2.5},
                                        {'type' : 'dashpot', 'id' : [4], 'dir' : 'normal_ref', 'visc'  : 0.0005},
                                        {'type' : 'spring',  'id' : [4], 'dir' : 'xyz_ref', 'stiff' : 0.25},
                                        {'type' : 'dashpot', 'id' : [4], 'dir' : 'xyz_ref', 'visc'  : 0.0005}] }

    # Pass parameters to Ambit to set up the problem
    problem = ambit_fe.ambit_main.Ambit(IO_PARAMS, [TIME_PARAMS_SOLID, TIME_PARAMS_FLOW0D], SOLVER_PARAMS, FEM_PARAMS, [MATERIALS, MODEL_PARAMS_FLOW0D], BC_DICT, time_curves=time_curves(), coupling_params=COUPLING_PARAMS)

    # Call the Ambit solver to solve the problem
    problem.solve_problem()



def init():
    
    # values in kg-mm-s unit system

    return {'Q_v_l_0' : 0.0,               # initial left ventricular flux
            'q_vin_l_0' : 0.0,             # initial left ventricular in-flow
            'p_at_l_0' : 0.599950804034,   # initial left atrial pressure
            'q_vout_l_0' : 0.0,            # initial left ventricular out-flow
            'p_v_l_0' : 0.599950804034,    # initial left ventricular pressure (will be initial value of Lagrange multiplier!)
            'p_ar_sys_0' : 9.68378038166,  # initial systemic arterial pressure
            'q_ar_sys_0' : 0.0,            # initial systemic arterial flux
            'p_ven_sys_0' : 2.13315841434, # initial systemic venous pressure
            'q_ven_sys_0' : 0.0,           # initial systemic venous flux
            'Q_v_r_0' : 0.0,               # initial right ventricular flux
            'q_vin_r_0' : 0.0,             # initial right ventricular in-flow
            'p_at_r_0' : 0.0933256806275,  # initial right atrial pressure
            'q_vout_r_0' : 0.0,            # initial right ventricular out-flow
            'p_v_r_0' : 0.0933256806275,   # initial right ventricular pressure (will be initial value of Lagrange multiplier!)
            'p_ar_pul_0' : 3.22792679389,  # initial pulmonary arterial pressure
            'q_ar_pul_0' : 0.0,            # initial pulmonary arterial flux
            'p_ven_pul_0' : 1.59986881076, # initial pulmonary venous pressure
            'q_ven_pul_0' : 0.0}           # initial pulmonary venous flux


def param():

    # parameters in kg-mm-s unit system

    R_ar_sys = 120.0e-6              # systemic arterial resistance
    tau_ar_sys = 1.0311433159        # systemic arterial Windkessel time constant
    tau_ar_pul = 0.3                 # pulmonary arterial resistance

    # Diss Hirschvogel tab. 2.7
    C_ar_sys = tau_ar_sys/R_ar_sys   # systemic arterial compliance
    Z_ar_sys = R_ar_sys/20.          # systemic arterial characteristic impedance
    R_ven_sys = R_ar_sys/5.          # systemic venous resistance
    C_ven_sys = 30.*C_ar_sys         # systemic venous compliance
    R_ar_pul = R_ar_sys/8.           # pulmonary arterial resistance
    C_ar_pul = tau_ar_pul/R_ar_pul   # pulmonary arterial compliance
    R_ven_pul = R_ar_pul             # pulmonary venous resistance
    C_ven_pul = 2.5*C_ar_pul         # pulmonary venous resistance

    L_ar_sys = 0.667e-6              # systemic arterial inertance
    L_ven_sys = 0.                   # systemic venous inertance
    L_ar_pul = 0.                    # pulmonary arterial inertance
    L_ven_pul = 0.                   # pulmonary venous inertance

    # timings
    t_ed = 0.2                       # end-diastolic time
    t_es = 0.53                      # end-systolic time
    T_cycl = 1.0                     # cardiac cycle time

    # atrial elastances
    E_at_max_l = 2.9e-5              # maximum left atrial elastance
    E_at_min_l = 9.0e-6              # minimum left atrial elastance
    E_at_max_r = 1.8e-5              # maximum right atrial elastance
    E_at_min_r = 8.0e-6              # minimum right atrial elastance
    # ventricular elastances - NOT used in this example, since the ventricles are 3D bodies
    E_v_max_l = 30.0e-5              # maximum left ventricular elastance
    E_v_min_l = 12.0e-6              # minimum left ventricular elastance
    E_v_max_r = 20.0e-5              # maximum right ventricular elastance
    E_v_min_r = 10.0e-6              # minimum right ventricular elastance


    return {'R_ar_sys' : R_ar_sys,
            'C_ar_sys' : C_ar_sys,
            'L_ar_sys' : L_ar_sys,
            'Z_ar_sys' : Z_ar_sys,
            'R_ar_pul' : R_ar_pul,
            'C_ar_pul' : C_ar_pul,
            'L_ar_pul' : L_ar_pul,
            'R_ven_sys' : R_ven_sys,
            'C_ven_sys' : C_ven_sys,
            'L_ven_sys' : L_ven_sys,
            'R_ven_pul' : R_ven_pul,
            'C_ven_pul' : C_ven_pul,
            'L_ven_pul' : L_ven_pul,
            # atrial elastances
            'E_at_max_l' : E_at_max_l,
            'E_at_min_l' : E_at_min_l,
            'E_at_max_r' : E_at_max_r,
            'E_at_min_r' : E_at_min_r,
            # ventricular elastances
            'E_v_max_l' : E_v_max_l,
            'E_v_min_l' : E_v_min_l,
            'E_v_max_r' : E_v_max_r,
            'E_v_min_r' : E_v_min_r,
            # valve resistances
            'R_vin_l_min' : 1.0e-6,  # mitral valve open resistance
            'R_vin_l_max' : 1.0e1,   # mitral valve closed resistance
            'R_vout_l_min' : 1.0e-6, # aortic valve open resistance
            'R_vout_l_max' : 1.0e1,  # aortic valve closed resistance
            'R_vin_r_min' : 1.0e-6,  # tricuspid valve open resistance
            'R_vin_r_max' : 1.0e1,   # tricuspid valve closed resistance
            'R_vout_r_min' : 1.0e-6, # pulmonary valve open resistance
            'R_vout_r_max' : 1.0e1,  # pulmonary valve closed resistance
            # timings
            't_ed' : t_ed,
            't_es' : t_es,
            'T_cycl' : T_cycl,
            # unstressed compartment volumes (only for post-processing, since 0D model is formulated in fluxes = dVolume/dt)
            'V_at_l_u' : 5e3,       # unstressed left atrial volume
            'V_at_r_u' : 4e3,       # unstressed right atrial volume
            'V_v_l_u' : 10e3,       # unstressed left ventricular volume - NOT used here, since from 3D
            'V_v_r_u' : 8e3,        # unstressed right ventricular volume - NOT used here, since from 3D
            'V_ar_sys_u' : 611e3,   # unstressed systemic arterial volume
            'V_ar_pul_u' : 123e3,   # unstressed pulmonary arterial volume
            'V_ven_sys_u' : 2596e3, # unstressed systemic venous volume
            'V_ven_pul_u' : 120e3}  # unstressed pulmonary venous volume




if __name__ == "__main__":

    main()
